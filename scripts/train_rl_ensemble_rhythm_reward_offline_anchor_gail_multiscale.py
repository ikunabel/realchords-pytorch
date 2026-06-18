#!/usr/bin/env python3
"""GAPT RL training with multiscale contrastive reward (legacy + sliding windows)."""

import os
from copy import deepcopy
from pathlib import Path

import argbind
import math
import torch

torch.set_float32_matmul_precision("high")

from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.lit_module.contrastive_reward_rhythm import LitContrastiveRewardRhythm
from realchords.lit_module.contrastive_reward_segment import LitContrastiveRewardSegment
from realchords.lit_module.decoder_only import LitDecoder
from realchords.lit_module.discriminative_reward import LitDiscriminativeReward
from realchords.lit_module.discriminative_reward_rhythm import (
    LitDiscriminativeRewardRhythm,
)
from realchords.lit_module.enc_dec import LitEncoderDecoder
from realchords.model.reward_model import DiscriminativeReward
from realchords.rl.actor import DecoderSingleAgentActor, EncoderDecoderOfflineAnchor
from realchords.rl.critic import Critic
from realchords.rl.deepspeed import get_strategy
from realchords.rl.experience_maker import ExperienceMaker
from realchords.rl.openrlhf_local import AdaptiveKLController, FixedKLController
from realchords.rl.reward.model_based_rewards import (
    ContrastiveRewardRhythmFn,
    DiscriminativeRewardFn,
    DiscriminativeRewardRhythmFn,
    GAILDiscriminativeRewardFn,
)
from realchords.rl.reward.multiscale_contrastive_rewards import (
    MultiscaleContrastiveRewardFn,
)
from realchords.rl.reward.rule_based_rewards import (
    EarlyStopPenalty,
    InvalidOutputPenalty,
    LongNotePenalty,
    RepetitionPenalty,
    SilencePenalty,
)
from realchords.rl.rl_trainer import ReaLchordsGAILPPOTrainer
from realchords.rl.utils import ModelPreparer
from realchords.utils.inference_utils import load_lit_model, prepare_model_for_deepspeed
from realchords.utils.lr_scheduler import LinearWarmupCosineDecay
from realchords.utils.train_utils import AttrDict


@argbind.bind(without_prefix=True)
def main(args, save_dir: str = ""):
    if not save_dir:
        raise ValueError("save_dir must be provided.")
    args.save_dir = save_dir
    args.wandb_run_name = Path(save_dir).name

    strategy = get_strategy(args)
    strategy.setup_distributed()

    target_dtype = torch.bfloat16 if getattr(args, "bf16", False) else None

    model, tokenizer, dataloaders = load_lit_model(
        model_path=args.pretrain_model_path,
        lit_module_cls=LitDecoder,
        batch_size=args.rollout_batch_size,
        compile=False,
        override_args={
            "DecoderTransformer.dropout": 0.0,
            **args.lit_module_override_args,
        },
    )
    train_dataloader, val_dataloader = dataloaders
    model = prepare_model_for_deepspeed(model, target_dtype)

    actor = DecoderSingleAgentActor(
        model,
        tokenizer=tokenizer,
        model_part=args.model_part,
        max_seq_len=model.max_seq_len - 2,
    )
    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    critic = Critic(deepcopy(model))
    strategy.print(actor)
    strategy.print(critic)

    anchor_model, _, _ = load_lit_model(
        model_path=args.anchor_model_path,
        lit_module_cls=LitEncoderDecoder,
        batch_size=args.rollout_batch_size,
        compile=False,
        override_args={
            "EncoderDecoderTransformer.dropout": 0.0,
            **args.lit_module_override_args,
        },
    )
    anchor_model = prepare_model_for_deepspeed(anchor_model, target_dtype)
    initial_model = EncoderDecoderOfflineAnchor(
        anchor_model,
        bos_token_id=tokenizer.bos_token,
        eos_token_id=tokenizer.eos_token,
        pad_token_id=tokenizer.pad_token,
        max_seq_len=model.max_seq_len // 2 + 1,
    )

    gail_reward_model = DiscriminativeReward(
        dim=args.gail_discriminative_model_configs["dim"],
        depth=args.gail_discriminative_model_configs["depth"],
        heads=args.gail_discriminative_model_configs["heads"],
        num_tokens=tokenizer.num_tokens,
        max_seq_len=model.max_seq_len // 2 + 1,
        dropout=args.gail_discriminative_model_configs["dropout"],
    )

    actor_optim = strategy.create_optimizer(
        actor,
        lr=args.actor_learning_rate,
        betas=args.adam_betas,
        weight_decay=args.l2,
    )
    critic_optim = strategy.create_optimizer(
        critic,
        lr=args.critic_learning_rate,
        betas=args.adam_betas,
        weight_decay=args.l2,
    )
    reward_optim = strategy.create_optimizer(
        gail_reward_model,
        lr=args.gail_reward_learning_rate,
        betas=args.adam_betas,
        weight_decay=args.l2,
    )

    num_backward_per_episode = args.rollout_batch_size // args.train_batch_size
    max_steps = args.num_steps * num_backward_per_episode
    if isinstance(args.warmup_steps, float):
        num_warmup_steps = math.ceil(max_steps * args.warmup_steps)
    elif isinstance(args.warmup_steps, int):
        num_warmup_steps = args.warmup_steps * num_backward_per_episode
    else:
        raise ValueError("warmup_steps must be either float or int.")

    actor_scheduler = LinearWarmupCosineDecay(
        optimizer=actor_optim,
        warmup_iters=num_warmup_steps,
        total_iters=max_steps,
        eta_min=args.actor_learning_rate * 0.1,
    )
    critic_scheduler = LinearWarmupCosineDecay(
        optimizer=critic_optim,
        warmup_iters=num_warmup_steps,
        total_iters=max_steps,
        eta_min=args.critic_learning_rate * 0.1,
    )
    reward_scheduler = LinearWarmupCosineDecay(
        optimizer=reward_optim,
        warmup_iters=num_warmup_steps,
        total_iters=max_steps,
        eta_min=args.gail_reward_learning_rate * 0.1,
    )

    legacy_contrastive_models = [
        prepare_model_for_deepspeed(
            load_lit_model(
                model_path,
                lit_module_cls=LitContrastiveReward,
                compile=False,
                return_only_model=True,
            ),
            target_dtype,
        )
        for model_path in args.contrastive_reward_model_path
    ]

    if len(args.multiscale_contrastive_reward_model_path) != len(
        args.multiscale_contrastive_window_lens
    ):
        raise ValueError(
            "multiscale_contrastive_reward_model_path and "
            "multiscale_contrastive_window_lens must have the same length."
        )

    multiscale_contrastive_models = [
        prepare_model_for_deepspeed(
            load_lit_model(
                model_path,
                lit_module_cls=LitContrastiveRewardSegment,
                compile=False,
                return_only_model=True,
            ),
            target_dtype,
        )
        for model_path in args.multiscale_contrastive_reward_model_path
    ]

    discriminative_reward_models = [
        prepare_model_for_deepspeed(
            load_lit_model(
                model_path,
                lit_module_cls=LitDiscriminativeReward,
                compile=False,
                return_only_model=True,
            ),
            target_dtype,
        )
        for model_path in args.discriminative_reward_model_path
    ]
    contrastive_reward_rhythm_models = [
        prepare_model_for_deepspeed(
            load_lit_model(
                model_path,
                lit_module_cls=LitContrastiveRewardRhythm,
                compile=False,
                return_only_model=True,
            ),
            target_dtype,
        )
        for model_path in args.contrastive_reward_rhythm_model_path
    ]
    discriminative_reward_rhythm_models = [
        prepare_model_for_deepspeed(
            load_lit_model(
                model_path,
                lit_module_cls=LitDiscriminativeRewardRhythm,
                compile=False,
                return_only_model=True,
            ),
            target_dtype,
        )
        for model_path in args.discriminative_reward_rhythm_model_path
    ]

    preparer = ModelPreparer(strategy)
    preparer.add_trainable("actor", actor, actor_optim, actor_scheduler)
    preparer.add_trainable("critic", critic, critic_optim, critic_scheduler)
    preparer.add_trainable("gail_reward", gail_reward_model, reward_optim, reward_scheduler)
    preparer.add_model("initial_model", initial_model)
    preparer.add_model_list("legacy_contrastive_rewards", legacy_contrastive_models)
    preparer.add_model_list(
        "multiscale_contrastive_rewards", multiscale_contrastive_models
    )
    preparer.add_model_list("discriminative_rewards", discriminative_reward_models)
    preparer.add_model_list(
        "contrastive_rhythm_rewards", contrastive_reward_rhythm_models
    )
    preparer.add_model_list(
        "discriminative_rhythm_rewards", discriminative_reward_rhythm_models
    )

    models = preparer.prepare(is_rlhf=True)

    actor, actor_optim, actor_scheduler = models.get_trainable("actor")
    critic, critic_optim, critic_scheduler = models.get_trainable("critic")
    gail_reward_model, reward_optim, reward_scheduler = models.get_trainable("gail_reward")
    initial_model = models.get_model("initial_model")
    legacy_contrastive_models = models.get_model_list("legacy_contrastive_rewards")
    multiscale_contrastive_models = models.get_model_list("multiscale_contrastive_rewards")
    discriminative_reward_models = models.get_model_list("discriminative_rewards")
    contrastive_reward_rhythm_models = models.get_model_list("contrastive_rhythm_rewards")
    discriminative_reward_rhythm_models = models.get_model_list(
        "discriminative_rhythm_rewards"
    )

    gail_reward_fn = GAILDiscriminativeRewardFn(
        model=gail_reward_model,
        pad_token_id=tokenizer.pad_token,
        bos_token_id=tokenizer.bos_token,
        eos_token_id=tokenizer.eos_token,
        model_part=args.model_part,
        reward_formulation=getattr(args, "gail_reward_formulation", "prob"),
    )

    multiscale_contrastive_reward_fn = MultiscaleContrastiveRewardFn(
        legacy_models=legacy_contrastive_models,
        multiscale_models=multiscale_contrastive_models,
        window_lens=args.multiscale_contrastive_window_lens,
        pad_token_id=tokenizer.pad_token,
        bos_token_id=tokenizer.bos_token,
        eos_token_id=tokenizer.eos_token,
        model_part=args.model_part,
    )

    reward_configs = []

    for i, discriminative_reward_model in enumerate(discriminative_reward_models):
        reward_configs.append(
            {
                "reward_fn": DiscriminativeRewardFn(
                    model=discriminative_reward_model,
                    pad_token_id=tokenizer.pad_token,
                    bos_token_id=tokenizer.bos_token,
                    eos_token_id=tokenizer.eos_token,
                    model_part=args.model_part,
                ),
                "weight": getattr(args, "discriminative_reward_weight", 1.0),
                "name": f"discriminative_reward_{i}",
                "clip_range": getattr(args, "discriminative_reward_clip_range", None),
            }
        )

    for i, contrastive_reward_rhythm_model in enumerate(
        contrastive_reward_rhythm_models
    ):
        reward_configs.append(
            {
                "reward_fn": ContrastiveRewardRhythmFn(
                    model=contrastive_reward_rhythm_model,
                    tokenizer=tokenizer,
                    model_part=args.model_part,
                ),
                "weight": args.contrastive_reward_rhythm_weight,
                "name": f"contrastive_reward_rhythm_{i}",
                "clip_range": getattr(
                    args, "contrastive_reward_rhythm_clip_range", None
                ),
            }
        )

    for i, discriminative_reward_rhythm_model in enumerate(
        discriminative_reward_rhythm_models
    ):
        reward_configs.append(
            {
                "reward_fn": DiscriminativeRewardRhythmFn(
                    model=discriminative_reward_rhythm_model,
                    tokenizer=tokenizer,
                    model_part=args.model_part,
                ),
                "weight": args.discriminative_reward_rhythm_weight,
                "name": f"discriminative_reward_rhythm_{i}",
                "clip_range": getattr(
                    args, "discriminative_reward_rhythm_clip_range", None
                ),
            }
        )

    reward_configs.extend(
        [
            {
                "reward_fn": EarlyStopPenalty(
                    pad_token_id=tokenizer.pad_token,
                    bos_token_id=tokenizer.bos_token,
                    eos_token_id=tokenizer.eos_token,
                ),
                "weight": getattr(args, "early_stop_penalty_weight", 1.0),
                "name": "early_stop_penalty",
            },
            {
                "reward_fn": InvalidOutputPenalty(
                    tokenizer=tokenizer,
                    model_part=args.model_part,
                ),
                "weight": args.invalid_output_penalty_weight,
                "name": "invalid_output_penalty",
            },
            {
                "reward_fn": SilencePenalty(tokenizer=tokenizer),
                "weight": getattr(args, "silence_penalty_weight", 1.0),
                "name": "silence_penalty",
            },
            {
                "reward_fn": LongNotePenalty(tokenizer=tokenizer),
                "weight": getattr(args, "long_note_penalty_weight", 0.0),
                "name": "long_note_penalty",
            },
            {
                "reward_fn": RepetitionPenalty(
                    tokenizer=tokenizer,
                    model_part=args.model_part,
                    threshold=getattr(args, "repetition_penalty_threshold", 4),
                ),
                "weight": getattr(args, "repetition_penalty_weight", 0.0),
                "name": "repetition_penalty",
            },
        ]
    )

    reward_configs.append(
        {
            "reward_fn": multiscale_contrastive_reward_fn,
            "weight": getattr(args, "contrastive_reward_weight", 1.0),
            "name": "multiscale_contrastive_reward",
            "clip_range": getattr(args, "contrastive_reward_clip_range", None),
        }
    )

    reward_configs.append(
        {
            "reward_fn": gail_reward_fn,
            "weight": getattr(args, "gail_reward_weight", 1.0),
            "name": "gail_reward",
            "clip_range": getattr(args, "gail_reward_clip_range", None),
        }
    )

    init_kl_coef = args.init_kl_coef
    kl_target = args.kl_target
    kl_horizon = args.kl_horizon
    if args.kl_target:
        kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
    else:
        kl_ctl = FixedKLController(init_kl_coef)

    experience_maker = ExperienceMaker(
        actor=actor,
        critic=critic,
        reward_configs=reward_configs,
        initial_model=initial_model,
        tokenizer=tokenizer,
        kl_controller=kl_ctl,
        strategy=strategy,
        reward_vram_swap=args.reward_vram_swap,
        logits_vram_swap=args.logits_vram_swap,
        trainable_reward_names=["gail_reward"],
    )

    ema_model = None
    os.makedirs(save_dir, exist_ok=True)

    trainer = ReaLchordsGAILPPOTrainer(
        strategy,
        experience_maker=experience_maker,
        kl_ctl=kl_ctl,
        actor=actor,
        critic=critic,
        initial_model=initial_model,
        ema_model=ema_model,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        max_norm=args.max_norm,
        limit_eval_batches=args.limit_eval_batches,
        max_log_examples=args.max_log_examples,
        counterpart_prediction_loss_coef=getattr(
            args, "counterpart_prediction_loss_coef", 0.0
        ),
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        buffer_cpu_offload=args.buffer_cpu_offload,
        dataloader_pin_memory=args.dataloader_pin_memory,
        trainer_empty_cache=args.trainer_empty_cache,
        reward_fn_name="gail_reward",
        reward_optim=reward_optim,
        reward_scheduler=reward_scheduler,
        reward_update_steps=args.reward_update_steps,
        enable_reward_label_smoothing=getattr(
            args, "enable_reward_label_smoothing", False
        ),
        reward_update_early_stop_steps=getattr(
            args, "reward_update_early_stop_steps", None
        ),
        reward_update_strategy=getattr(args, "reward_update_strategy", "steps"),
        reward_average_steps=getattr(args, "reward_average_steps", None),
        reward_update_threshold=getattr(args, "reward_update_threshold", None),
        reward_apply_threshold_after_steps=getattr(
            args, "reward_apply_threshold_after_steps", None
        ),
    )

    trainer.fit(args, train_dataloader, val_dataloader)

    if strategy.is_rank_0():
        torch.save(actor.state_dict(), Path(save_dir) / "actor.pth")
        if args.save_value_network:
            torch.save(critic.state_dict(), Path(save_dir) / "critic.pth")


if __name__ == "__main__":
    args = argbind.parse_args()
    argbind.dump_args(args, Path(args["save_dir"]) / "args.yml")
    with argbind.scope(args):
        args = AttrDict(args)
        main(args)
