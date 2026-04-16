"""Utilities for load model and inference with a trained model."""

import copy
import torch
from typing import Optional
from pathlib import Path
import argbind

from realchords.base_trainer import BaseLightningModel
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.logging_utils import logger


def load_lit_model(
    model_path: str,
    lit_module_cls: BaseLightningModel,
    batch_size: Optional[int] = None,
    override_args: Optional[dict] = None,
    compile: bool = True,
    return_only_model: bool = False,
    return_lit_module: bool = False,
):
    """Load model and dataloader via lightning module from a given model path."""
    model_path = Path(model_path)
    experiment_dir = model_path.parent

    logger.info(f"Loading model from {model_path}")

    args = argbind.load_args(experiment_dir / "args.yml")
    if batch_size:
        # If batch_size is provided, override the one in the args
        args["batch_size"] = batch_size

    if not compile:
        args["compile"] = False

    if override_args:
        args.update(override_args)

    with argbind.scope(args):
        lit_module = lit_module_cls()
        train_dataloader, val_dataloader = lit_module.get_dataloaders()

    logger.info(f"Loading checkpoint weights (torch.load) from {model_path}")
    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )["state_dict"]
    logger.info(f"Finished torch.load for {model_path}")

    clean_state_dict = {
        k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state_dict.items()
    }

    if getattr(lit_module, "compile", False) or args.get("compile", False):
        target_keys = lit_module.state_dict().keys()
        final_state_dict = {}
        for k in target_keys:
            clean_k = k.replace("_orig_mod.", "").replace("._orig_mod", "")
            if clean_k in clean_state_dict:
                final_state_dict[k] = clean_state_dict[clean_k]
            elif k in state_dict:
                final_state_dict[k] = state_dict[k]
        state_dict = final_state_dict
    else:
        state_dict = clean_state_dict

    lit_module.load_state_dict(state_dict)
    model = lit_module.model
    model.config = args  # Add config to model for saving

    tokenizer = lit_module.tokenizer
    dataloaders = train_dataloader, val_dataloader
    if return_only_model and return_lit_module:
        raise ValueError(
            "Only one of return_only_model and return_lit_module can be True."
        )
    if return_only_model:
        return model
    if return_lit_module:
        return lit_module

    return model, tokenizer, dataloaders


def load_rl_model(
    model_path: str,
    model: torch.nn.Module,
    compile: bool = True,
):
    """Load RL-finetuned model from a given model path."""
    # Note that the model input should not be compiled.
    # check if the model is compiled
    if list(model.state_dict().keys())[0].startswith("_orig_mod"):
        raise ValueError("The model should not be compiled.")

    model_path = Path(model_path)
    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    state_dict = {k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state_dict.items()}
    state_dict_model = {}
    for k, v in state_dict.items():
        if k.startswith("model.module."):
            k = k.replace("model.module.", "")
            state_dict_model[k] = v
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(state_dict_model)
    if compile:
        new_model = torch.compile(new_model)
    return new_model


def load_model_state_dict_from_lit_checkpoint(
    model_path: str,
):
    """Load model state dict from a given pytorch lightning model checkpoint path."""
    model_path = Path(model_path)
    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    state_dict = {k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state_dict.items()}
    return state_dict


def prepare_model_for_deepspeed(model, target_dtype=None):
    """Convert model dtype before DeepSpeed prepare.

    DeepSpeed converts weights at initialize() time, but torch.compile may
    trace the graph before that conversion, causing dtype mismatches in
    compiled kernels. Converting dtype explicitly before DeepSpeed avoids this.

    Args:
        model: The model to prepare.
        target_dtype: Target dtype (e.g., torch.bfloat16). None keeps original.
    """
    if target_dtype is not None:
        model = model.to(target_dtype)
    return model


def load_gen_model_from_state_dict(
    state_dict_path: str,
    model_cls: torch.nn.Module,
    compile: bool = True,
    override_args: Optional[dict] = None,
):
    """Load generative model in ReaLchords from a given state dict path."""
    state_dict = torch.load(
        state_dict_path, weights_only=True, map_location=torch.device("cpu")
    )
    model_cls = argbind.bind(model_cls)
    args = argbind.load_args(Path(state_dict_path).parent / "args.yml")
    if override_args:
        args.update(override_args)
    with argbind.scope(args):
        model = model_cls()
    model.load_state_dict(state_dict)
    if compile:
        model = torch.compile(model)
    return model
