# Bugfix for realchords_opensource

This document contains the exact code changes made to fix the `bfloat16` mismatch, missing `num_steps` argument, and inconsistent checkpoint loading issues in the ReaLchords RL training pipeline.

You can safely copy the following unified patch content and apply it to the `realchords_opensource` repository using `git am` or `git apply`.

## Patch Content
```diff
From 4e2d2ab58a91347b5131def80bc1c6a319dc3fad Mon Sep 17 00:00:00 2001
From: Yusong Wu <wuyusongwys@gmail.com>
Date: Sat, 14 Mar 2026 10:39:53 +0000
Subject: [PATCH] fix dtype mismatch and config attribute bug

---
 realchords/model/gen_model.py                 | 56 +++++++++++--------
 realchords/model/reward_model.py              | 22 +++++---
 realchords/rl/actor.py                        | 24 +++++---
 realchords/utils/inference_utils.py           | 24 ++++++--
 ...alchords_ensemble_rhythm_offline_anchor.py |  5 +-
 5 files changed, 85 insertions(+), 46 deletions(-)

diff --git a/realchords/model/gen_model.py b/realchords/model/gen_model.py
index da6a9d2..8273442 100644
--- a/realchords/model/gen_model.py
+++ b/realchords/model/gen_model.py
@@ -77,7 +77,11 @@ class DecoderTransformer(AutoregressiveWrapper):
         return self.decoder
 
     def forward(self, x, mask=None, **kwargs):
-        return self.decoder(x, mask=mask, **kwargs)
+        config = getattr(self, "config", None)
+        is_bf16 = getattr(config, "bf16", False) if config else False
+        target_dtype = torch.bfloat16 if is_bf16 else torch.float16
+        with torch.autocast(device_type="cuda", dtype=target_dtype):
+            return self.decoder(x, mask=mask, **kwargs)
 
     @torch.no_grad()
     @torch.jit.export
@@ -470,29 +474,33 @@ class EncoderDecoderTransformer(nn.Module):
         dec_mask=None,
         return_attn_z_loss=False,
     ):
-        if return_attn_z_loss:
-            enc, cache = self.encoder(
-                x_enc,
-                mask=enc_mask,
-                return_embeddings=True,
-                return_attn_z_loss=True,
-            )
-            z_loss_enc = cache.attn_z_loss
-            dec, cache = self.decoder(
-                x_dec,
-                context=enc,
-                context_mask=enc_mask,
-                mask=dec_mask,
-                return_attn_z_loss=True,
-            )
-            z_loss_dec = cache.attn_z_loss
-            return dec, z_loss_enc + z_loss_dec
-        else:
-            enc = self.encoder(x_enc, mask=enc_mask, return_embeddings=True)
-            dec = self.decoder(
-                x_dec, context=enc, context_mask=enc_mask, mask=dec_mask
-            )
-            return dec
+        config = getattr(self, "config", None)
+        is_bf16 = getattr(config, "bf16", False) if config else False
+        target_dtype = torch.bfloat16 if is_bf16 else torch.float16
+        with torch.autocast(device_type="cuda", dtype=target_dtype):
+            if return_attn_z_loss:
+                enc, cache = self.encoder(
+                    x_enc,
+                    mask=enc_mask,
+                    return_embeddings=True,
+                    return_attn_z_loss=True,
+                )
+                z_loss_enc = cache.attn_z_loss
+                dec, cache = self.decoder(
+                    x_dec,
+                    context=enc,
+                    context_mask=enc_mask,
+                    mask=dec_mask,
+                    return_attn_z_loss=True,
+                )
+                z_loss_dec = cache.attn_z_loss
+                return dec, z_loss_enc + z_loss_dec
+            else:
+                enc = self.encoder(x_enc, mask=enc_mask, return_embeddings=True)
+                dec = self.decoder(
+                    x_dec, context=enc, context_mask=enc_mask, mask=dec_mask
+                )
+                return dec
 
     @torch.no_grad()
     def generate(
diff --git a/realchords/model/reward_model.py b/realchords/model/reward_model.py
index e92a65b..39f82b5 100644
--- a/realchords/model/reward_model.py
+++ b/realchords/model/reward_model.py
@@ -87,9 +87,13 @@ class ContrastiveReward(Module):
         melody_mask: Tensor = None,
     ) -> Tensor:
         """Forward pass."""
-        chord_embed = self.get_chord_embed(chord, chord_mask)
-        melody_embed = self.get_melody_embed(melody, melody_mask)
-        return chord_embed, melody_embed, self.logit_scale.exp()
+        config = getattr(self, "config", None)
+        is_bf16 = getattr(config, "bf16", False) if config else False
+        target_dtype = torch.bfloat16 if is_bf16 else torch.float16
+        with torch.autocast(device_type="cuda", dtype=target_dtype):
+            chord_embed = self.get_chord_embed(chord, chord_mask)
+            melody_embed = self.get_melody_embed(melody, melody_mask)
+            return chord_embed, melody_embed, self.logit_scale.exp()
 
 
 class DiscriminativeReward(Module):
@@ -124,7 +128,11 @@ class DiscriminativeReward(Module):
         input_mask: Tensor = None,
     ) -> Tensor:
         """Forward pass."""
-        logits = self.out_proj(self.encoder(input, mask=input_mask))
-        # use the first token as output
-        logits = logits[:, 0]
-        return logits
+        config = getattr(self, "config", None)
+        is_bf16 = getattr(config, "bf16", False) if config else False
+        target_dtype = torch.bfloat16 if is_bf16 else torch.float16
+        with torch.autocast(device_type="cuda", dtype=target_dtype):
+            logits = self.out_proj(self.encoder(input, mask=input_mask))
+            # use the first token as output
+            logits = logits[:, 0]
+            return logits
diff --git a/realchords/rl/actor.py b/realchords/rl/actor.py
index e5cedef..b7b5855 100644
--- a/realchords/rl/actor.py
+++ b/realchords/rl/actor.py
@@ -32,6 +32,7 @@ class ReaLchordsActor(Actor):
         eos_token_id: int,
         pad_token_id: int,
         max_seq_len: int,
+        bf16: bool = False,
     ) -> None:
         """Compared to openrlhf Actor class, we remove the following supports and features:
 
@@ -49,6 +50,7 @@ class ReaLchordsActor(Actor):
         self.eos_token_id = eos_token_id
         self.pad_token_id = pad_token_id
         self.max_seq_len = max_seq_len
+        self.bf16 = bf16
 
     @torch.no_grad()
     def generate(
@@ -89,7 +91,9 @@ class ReaLchordsActor(Actor):
         Compared to the original forward, remove the creation of position_ids.
         """
 
-        logits = self.model(sequences, mask=attention_mask)
+        target_dtype = torch.bfloat16 if getattr(self, "bf16", False) else torch.float16
+        with torch.autocast(device_type="cuda", dtype=target_dtype):
+            logits = self.model(sequences, mask=attention_mask)
 
         if num_actions is None:
             assert return_output
@@ -183,6 +187,7 @@ class DecoderSingleAgentActor(ReaLchordsActor):
         tokenizer: HooktheoryTokenizer,
         max_seq_len: int,
         model_part: str,
+        bf16: bool = False,
     ) -> None:
         """Compared to the base class, we add the tokenizer."""
         # skip the super call, directly call the Actor class's __init__
@@ -196,6 +201,7 @@ class DecoderSingleAgentActor(ReaLchordsActor):
         self.pad_token_id = tokenizer.pad_token
         self.silence_token_id = tokenizer.silence_token
         self.model_part = model_part
+        self.bf16 = bf16
         self.init_filter_fn()
 
     def init_filter_fn(self):
@@ -356,13 +362,15 @@ class EncoderDecoderOfflineAnchor(ReaLchordsActor):
             self.get_inputs_from_sequence(sequences)
         )
 
-        # In x-transformers, mask is True for unmasked tokens
-        model_part_logits = self.model(
-            context_tokens,
-            model_tokens,
-            enc_mask=context_mask,
-            dec_mask=model_mask,
-        )
+        target_dtype = torch.bfloat16 if getattr(self, "bf16", False) else torch.float16
+        with torch.autocast(device_type="cuda", dtype=target_dtype):
+            # In x-transformers, mask is True for unmasked tokens
+            model_part_logits = self.model(
+                context_tokens,
+                model_tokens,
+                enc_mask=context_mask,
+                dec_mask=model_mask,
+            )
 
         # remove the EOS token logits
         model_part_logits = model_part_logits[:, :-1, :]
diff --git a/realchords/utils/inference_utils.py b/realchords/utils/inference_utils.py
index d5d0309..0e77f25 100644
--- a/realchords/utils/inference_utils.py
+++ b/realchords/utils/inference_utils.py
@@ -44,10 +44,22 @@ def load_lit_model(
         model_path, weights_only=True, map_location=torch.device("cpu")
     )["state_dict"]
 
-    if not args["compile"]:
-        state_dict = {
-            k.replace("._orig_mod", ""): v for k, v in state_dict.items()
-        }
+    clean_state_dict = {
+        k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state_dict.items()
+    }
+
+    if getattr(lit_module, "compile", False) or args.get("compile", False):
+        target_keys = lit_module.state_dict().keys()
+        final_state_dict = {}
+        for k in target_keys:
+            clean_k = k.replace("_orig_mod.", "").replace("._orig_mod", "")
+            if clean_k in clean_state_dict:
+                final_state_dict[k] = clean_state_dict[clean_k]
+            elif k in state_dict:
+                final_state_dict[k] = state_dict[k]
+        state_dict = final_state_dict
+    else:
+        state_dict = clean_state_dict
 
     lit_module.load_state_dict(state_dict)
     model = lit_module.model
@@ -82,7 +94,7 @@ def load_rl_model(
     state_dict = torch.load(
         model_path, weights_only=True, map_location=torch.device("cpu")
     )
-    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
+    state_dict = {k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state_dict.items()}
     state_dict_model = {}
     for k, v in state_dict.items():
         if k.startswith("model.module."):
@@ -103,7 +115,7 @@ def load_model_state_dict_from_lit_checkpoint(
     state_dict = torch.load(
         model_path, weights_only=True, map_location=torch.device("cpu")
     )
-    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
+    state_dict = {k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state_dict.items()}
     return state_dict
 
 
diff --git a/scripts/train_rl_realchords_ensemble_rhythm_offline_anchor.py b/scripts/train_rl_realchords_ensemble_rhythm_offline_anchor.py
index b682e19..3cbbeaf 100644
--- a/scripts/train_rl_realchords_ensemble_rhythm_offline_anchor.py
+++ b/scripts/train_rl_realchords_ensemble_rhythm_offline_anchor.py
@@ -52,10 +52,11 @@ from realchords.utils.train_utils import AttrDict
 
 
 @argbind.bind(without_prefix=True)
-def main(args, save_dir: str = ""):
+def main(args, save_dir: str = "", num_steps: int = 1000):
     if not save_dir:
         raise ValueError("save_dir must be provided.")
     args.save_dir = save_dir
+    args.num_steps = num_steps
     args.wandb_run_name = Path(save_dir).name
     # configure strategy
     strategy = get_strategy(args)
@@ -78,6 +79,7 @@ def main(args, save_dir: str = ""):
         tokenizer=tokenizer,
         model_part=args.model_part,
         max_seq_len=model.max_seq_len - 2,  # -2 for bos and eos
+        bf16=getattr(args, "bf16", False),
     )
 
     if args.actor_init_on_gpu:
@@ -106,6 +108,7 @@ def main(args, save_dir: str = ""):
         eos_token_id=tokenizer.eos_token,
         pad_token_id=tokenizer.pad_token,
         max_seq_len=model.max_seq_len // 2 + 1,  # doesn't matter
+        bf16=getattr(args, "bf16", False),
     )
 
     # configure optimizer
-- 
2.25.1

```
