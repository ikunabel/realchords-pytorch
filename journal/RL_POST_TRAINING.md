# RL post-training equips the actor with new properties

Notes on how PPO + KL distillation from the anchor (teacher) change behavior relative to
MLE-only decoder training in ReaLchords / ReaLJam.

---

## Anticipation

**KL penalty in reward** pulls the student's action log-probs towards the teacher's log-probs
(knowledge distillation).

→ By coming closer to the teacher's policy, the student learns to predict ahead →
**anticipation**.

---

## Adaptation

The actor conditions on its own past chord predictions and the fixed melody conditions.

The trajectory sampled from the actor's policy can contain **mispredicted chords**.

→ The policy **adapts to the unfolding trajectory**.

---

## Actor / student (MLE-trained decoder)

$$\pi_\theta(c_t \mid m_{<t}, c_{<t})$$

- MLE only trained on GT data: every token correct, no bad prefixes.
- At inference time, the human can play an unexpected melody or the model can wrongly predict chords.
  → **OOD states**, **exposure bias**.
  → Errors **compound autoregressively**, because later steps are conditioned on a broken context.

**One causal self-attention:** Q, K, V come from the interleaved prefix.

- Wrong chords corrupt later K and V.
  → The actor predicts future tokens according to wrong / corrupted context.
  → The actor **cannot recover**.

---

## Anchor / teacher (MLE-trained encoder–decoder)

$$\psi_\omega(c_t \mid m, c_{<t})$$

PPO trains on bad prefixes.

- Training signal richer than next-token prediction.
- PPO updates the policy on prefixes containing mispredicted chords.
  → Learns a policy over states that are actually visited during inference, not only dataset states.

**Q** from self-attention (chords), but **K, V in cross-attention** come from uncorrupted past-,
current, and **future melody**.

→ The anchor can anticipate future melody; the prediction of the next token is conditioned not
only on corrupted chord history, but on the **uncorrupted full melody**.

→ The anchor can **reset and recover**.
