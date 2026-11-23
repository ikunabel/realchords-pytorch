"""How much probability do the models put on chord tokens a lead-sheet vocabulary would merge away?

Mergeable tokens (journal reference/CHORD_CONVERSION_DB.md, "Lead-sheet-oriented tokens"): names that are one of
note_seq's 29 chord types on the same root (and slash bass) with notes left out ("omissions", e.g.
Ebped(addb7)(add3) = Eb7 without its 5th), or the same notes in a clumsier spelling ("respellings", e.g.
Dbm9(add6)/Bb = Dbm9/Bb). Teacher-forced on test songs: at every frame, the model's next-chord distribution; we
report the mergeable onset tokens' share of the chord-onset probability (what sampling a chord change draws from).

    python scripts/eval/measure_mergeable_token_mass.py [n_songs]
"""
import sys, types
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "eval"), str(REPO / "scripts" / "extract_voicings")]
import analyze_chord_onset_timing as A  # noqa: E402
from fit_voicings_to_chords import chord_rule  # noqa: E402
from note_seq import chord_symbols_lib as csl  # noqa: E402
from realchords.utils.data_utils import chord_symbol_pitches  # noqa: E402

N_SONGS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
KINDS = [abbrevs[0] for abbrevs, _ in csl._CHORD_KINDS]


def pcs(name):
    return set(chord_symbol_pitches(name)) | {csl.chord_symbol_bass(name) % 12}


def merge_target(name):
    """(target name, 'omission' | 'respelling') or None."""
    rule = chord_rule(name)
    if rule is None:
        return None
    P, root = pcs(name), csl.chord_symbol_root(name) % 12
    root_str, _, _, bass_str = csl._split_chord_symbol(name)
    best = None
    for kind in KINDS:
        cand = root_str + kind + ("" if rule["bass"] == root else bass_str)
        try:
            Q, rk = pcs(cand), chord_rule(cand)
        except Exception:  # noqa: BLE001
            continue
        if rk and P <= Q and rk["required"] <= P and len(P) >= min(3, len(Q)):
            if best is None or len(Q) < len(best[1]):
                best = (cand, Q)
    if not best or best[0] == name:
        return None
    return best[0], ("omission" if P < best[1] else "respelling")


agent = A.agent_interface.Agent(compile=False, voicings_path=None)
RUNS = [("GAPT 7 datasets", "hooktheory"), ("GAPT 7 datasets", "emopia_plus"), ("GAPT 7 datasets", "pop909"),
        ("GAPT", "hooktheory"), ("ReaLchords", "hooktheory")]
for label, dataset in RUNS:
    model, tk = agent.models[label], agent.tokenizer_for(label)
    lo, hi = tk.chord_on_token_range
    kind_of = {}
    for tid in range(lo, hi + 1):
        m = merge_target(tk.id_to_name[tid].replace("CHORD_ON_", ""))
        if m:
            kind_of[tid] = m[1]
    ids = {k: torch.tensor([t - lo for t, v in kind_of.items() if v == k]) for k in ("omission", "respelling")}
    args = types.SimpleNamespace(dataset_name=dataset, dataset_split="test", max_len=512, batch_size=8,
                                 frames_per_bar=16, only_4_4=False, common_vocab=False,
                                 chord_names_path=str(Path(A.REALJAM_CHECKPOINT_DIR) / "chord_names_augmented.json"))
    loader = A.build_loader(args, agent, label)
    mass = Counter(); top = Counter(); n = 0
    for bi, batch in enumerate(loader):
        if bi * args.batch_size >= N_SONGS:
            break
        seq = batch["targets"]
        d = A.chord_frame_distributions(model, seq, tk, keep_onset_dist=True)
        dist, p_on = d["onset_dist"], d["p_onset"]
        chords = seq[:, 1::2][:, :p_on.size(1)]
        valid = (chords != tk.pad_token) & (chords != tk.eos_token)
        w = (p_on * valid).unsqueeze(-1)  # weight each frame by its onset probability
        mass["onset"] += float(w.sum())
        for k, idx in ids.items():
            if len(idx):
                mass[k] += float((dist[..., idx] * w).sum())
        allm = torch.tensor(sorted(kind_of)) - lo
        per_tok = (dist[..., allm] * w).sum((0, 1))
        for t, v in zip(allm.tolist(), per_tok.tolist()):
            top[tk.id_to_name[t + lo].replace("CHORD_ON_", "")] += v
        n += seq.size(0)
    o, r = mass["omission"] / mass["onset"], mass["respelling"] / mass["onset"]
    print(f"{label:18} {dataset:12} songs {n:4}  mergeable tokens: {len(kind_of):4} of {hi - lo + 1}  "
          f"share of onset probability: omissions {100 * o:.3f}%, respellings {100 * r:.3f}%, "
          f"total {100 * (o + r):.3f}%", flush=True)
    print("   top:", [(name, f"{100 * v / mass['onset']:.3f}%") for name, v in top.most_common(5)], flush=True)
