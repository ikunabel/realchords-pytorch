"""Does the model find the beat from the content or from its position in the window?

Teacher-forced Hooktheory test songs, cropped to start on a bar line + s frames (s = 0..3).
s = 0 is how training crops look (start on a beat); s = 1..3 is what ReaLJam's sliding
255-frame window gives after 16 bars in 3 of 4 frames. If the model uses position,
its onset spikes move to window positions k % 4 == 0 instead of the true beats.
"""
import sys, types, math
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/abel/Projects/Thesis/realchords-pytorch")
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "eval")]
import analyze_chord_onset_timing as A  # noqa: E402

N_SONGS, MODELS = int(sys.argv[1]), sys.argv[2:]


def make_crop(shift):
    def crop(self, melody, chord, idx=0):
        L = self.max_len_per_part
        starts = list(range(0, max(melody.shape[0] - L - shift, 0) + 1, 16)) or [0]
        s = starts[idx % len(starts)] + shift
        return melody[s:s + L], chord[s:s + L]
    return crop


agent = A.agent_interface.Agent(compile=False, voicings_path=None)
args = types.SimpleNamespace(dataset_name="hooktheory", dataset_split="test", max_len=512,
                             batch_size=8, frames_per_bar=16, only_4_4=False, common_vocab=False,
                             chord_names_path=str(Path(A.REALJAM_CHECKPOINT_DIR) / "chord_names_augmented.json"))
for label in MODELS:
    model, tk = agent.models[label], agent.tokenizer_for(label)
    print(f"\n== {label}: mean P(chord onset) | true on-beat / off-beat | at window pos k%4==0 / other | GT onsets on true beats")
    for shift in range(4):
        loader = A.build_loader(args, agent, label)
        loader.dataset.random_crop = types.MethodType(make_crop(shift), loader.dataset)
        on, off, pos0, posx, gt_on, gt_all = [], [], [], [], 0, 0
        for bi, batch in enumerate(loader):
            if bi * args.batch_size >= N_SONGS:
                break
            seq = batch["targets"]
            d = A.chord_frame_distributions(model, seq, tk)
            p = d["p_onset"]
            chords = seq[:, 1::2][:, :p.size(1)]
            lo, hi = tk.chord_on_token_range
            valid = (chords != tk.pad_token) & (chords != tk.eos_token)
            k = torch.arange(p.size(1))
            late = k >= 64                       # skip the first 4 bars of context
            true_beat = ((k + shift) % 4 == 0)
            win_beat = (k % 4 == 0)
            for b in range(p.size(0)):
                m = valid[b] & late
                on += p[b][m & true_beat].tolist(); off += p[b][m & ~true_beat].tolist()
                pos0 += p[b][m & win_beat].tolist(); posx += p[b][m & ~win_beat].tolist()
                g = (chords[b] >= lo) & (chords[b] <= hi) & m
                gt_on += int((g & true_beat).sum()); gt_all += int(g.sum())
        mean = lambda x: float(np.mean(x)) if x else float("nan")
        print(f"  shift {shift}: true beats {mean(on):.3f} / {mean(off):.3f}   "
              f"window k%4==0 {mean(pos0):.3f} / {mean(posx):.3f}   GT on beats {gt_on/max(gt_all,1):.1%}")
