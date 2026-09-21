"""Sweep binning / EMD conventions for the chord-to-note onset interval
metric against ReaLchords paper Table 1 (x1000). See journal/METRICS.md,
"Resolution: histograms passed as values". Run from repo root."""
import numpy as np
import torch
from scipy.stats import wasserstein_distance

ROOT = "logs/custom_eval"
FW = f"{ROOT}/four_way_hooktheory_vs_7sets"
RV = f"{ROOT}/realchords_vs_realchords_multiscale_test/cropped_songs"

def load(p):
    return torch.load(p, weights_only=False)

gt = load(f"{FW}/gt/sync_intervals.pt")
systems = {
    "OnlineMLE(dec_ht)": (load(f"{FW}/models/decoder_ht/sync_intervals.pt"), 14.23),
    "OfflineMLE(encdec_ht)": (load(f"{FW}/models/encdec_ht/sync_intervals.pt"), 9.85),
    "ReaLchords": (load(f"{RV}/realchords_sync_intervals.pt"), 16.09),
    "ReaLchords-M": (load(f"{RV}/realchords_multiscale_sync_intervals.pt"), 17.17),
}


def hist(x, edges, normalize=True):
    h, _ = np.histogram(x, bins=edges)
    h = h.astype(float)
    return h / h.sum() if normalize else h


def edges_for(n):
    # n bins: exact 0..n-2, overflow [n-1, inf)  -> paper: [0,1,...,17,inf] = 18 bins
    return np.array(list(range(n)) + [np.inf], dtype=float)


def v_pos(p, q):  # proper EMD over bin positions (ground distance = |i-j| frames)
    pos = np.arange(len(p))
    return wasserstein_distance(pos, pos, p, q)


def v_pos_norm(p, q):  # positions rescaled to [0,1]
    pos = np.arange(len(p)) / (len(p) - 1)
    return wasserstein_distance(pos, pos, p, q)


def v_values(p, q):  # histograms passed as *values* (common misuse)
    return wasserstein_distance(p, q)


def v_tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def v_l1_mean(p, q):
    return np.abs(p - q).mean()


def v_cdf_mean(p, q):
    return np.abs(np.cumsum(p) - np.cumsum(q)).mean()


VARIANTS = {
    "EMD over bin pos (frames)": v_pos,
    "EMD over bin pos, pos in [0,1]": v_pos_norm,
    "wasserstein(hist, hist) as values": v_values,
    "total variation": v_tv,
    "mean |p-q|": v_l1_mean,
    "mean |CDF diff|": v_cdf_mean,
}


def pooled(d):
    return d["intervals_flat"].numpy()


def per_seq_avg_hist(d, edges):
    hs = [hist(np.asarray(s), edges) for s in d["intervals"] if len(s) > 0]
    return np.mean(hs, axis=0)


targets = np.array([t for _, t in systems.values()])
rows = []
for nbins in [12, 18, 19, 34, 108]:
    edges = edges_for(nbins)
    for agg in ["pooled", "per-seq avg"]:
        if agg == "pooled":
            q = hist(pooled(gt), edges)
            ps = [hist(pooled(d), edges) for d, _ in systems.values()]
        else:
            q = per_seq_avg_hist(gt, edges)
            ps = [per_seq_avg_hist(d, edges) for d, _ in systems.values()]
        for vname, fn in VARIANTS.items():
            vals = np.array([fn(p, q) * 1000 for p in ps])
            # log-scale error vs paper, and rank agreement
            err = np.mean(np.abs(np.log(vals / targets)))
            rows.append((err, nbins, agg, vname, vals))

rows.sort(key=lambda r: r[0])
names = list(systems)
print("targets:", dict(zip(names, targets)))
print(f"{'err':>6} {'bins':>4} {'agg':<12} {'variant':<36} " + " ".join(f"{n[:14]:>14}" for n in names))
for err, nb, agg, vn, vals in rows:
    print(f"{err:6.3f} {nb:4d} {agg:<12} {vn:<36} " + " ".join(f"{v:14.2f}" for v in vals))
