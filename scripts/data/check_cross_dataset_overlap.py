"""Cross-dataset song overlap (Wikifonia, JAZZMUS, Chord Melody, FiloBass, WJD) by normalised title, with a
transposition-invariant chord-sequence similarity for each shared title. See journal/reference/DATASET_SUMMARY.md,
"Cross-dataset overlap". Run from the repo root: python scripts/data/check_cross_dataset_overlap.py
"""
import json, re, collections, unicodedata, difflib
DS={"wikifonia":"wikifonia","jazzmus":"jazzmus","chord_melody_dataset":"chord_melody_dataset","filobass":"filobass","wjd":"wjazzd"}
def norm(t):
    t=unicodedata.normalize("NFKD",t or "").encode("ascii","ignore").decode().lower().replace("_"," ")
    t=re.sub(r"\(.*?\)","",t); t=re.sub(r"[^a-z0-9 ]"," ",t); t=re.sub(r"^(the|a) ","",t.strip())
    return re.sub(r"\s+"," ",t).strip()
def chords(d):
    return [(h["root_pitch_class"],tuple(h["root_position_intervals"])) for h in d["annotations"].get("harmony",[])]
songs=collections.defaultdict(list)  # title -> [(ds, split, id, chords)]
for ds,key in DS.items():
    for sp in ("train","valid","test"):
        for line in open(f"data/cache/{ds}/{sp}.jsonl"):
            d=json.loads(line); m=d.get(key) or {}
            t=norm(m.get("title"))
            if t: songs[t].append((ds,sp,str(m.get("id",m.get("melid"))),chords(d)))
def transp_sim(a,b):
    best=0
    for s in range(12):
        bb=[((r+s)%12,iv) for r,iv in b]
        best=max(best,difflib.SequenceMatcher(None,a,bb,autojunk=False).ratio())
    return best
pairs=collections.Counter(); leak=collections.Counter(); ex=collections.defaultdict(list); sims=collections.defaultdict(list)
for t,L in songs.items():
    dss={x[0] for x in L}
    if len(dss)<2: continue
    for i in range(len(L)):
        for j in range(len(L)):
            a,b=L[i],L[j]
            if a[0]>=b[0]: continue
            pairs[(a[0],b[0])]+=1
            if a[1]!=b[1] and "train" in (a[1],b[1]):
                k=(a[0],b[0]); leak[k]+=1
                if len(sims[k])<400: sims[k].append(transp_sim(a[3],b[3]))
                if len(ex[k])<4: ex[k].append((t,a[1],b[1]))
for k in sorted(pairs):
    s=sims.get(k,[]); hi=sum(1 for x in s if x>=0.8)
    print(f"{k[0]:22} {k[1]:22} shared-title pairs {pairs[k]:4}  train<->eval {leak[k]:4}  chord-sim>=0.8 {hi}/{len(s)}  e.g. {ex[k][:3]}")
