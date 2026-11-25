"""End-to-end timing of ReaLJam's auto mode: frontend + HTTP + server, in a real browser session.

Drives headless Chromium over the DevTools protocol against a running server: loads the app, sets model, bpm,
lookahead and commit-ahead, starts a robot-melody session (the frontend plays a Hooktheory test melody by
itself and the model accompanies it, as in a live session), lets it run, and reads back the frontend's
per-request timing log (`curSession.syncLog`, written in `syncWithServer`).

Per request the log holds the frame it was sent at (reqFrame), the frame asked for (askFrame = reqFrame + 1;
the server's prompt has the real melody up to askFrame - 1), the frame the response was applied at, the round
trip as the frontend sees it, and the chord-token count and window settings, from which the server's commit
size follows (commit_frames = nTokens - lookahead + commitahead - askFrame, as in generate_live).

From that, every played frame j gets the request its chord came from: a response writes its new chords
(after its committed frames) to frames it can still change, i.e. from max(askFrame, appliedFrame + 1) on;
committed frames keep the request that first chose them. The chord at j then depended on
j - origin.askFrame frames of imagined melody.

    python scripts/eval/live_session_timing.py --url http://127.0.0.1:8899 [--seconds 40]
    # macOS: --browser "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    python scripts/eval/live_session_timing.py --logs logs/realjam/"GenJam Session - ..."   # sessions you played

--logs analyses saved session logs instead of driving a browser: every auto-mode session saved since
2026-11-25 carries the same timing log (the frontend saves the whole session object).
"""
import argparse
import collections
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request

import numpy as np
import websocket

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--url", default="http://127.0.0.1:8899")
ap.add_argument("--model", default="GAPT")
ap.add_argument("--song", default="hooktheory/test/pyvgPG-dmYq", help="dataset/split/id (Just Can't Get Enough)")
ap.add_argument("--bpms", type=int, nargs="+", default=[80, 120])
ap.add_argument("--settings", nargs="+", default=["1/0", "1/1", "2/0", "2/2", "4/0", "4/4"],
                help="lookahead/commit-ahead in beats")
ap.add_argument("--seconds", type=float, default=40)
ap.add_argument("--port", type=int, default=9333)
ap.add_argument("--out", default=None, help="write the raw logs and the summary as JSON")
ap.add_argument("--browser", default=None, help="path to Chrome/Chromium (default: found on PATH)")
ap.add_argument("--logs", nargs="+", default=None, help="analyse saved session logs instead of running sessions")
args = ap.parse_args()

def analyse(r):
    log = [e for e in r["log"] if e["nNew"] > 0]
    origin = {}
    for e in log:
        la, ca, ask = e["lookahead"], e["commitahead"], e["askFrame"]
        commit = max(e["nTokens"] - la + ca - ask, 0)
        if commit >= la:
            continue
        for j in range(max(ask + commit, e["appliedFrame"] + 1), ask + e["nNew"]):
            origin[j] = ask
    frames = sorted(origin)
    imagined = np.array([j - origin[j] for j in frames])
    lag = np.array([e["appliedFrame"] - e["reqFrame"] for e in log])
    return {"requests": len(log), "fetch_ms_median": float(np.median([e["fetchMs"] for e in log])),
            "total_ms_median": float(np.median([e["totalMs"] for e in log])),
            "apply_lag_frames_median": float(np.median(lag)),
            "frames": len(frames), "imagined_median": float(np.median(imagined)),
            "imagined_p10": float(np.percentile(imagined, 10)),
            "imagined_p90": float(np.percentile(imagined, 90)),
            "imagined_hist": dict(sorted(collections.Counter(imagined.tolist()).items()))}


def analyse_complete(r):
    """Complete mode (candLog, completionHistory): per frame, did that frame's own candidate list arrive before
    the frame started, during it, later, or never; and how long gestures waited."""
    log = r["candLog"]
    first, last = min(e["askFrame"] for e in log), max(e["askFrame"] for e in log)
    arrived = {}
    for e in log:
        arrived[e["askFrame"]] = min(arrived.get(e["askFrame"], 10 ** 9), e["arrivedFrame"])
    cls = collections.Counter()
    for j in range(first, last + 1):
        a = arrived.get(j)
        cls["none" if a is None else "before" if a < j else "during" if a == j else "after"] += 1
    n = sum(cls.values())
    gest = [g for g in r.get("completionHistory") or [] if "waitMs" in g]
    return {"requests": len(log), "ms_median": float(np.median([e["ms"] for e in log])),
            "frames": n, **{k: cls[k] / n for k in ("before", "during", "after", "none")},
            "gestures": len(gest), "refetched": float(np.mean([g["refetched"] for g in gest])) if gest else None,
            "wait_ms_median": float(np.median([g["waitMs"] for g in gest])) if gest else None}


if args.logs:
    for path in args.logs:
        r = json.load(open(path))
        if r.get("candLog"):
            c = analyse_complete(r)
            print(f"{path}: complete mode, {c['requests']} candidate requests, round trip {c['ms_median']:.0f} ms; "
                  f"of {c['frames']} frames, own list arrived before the frame {c['before']:.0%}, during it "
                  f"{c['during']:.0%}, later {c['after']:.0%}, never {c['none']:.0%}"
                  + (f"; {c['gestures']} gestures, {c['refetched']:.0%} had to fetch, wait median "
                     f"{c['wait_ms_median']:.0f} ms" if c["gestures"] else ""))
            continue
        if not r.get("syncLog"):
            print(f"{path}: no timing log (saved before 2026-11-25?)")
            continue
        a = analyse({"log": r["syncLog"]})
        print(f"{path}: {a['requests']} requests, round trip {a['fetch_ms_median']:.0f} ms, applied "
              f"{a['apply_lag_frames_median']:.0f} frames after sending; imagined frames per played chord median "
              f"{a['imagined_median']:.0f} (p10-p90 {a['imagined_p10']:.0f}-{a['imagined_p90']:.0f})")
    raise SystemExit

browser = args.browser or shutil.which("chromium-browser") or shutil.which("chromium") or shutil.which("google-chrome")
profile = tempfile.mkdtemp(prefix="realjam_cdp_")
proc = subprocess.Popen([browser, "--headless=new", f"--remote-debugging-port={args.port}",
                         f"--user-data-dir={profile}", "--autoplay-policy=no-user-gesture-required",
                         "--mute-audio", "--no-first-run", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(100):
        try:
            tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json").read())
            page = next(t for t in tabs if t["type"] == "page")
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120)
    msg_id = 0

    def cdp(method, **params):
        global msg_id
        msg_id += 1
        ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == msg_id:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(expr):
        r = cdp("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    def run(bpm, la, ca):
        cdp("Page.navigate", url=args.url)
        for _ in range(300):
            time.sleep(0.2)
            try:
                if js("typeof playSongMelody === 'function' && !!lookaheadInput && !!modelSelect"
                      " && [...modelSelect.options].length > 0"):
                    break
            except Exception:  # noqa: BLE001
                pass
        js("showMainScreen(); true")
        js(f"""(async () => {{
            modelSelect.value = {json.dumps(args.model)};
            downloadSessionCheck.checked = false;  // robot test sessions: not into logs/realjam
            bpmInput.value = {bpm}; setBPM({bpm});
            lookaheadInput.value = {la}; setLookahead({la});
            commitaheadInput.value = {ca};
            await Tone.start();
            const r = await fetch('/songs/{args.song}/melody');
            const {{ melody, num_beats }} = await r.json();
            playSongMelody(melody, num_beats);
            return true;
        }})()""")
        time.sleep(args.seconds)
        out = js("JSON.stringify({log: curSession.syncLog || [], model: sessionModel(),"
                 " bpm: Tone.Transport.bpm.value, la: getLookaheadFrames(), ca: getCommitaheadFrames(),"
                 " frame: getSessionCurrentFrame(), notes: curSession.noteHistory.length})")
        js("if (curSession) toggleLiveSession(); true")
        return json.loads(out)

    results = []
    for bpm in args.bpms:
        for s in args.settings:
            la, ca = (int(x) for x in s.split("/"))
            r = run(bpm, la, ca)
            assert r["model"] == args.model and r["la"] == 4 * la and r["ca"] == 4 * ca and r["bpm"] == bpm, r
            a = analyse(r)
            results.append({"bpm": bpm, "lookahead_beats": la, "commit_beats": ca, **a, "raw": r})
            print(f"{bpm} bpm, lookahead {la} beat(s), commit {ca}: {a['requests']} requests, round trip "
                  f"{a['fetch_ms_median']:.0f} ms (incl. applying {a['total_ms_median']:.0f} ms), response applied "
                  f"{a['apply_lag_frames_median']:.0f} frames after sending; imagined frames per played chord "
                  f"median {a['imagined_median']:.0f} (p10-p90 {a['imagined_p10']:.0f}-{a['imagined_p90']:.0f}), "
                  f"{a['frames']} frames", flush=True)
finally:
    proc.terminate()
    shutil.rmtree(profile, ignore_errors=True)

print("\n| bpm | lookahead | commit | requests | round trip (frontend) | applied after (frames) | "
      "imagined frames per played chord: median (p10-p90) |")
print("|---|---|---|---|---|---|---|")
for x in results:
    print(f"| {x['bpm']} | {x['lookahead_beats']} beat(s) | {x['commit_beats']} beat(s) | {x['requests']} | "
          f"{x['fetch_ms_median']:.0f} ms | {x['apply_lag_frames_median']:.0f} | {x['imagined_median']:.0f} "
          f"({x['imagined_p10']:.0f}-{x['imagined_p90']:.0f}) |")
if args.out:
    json.dump(results, open(args.out, "w"))
