"""Serves frontend and model endpoints for genjam interface.

@Author Alex Scarlatos, Tia-Jane Fowler, Yusong Wu
"""

import json
import os
import flask
import argparse

from realchords.realjam import agent_interface
from realchords.realjam import song_catalogue

DEFAULT_PORT = 8080

base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, "frontend")

app = flask.Flask(__name__, static_url_path="", static_folder=frontend_dir)

agent: agent_interface.Agent = None

# Flat song list and (dataset, split, id) -> cache record lookup for the
# "robot melody" playback feature. Built once at startup (see main())
# -- read-only afterward, so safe to share across requests.
song_catalogue_index: list = []
song_record_index: dict = {}


@app.get("/")
def index() -> str:
    """Get index page."""
    return flask.send_file(os.path.join(frontend_dir, "index.html"))


@app.get("/models")
def get_models() -> str:
    """Get available model names."""
    assert agent is not None
    return json.dumps(agent.get_models())


@app.get("/songs")
def get_songs() -> str:
    """Get the song list for the robot-melody feature's fuzzy search."""
    return json.dumps(song_catalogue_index)


@app.get("/songs/<dataset>/<split>/<song_id>/melody")
def get_song_melody(dataset: str, split: str, song_id: str) -> str:
    """Get a specific song's ground-truth melody notes for robot playback."""
    record = song_record_index.get((dataset, split, song_id))
    if record is None:
        flask.abort(404, description="Song not found")
    # num_beats sets the loop period: a melody that rests through its last
    # beats would otherwise be looped from its final note and drift off the
    # bar grid on every repetition.
    return json.dumps({
        "melody": song_catalogue.extract_melody_notes(record),
        "num_beats": record["annotations"].get("num_beats"),
    })


@app.get("/songs/<dataset>/<split>/<song_id>/reference")
def get_song_reference(dataset: str, split: str, song_id: str) -> str:
    """Get a song's ground-truth melody *and* its ground-truth chords.

    For A/B listening: play the original accompaniment the model was trained
    to reproduce, with no generation involved, against what the model does
    with the same melody.

    The chords are voiced through the same rule the live system would use
    (custom voicings when `?custom=1`, otherwise the fixed-octave fallback),
    and `prev_voicing` is threaded across the progression exactly as in live
    playback -- so the comparison isolates *which chords* were chosen rather
    than how they happen to be voiced.
    """
    assert agent is not None
    record = song_record_index.get((dataset, split, song_id))
    if record is None:
        flask.abort(404, description="Song not found")

    use_custom = flask.request.args.get("custom") == "1"
    selector = agent.voicing_selector if use_custom else None
    melody = song_catalogue.extract_melody_notes(record)

    def melody_ceiling(onset: float, offset: float):
        """Lowest melody note sounding at any point during a chord.

        The live path re-voices every frame, so it can track the melody with
        a point sample of whatever is sounding now. A reference chord is
        voiced once and held for its whole duration, so it has to clear the
        lowest note the melody reaches anywhere in that span -- sampling only
        at the onset leaves the chord colliding with the tune as it rises.
        Returns None where the melody rests throughout (e.g. intro chords
        before the tune enters), leaving the voicing unconstrained.
        """
        sounding = [
            note["pitch"] for note in melody
            if note["onset"] < offset and note["offset"] > onset
        ]
        return min(sounding) if sounding else None

    chords = []
    prev_voicing = None
    for chord in song_catalogue.extract_reference_chords(record):
        pitches = chord.get("pitches")  # a demo song's exact keys, if given
        if pitches is None and selector is not None:
            pitches = selector.select(
                chord["symbol"],
                prev_voicing=prev_voicing,
                update_state=False,
                melody_pitch=melody_ceiling(chord["onset"], chord["offset"]),
                melody_role="top",
            )
        if pitches is None:
            pitches = agent.legacy_chord_pitches(chord["symbol"])
        if not pitches:
            continue
        prev_voicing = pitches
        chords.append({**chord, "pitches": pitches})

    return json.dumps({
        "melody": melody,
        "chords": chords,
        "num_beats": record["annotations"].get("num_beats"),
    })


@app.get("/voicings/chords")
def get_voicing_chords() -> str:
    """Get the full chord vocabulary, for the standalone voicing-browser UI
    (independent of any session). Not limited to what the custom voicing
    lookup covers -- uncovered chords fall back to a legacy voicing, see
    get_voicings() below -- so every chord in the vocabulary is browsable."""
    assert agent is not None
    return json.dumps(agent.list_all_chords())


@app.get("/voicings")
def get_voicings() -> str:
    """Get the voicing list for one chord symbol, for browsing.

    Takes the chord name as a query param (`?chord=...`), not a path
    segment, since many chord symbols contain '/' (slash chords like "C/E")
    which would otherwise be misparsed as extra path segments. Falls back to
    a single legacy fixed-octave voicing (tagged "legacy": true) for chords
    the custom lookup doesn't cover.
    """
    assert agent is not None
    chord_name = flask.request.args.get("chord", "")
    return json.dumps(agent.get_voicings_for_browser(chord_name))


@app.post("/play")
def play() -> str:
    """Generate new chords given context."""
    assert agent is not None
    payload = flask.request.get_json()
    new_chords, new_chord_tokens, intro_chord_tokens = agent.generate_live(
        payload["model"],
        payload["notes"],
        payload["chordTokens"],
        payload["frame"],
        payload["lookahead"],
        payload["commitahead"],
        float(payload["temperature"]),
        payload["silenceTill"],
        payload["introSet"],
        use_custom_voicings=payload.get("useCustomVoicings", False),
        prev_voicing=payload.get("prevVoicing"),
        vl_weight=payload.get("vlWeight"),
        reg_weight=payload.get("regWeight"),
        note_count_weight=payload.get("noteCountWeight"),
        density_weight=payload.get("densityWeight"),
        target_mid=payload.get("targetMid"),
        density_target=payload.get("densityTarget"),
    )
    return json.dumps(
        {
            "newChords": new_chords,
            "newChordTokens": new_chord_tokens,
            "introChordTokens": intro_chord_tokens,
            "frame": payload["frame"],
        }
    )


@app.post("/advance_chord")
def advance_chord() -> str:
    """Pick the chord to start right now, for manual chord-timing mode.

    Unlike /play, this generates no lookahead and commits nothing: the
    performer decides when each chord change happens, so the model is only
    asked which chord it would play at this instant.
    """
    assert agent is not None
    payload = flask.request.get_json()
    return json.dumps(
        agent.advance_chord(
            payload["model"],
            payload["notes"],
            payload["chordTokens"],
            payload["frame"],
            float(payload["temperature"]),
            use_custom_voicings=payload.get("useCustomVoicings", False),
            prev_voicing=payload.get("prevVoicing"),
            vl_weight=payload.get("vlWeight"),
            reg_weight=payload.get("regWeight"),
        )
    )


@app.post("/chord_candidates")
def chord_candidates() -> str:
    """Ranked chord onsets the model would start now, for chord-completion
    mode: the client picks the most probable one containing the notes the
    performer plays."""
    assert agent is not None
    payload = flask.request.get_json()
    return json.dumps(
        agent.chord_candidates(
            payload["model"],
            payload["notes"],
            payload["chordTokens"],
            payload["frame"],
            top_k=int(payload.get("topK", 200)),
        )
    )


def main() -> None:
    global agent, song_catalogue_index, song_record_index

    parser = argparse.ArgumentParser(description="Run the RealJam server")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the server on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--ssl", action="store_true", help="Enable SSL with adhoc certificate"
    )
    parser.add_argument(
        "--onnx", action="store_true", help="Use ONNX model instead of PyTorch"
    )
    parser.add_argument(
        "--mlx", action="store_true", help="Use MLX model instead of PyTorch"
    )
    parser.add_argument(
        "--onnx_provider",
        type=str,
        default=None,
        help=(
            "Execution provider for ONNX Runtime. "
            "If omitted, the script selects `CUDAExecutionProvider` when CUDA is "
            "available, otherwise `CPUExecutionProvider`. "
            "Use this flag to override the default—for example: "
            "`--onnx_provider CUDAExecutionProvider` or "
            "`--onnx_provider CPUExecutionProvider`."
        ),
    )
    args = parser.parse_args()
    if args.onnx and args.mlx:
        parser.error("--onnx and --mlx cannot be used together")

    agent = agent_interface.Agent(
        onnx=args.onnx,
        mlx=args.mlx,
        provider=args.onnx_provider,
    )

    try:
        song_catalogue_index, song_record_index = song_catalogue.build_catalogue()
        n_real = sum(e["dataset"] != "demo" for e in song_catalogue_index)
        print(
            f"Loaded robot-melody song catalogue: "
            f"{len(song_catalogue_index)} songs ({n_real} from datasets)"
        )
        if n_real == 0:
            print(
                "  No dataset songs found under data/cache/ (data/ is git-ignored, "
                "so a fresh checkout has none). Copy data/cache/<dataset>/"
                "{train,valid,test}.jsonl for hooktheory, pop909, nottingham, "
                "wikifonia, chord_melody_dataset and jazzmus from a machine that "
                "has them, and start the server from the repository root."
            )
    except Exception as e:
        print(
            f"Could not build song catalogue ({e}); "
            "robot-melody feature will show an empty list."
        )

    ssl_context = "adhoc" if args.ssl else None
    app.run(
        host="0.0.0.0", port=args.port, ssl_context=ssl_context, threaded=False
    )


if __name__ == "__main__":
    main()
