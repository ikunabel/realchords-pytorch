"""Interface for interacting with generative models.

@Author Tia-Jane Fowler, Alex Scarlatos, Yusong Wu
"""

from typing import Dict, List, Optional, Tuple, TypedDict
import note_seq
import numpy as np
import torch
import logging
import os
import time
import json
import copy

import argbind
import wget
from pathlib import Path
from realchords.utils.inference_utils import (
    load_gen_model_from_state_dict,
    load_rl_model,
)
from realchords.model.gen_model import (
    DecoderTransformer,
    EncoderDecoderTransformer,
)
from realchords.realjam.mlx_models import (
    MLXDecoderTransformer,
    MLXEncoderDecoderTransformer,
    MLX_IMPORT_ERROR,
)
from realchords.constants import (
    CHORD_OCTAVE,
    ZERO_OCTAVE,
    FRAME_PER_BEAT,
    BASS_OCTAVE,
    REALJAM_CHECKPOINT_DIR,
)
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.voicing_selector import VoicingSelector
from realchords.utils.data_utils import chord_symbol_pitches, to_chord_name
from realchords.model.sampling import (
    filter_invalid_tokens_generate_unconditional,
    filter_invalid_tokens_generate_online,
    filter_invalid_tokens_generate_single_part,
)
from functools import partial
import onnxruntime as ort
from realchords.realjam.onnx_kv import generate_tokens, generate_tokens_online
from realchords.realjam.prepare_onnx import prepare_online

logging.basicConfig(
    level=logging.INFO,  # show INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ChordInfo = Tuple[str, List[int], bool]


class NoteInfo(TypedDict):
    pitch: int
    frame: int
    on: bool


# With a walking bass line in the melody lane, chord voicings are picked from
# the voicing lookup among those lying wholly above this pitch. 99.65% of the
# WJD + FiloBass bass notes are at or below G3 (journal/reference/
# REALJAM_CUSTOM_VOICINGS.md, 2026-11-16).
BASS_LINE_VOICING_FLOOR = 55  # G3


def _active_melody_pitch(notes: List[NoteInfo], frame: int) -> Optional[int]:
    """Return the melody pitch sounding at `frame`, if any.

    Scans the full note-on/off event history up to and including `frame` and
    returns whichever pitch's most recent event was a note-on with no
    matching note-off yet. Melody is nominally monophonic, but if multiple
    pitches are somehow still held, returns the most recently onset one.
    Doesn't assume `notes` is sorted by frame (client-sent event order isn't
    strictly guaranteed).
    """
    onset_frame: Dict[int, int] = {}
    for note in notes:
        if note["frame"] > frame:
            continue
        if note["on"]:
            onset_frame[note["pitch"]] = note["frame"]
        else:
            onset_frame.pop(note["pitch"], None)
    if not onset_frame:
        return None
    return max(onset_frame.items(), key=lambda kv: kv[1])[0]


ONLINE_MODEL_PATH = os.path.join(
    REALJAM_CHECKPOINT_DIR,
    "realjam_decoder_only_online_chord_3_datasets",
    "step=11000.ckpt",
)

OFFLINE_MODEL_PATH = os.path.join(
    REALJAM_CHECKPOINT_DIR,
    "realjam_enc_dec_base_chord_random_truncate_3_datasets",
    "step=20000.ckpt",
)

# Chord vocabularies (journal/reference/TOKENIZERS.md, section 0). Every model
# uses the one its checkpoint was trained with, told apart by the checkpoint's
# token count (the embedding size is the ground truth): the old one (2,821
# chords, 5,902 tokens) for the downloaded models and everything converted to
# it, the new one (2,969 chords, 6,198 tokens) for the 7-dataset models.
# Melody and special token ids are the same in both; every chord id differs.
GROWN_CHORD_NAMES_PATH = os.path.join("data", "cache", "chord_names_augmented.json")

MODEL_PATHS = {
    "GAPT": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "gapt",
        "actor.pth",
    ),
    "ReaLchords": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "realchords",
        "actor.pth",
    ),
    "Online MLE": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "realjam_decoder_only_online_chord_3_datasets",
        "step=11000.ckpt",
    ),
}

# Extra models, listed only where their checkpoint exists (they aren't part of
# the downloaded bundle). Trained in this repo on the grown chord vocabulary
# and converted to ReaLJam's with scripts/convert_checkpoint_for_realjam.py.
OPTIONAL_MODEL_PATHS = {
    # GAPT trained on the 7-dataset mix (Hooktheory, POP909, Nottingham,
    # Wikifonia, Chord Melody Dataset, JAZZMUS, EMOPIA+), seed 42, alpha 0.5,
    # in its own (new) vocabulary of 2,969 chords: the RL actor as trained,
    # with the run's args.yml (architecture) next to it
    "GAPT 7 datasets": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "gapt_loo_7sets.alpha=0.5.seed=42",
        "actor.pth",
    ),
    # Online MLE trained on WJD + FiloBass only: its "melody" lane is a walking
    # bass line, so it expects a bass line where the other models expect a tune
    "WJD+FiloBass MLE (bass input)": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "wjd_filobass_online_chord",
        "step=2700.ckpt",
    ),
}

# Bass (melody-lane) models, trained interleaved [bass_t, chord_t, ...] to
# predict the next bass note from the chords so far. They can't serve the
# chord-generating modes; generate_bass plays them against a song's
# ground-truth chords. Listed only where the checkpoint exists.
OPTIONAL_BASS_MODEL_PATHS = {
    "WJD+FiloBass bass (chord input)": os.path.join(
        REALJAM_CHECKPOINT_DIR,
        "wjd_filobass_online_bass",
        "step=2000.ckpt",
    ),
}

ONLINE_MODEL_CANDIDATES = [ONLINE_MODEL_PATH]
OFFLINE_MODEL_CANDIDATES = [OFFLINE_MODEL_PATH]
MODEL_PATH_CANDIDATES = {
    model_name: [model_path] for model_name, model_path in MODEL_PATHS.items()
}
MODEL_PATH_CANDIDATES.update({
    model_name: [model_path]
    for model_name, model_path in OPTIONAL_MODEL_PATHS.items()
    if os.path.exists(model_path)
})


def _resolve_existing_path(candidates: List[str]) -> str:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find any checkpoint from candidates: {candidates}"
    )


def download_checkpoints(
    checkpoint_dir: Optional[str] = None, force: bool = False
) -> bool:
    """Download checkpoint files from Hugging Face using wget.

    This function downloads the required checkpoint files and models from the
    lukewys/realchords-pytorch repository on Hugging Face. It uses wget for
    downloading, which should be available on most Linux systems.

    Args:
        checkpoint_dir: Directory to save checkpoints. If None, uses REALJAM_CHECKPOINT_DIR.
        force: If True, re-download files even if they already exist.

    Returns:
        True if download was successful, False otherwise.

    Example:
        >>> download_checkpoints()  # Downloads to default checkpoint directory
        >>> download_checkpoints("/path/to/checkpoints", force=True)  # Force re-download
    """
    if checkpoint_dir is None:
        checkpoint_dir = REALJAM_CHECKPOINT_DIR

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Define files and directories to download
    base_url = "https://huggingface.co/lukewys/realchords-pytorch/resolve/main"

    downloads = [
        # Single file
        {
            "url": f"{base_url}/chord_names_augmented.json",
            "output": checkpoint_path / "chord_names_augmented.json",
            "is_file": True,
        },
        # Directories
        {
            "name": "gapt",
            "url": f"{base_url}/gapt",
            "output": checkpoint_path / "gapt",
            "files": ["actor.pth", "args.yml"],
        },
        {
            "name": "realchords",
            "url": f"{base_url}/realchords",
            "output": checkpoint_path / "realchords",
            "files": ["actor.pth", "args.yml"],
        },
        {
            "name": "realjam_decoder_only_online_chord_3_datasets",
            "url": f"{base_url}/realjam_decoder_only_online_chord_3_datasets",
            "output": checkpoint_path
            / "realjam_decoder_only_online_chord_3_datasets",
            "files": ["step=11000.ckpt", "args.yml"],
        },
        {
            "name": "realjam_enc_dec_base_chord_random_truncate_3_datasets",
            "url": f"{base_url}/realjam_enc_dec_base_chord_random_truncate_3_datasets",
            "output": checkpoint_path
            / "realjam_enc_dec_base_chord_random_truncate_3_datasets",
            "files": ["step=20000.ckpt", "args.yml"],
        },
    ]

    logging.info(f"Starting checkpoint download to {checkpoint_path}")

    # Download chord_names_augmented.json
    chord_names_file = downloads[0]
    output_file = chord_names_file["output"]

    if not output_file.exists() or force:
        logging.info(f"Downloading {chord_names_file['url']}...")
        try:
            wget.download(chord_names_file["url"], str(output_file))
            logging.info(
                f"\nSuccessfully downloaded chord_names_augmented.json"
            )
        except Exception as e:
            logging.error(f"Failed to download chord_names_augmented.json: {e}")
            return False
    else:
        logging.info(
            f"chord_names_augmented.json already exists, skipping download"
        )

    # Download model directories
    for download_item in downloads[1:]:
        output_dir = download_item["output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading {download_item['name']}...")

        for file_name in download_item["files"]:
            output_file = output_dir / file_name
            file_url = f"{download_item['url']}/{file_name}"

            if not output_file.exists() or force:
                logging.info(f"  Downloading {file_name}...")
                try:
                    wget.download(file_url, str(output_file))
                    logging.info(f"\n  Successfully downloaded {file_name}")
                except Exception as e:
                    logging.error(f"  Failed to download {file_name}: {e}")
                    return False
            else:
                logging.info(f"  {file_name} already exists, skipping download")

    logging.info("All checkpoints downloaded successfully!")
    return True


class Agent:
    """Interface for interacting with generative chord model."""

    def __init__(
        self,
        onnx: bool = False,
        mlx: bool = False,
        provider=None,
        compile: bool = True,
        auto_download: bool = True,
        voicings_path: Optional[str] = "data/voicings/merged/chord_voicings.json",
    ) -> None:
        """Initialize the agent.

        Args:
            onnx: Whether to use ONNX runtime for inference.
            mlx: Whether to use the MLX backend for inference.
            provider: ONNX execution provider (e.g., 'CUDAExecutionProvider', 'CPUExecutionProvider').
            compile: Whether to compile the model with torch.compile.
            auto_download: Whether to automatically download checkpoints if they don't exist.
            voicings_path: Path to a chord_voicings.json lookup table (see
                scripts/extract_voicings/) used for the optional "custom voicings"
                playback mode. Not part of the downloaded checkpoint bundle — if
                missing, custom-voicing requests silently fall back to the
                fixed-octave rule.
        """
        if onnx and mlx:
            raise ValueError("ONNX and MLX backends are mutually exclusive")

        logging.info("Creating model...")

        # Check if checkpoints exist, download if needed
        chord_names_path = os.path.join(
            REALJAM_CHECKPOINT_DIR, "chord_names_augmented.json"
        )
        if auto_download and not os.path.exists(chord_names_path):
            logging.info(
                "Checkpoints not found. Downloading from Hugging Face..."
            )
            if not download_checkpoints():
                raise RuntimeError(
                    "Failed to download checkpoints. Please download manually or "
                    "set auto_download=False and ensure checkpoints are available."
                )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # decide the onnx provider based on the device
        if provider is None:
            if self.device.type == "cuda":
                provider = "CUDAExecutionProvider"
            else:
                provider = "CPUExecutionProvider"

        self.compile = compile
        self.onnx = onnx
        self.mlx = mlx
        self.max_frames = 256

        # Create tokenizer
        chord_names_path = os.path.join(
            REALJAM_CHECKPOINT_DIR, "chord_names_augmented.json"
        )
        with open(chord_names_path, "r") as f:
            chord_names = json.load(f)
        tokenizer = HooktheoryTokenizer(chord_names=chord_names)
        # The old vocabulary: the default, and always that of the offline
        # (intro) model and the melody helpers (melody ids are the same in
        # both vocabularies)
        self.tokenizer = tokenizer
        self.chord_names = chord_names  # full vocabulary, for the voicing browser
        self._tokenizers = {tokenizer.num_tokens: tokenizer}  # by token count
        self.model_tokenizers: Dict[str, HooktheoryTokenizer] = {}
        self.bass_tokenizers: Dict[str, HooktheoryTokenizer] = {}
        self._filter_cache: Dict[int, Dict] = {}

        # Optional data-driven voicing lookup for the "custom voicings" live
        # toggle. Loaded once at startup and used statelessly per-request
        # (see decode_chord_token_custom_voicing) so it's safe to share
        # across concurrently connected clients.
        self.voicing_selector: Optional[VoicingSelector] = None
        if voicings_path and not os.path.exists(voicings_path) and \
                os.path.exists(voicings_path + ".gz"):
            voicings_path += ".gz"  # as shipped in the realjam-data repository
        if voicings_path and os.path.exists(voicings_path):
            try:
                self.voicing_selector = VoicingSelector(voicings_path)
                logging.info(
                    "Loaded custom voicing lookup from %s", voicings_path
                )
            except Exception as e:
                logging.warning(
                    "Failed to load voicing lookup from %s: %s. "
                    "Custom-voicing requests will fall back to the "
                    "fixed-octave rule.",
                    voicings_path,
                    e,
                )
        else:
            logging.info(
                "No voicing lookup found at %s; custom-voicing requests "
                "will fall back to the fixed-octave rule.",
                voicings_path,
            )

        # Alternative lookups to compare by ear (2026-11-23), loaded on first
        # use: the fit-rule mapping (scripts/extract_voicings/
        # fit_voicings_to_chords.py; journal reference/VOICING_EXTRACTION.md,
        # "Fit rule"). The page switches with POST /voicing_lookup.
        self.voicing_lookup_paths: Dict[str, str] = {}
        if self.voicing_selector is not None:
            self.voicing_lookup_paths["pitch-class set (current)"] = voicings_path
            base = os.path.dirname(voicings_path)
            for label, fname in (("fit rule, strict", "chord_voicings_fit_strict.json"),
                                 ("fit rule, with tensions", "chord_voicings_fit_tensions.json")):
                path = os.path.join(base, fname)
                if os.path.exists(path):
                    self.voicing_lookup_paths[label] = path
        self.voicing_lookup_active = next(iter(self.voicing_lookup_paths), None)
        self._voicing_selectors: Dict[str, VoicingSelector] = (
            {self.voicing_lookup_active: self.voicing_selector} if self.voicing_lookup_active else {})

        # Load non-causal (offline) model
        self.enc_dec_model_path = _resolve_existing_path(
            OFFLINE_MODEL_CANDIDATES
        )
        self.non_causal_model = self._load_offline_model(
            self.enc_dec_model_path
        )
        self.non_causal_model = self._prepare_model(self.non_causal_model)

        self.mle_model = self._load_online_model(
            _resolve_existing_path(ONLINE_MODEL_CANDIDATES)
        )

        self.models = {}
        for model_name, candidates in MODEL_PATH_CANDIDATES.items():
            model_path = _resolve_existing_path(candidates)
            tk = self._tokenizer_for_checkpoint(model_path)
            if tk is None:
                logging.warning("Skipping model %s: no chord vocabulary with its token count", model_name)
                continue
            if self.onnx and tk is not self.tokenizer:
                logging.warning("Skipping model %s: ONNX only supports the old vocabulary", model_name)
                continue

            if ".pth" not in os.path.basename(model_path):
                model = self._load_online_model(model_path, tokenizer=tk)

            else:
                model = self._load_online_model(model_path, is_rl=True, tokenizer=tk)

            model = self._prepare_model(model)

            if self.onnx:
                model = self._prepare_onnx_model(provider, model_path)
            self.models[model_name] = model
            self.model_tokenizers[model_name] = tk
            logging.info("Loaded model %s (%d tokens)", model_name, tk.num_tokens)

        self.bass_models = {}
        for model_name, model_path in OPTIONAL_BASS_MODEL_PATHS.items():
            if os.path.exists(model_path):
                tk = self._tokenizer_for_checkpoint(model_path) or self.tokenizer
                self.bass_models[model_name] = self._prepare_model(
                    self._load_online_model(model_path, tokenizer=tk))
                self.bass_tokenizers[model_name] = tk

        # Prepare MLE model after all models are loaded because
        # loading RL model needs un-compiled MLE model
        self.mle_model = self._prepare_model(self.mle_model)

        # Define filter functions to avoid invalid tokens during generation
        self.online_model_filter_fn = partial(
            filter_invalid_tokens_generate_unconditional,
            model_part="chord",
            tokenizer=self.tokenizer,
            filter_opposite_part=True,
        )
        self.online_model_filter_kwargs = dict()

        self.online_model_commit_filter_fn = partial(
            filter_invalid_tokens_generate_online,
            model_part="melody",
            tokenizer=self.tokenizer,
        )
        self.online_model_commit_filter_kwargs = dict()

        self.non_causal_model_filter_fn = partial(
            filter_invalid_tokens_generate_single_part,
            model_part="chord",
            tokenizer=self.tokenizer,
        )
        self.non_causal_model_filter_kwargs = dict()

    def _prepare_onnx_model(self, provider, model_path):
        logging.info(f"===========Preparing model for ONNX...===========")
        save_dir = os.path.dirname(model_path)
        save_dir = os.path.join(save_dir, "onnx")
        onnx_init, onnx_step = prepare_online(
            self.mle_model,
            self.tokenizer,
            save_dir=save_dir,
            model_name="onnx_online",
            max_gen_seq_len=self.max_frames * 2,
            provider=provider,
        )
        logging.info(f"===========Created ONNX model===========")
        return onnx_init, onnx_step

    def _tokenizer_for_checkpoint(self, path: str) -> Optional[HooktheoryTokenizer]:
        """The tokenizer whose token count matches the checkpoint's embedding
        (5,902 = the old vocabulary, 6,198 = the new one); None if neither."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        rows = next((v.shape[0] for k, v in state.items() if k.endswith("token_emb.emb.weight")), None)
        if rows is None:
            return None
        if rows not in self._tokenizers and os.path.exists(GROWN_CHORD_NAMES_PATH):
            with open(GROWN_CHORD_NAMES_PATH, "r") as f:
                grown = HooktheoryTokenizer(chord_names=json.load(f))
            self._tokenizers.setdefault(grown.num_tokens, grown)
        return self._tokenizers.get(rows)

    def tokenizer_for(self, model_name: Optional[str]) -> HooktheoryTokenizer:
        """The tokenizer (chord vocabulary) of a chord model or a bass model;
        the old one for None or an unknown name."""
        return (self.model_tokenizers.get(model_name) or self.bass_tokenizers.get(model_name)
                or self.tokenizer)

    def remap_chord_tokens(self, tokens: List[int], src: HooktheoryTokenizer,
                           dst: HooktheoryTokenizer) -> List[int]:
        """Chord tokens of one vocabulary in another, by name (melody and
        special ids are the same in both; -1 stays -1). A chord `dst` lacks
        becomes SILENCE."""
        if src is dst:
            return list(tokens)
        silence = dst.name_to_id["SILENCE"]
        return [t if t < 0 else dst.name_to_id.get(src.id_to_name[t], silence) for t in tokens]

    def _filters(self, tk: HooktheoryTokenizer) -> Dict:
        """The generation filters for a vocabulary (they read its token ranges)."""
        if tk is self.tokenizer:
            return {"online": self.online_model_filter_fn, "commit": self.online_model_commit_filter_fn}
        if tk.num_tokens not in self._filter_cache:
            self._filter_cache[tk.num_tokens] = {
                "online": partial(filter_invalid_tokens_generate_unconditional, model_part="chord",
                                  tokenizer=tk, filter_opposite_part=True),
                "commit": partial(filter_invalid_tokens_generate_online, model_part="melody", tokenizer=tk),
            }
        return self._filter_cache[tk.num_tokens]

    def get_models(self) -> List[str]:
        return list(self.models.keys())

    def get_bass_models(self) -> List[str]:
        return list(self.bass_models.keys())

    def song_chord_frames(self, harmony: List[Dict], num_frames: int,
                          tokenizer: Optional[HooktheoryTokenizer] = None) -> List[int]:
        """A song's chord annotation as one chord token per frame, in the given
        vocabulary (default the old one). Like HooktheoryTokenizer.chord_to_frames,
        but a chord the vocabulary lacks becomes silence instead of an error."""
        tk = tokenizer or self.tokenizer
        silence = tk.name_to_id["SILENCE"]
        frames = [silence] * num_frames
        for c in harmony:
            try:
                name = to_chord_name(c["root_pitch_class"], c["root_position_intervals"])
                hold = tk.name_to_id[f"CHORD_{name}"]
                onset = tk.name_to_id[f"CHORD_ON_{name}"]
            except Exception:  # unparsable, or not in this vocabulary
                continue
            start = int(c["onset"] * FRAME_PER_BEAT)
            end = max(int(c["offset"] * FRAME_PER_BEAT), start + 1)
            for f in range(max(start, 0), min(end, num_frames)):
                frames[f] = hold
            if 0 <= start < num_frames:
                frames[start] = onset
        return frames

    def generate_bass(
        self,
        model_name: str,
        song_chords: List[int],
        bass_tokens: List[int],
        frame: int,
        num_frames: int,
        temperature: float,
    ) -> List[int]:
        """Generate the bass (melody) lane for frames [frame, frame+num_frames)
        against a song's ground-truth chords.

        The model reads [BOS, bass_0, chord_0, bass_1, chord_1, ...]: each bass
        token is sampled -- restricted to valid melody tokens given the previous
        one -- and the song's chord for that frame is then appended as given.
        The chords are known for every frame, so nothing is ever re-planned.
        The song loops: frame f has chord song_chords[f % len(song_chords)].

        Args:
          model_name: one of get_bass_models()
          song_chords: one chord token per frame of the song (song_chord_frames)
          bass_tokens: the bass tokens generated so far, frames 0..frame-1
          frame: first frame to generate
          num_frames: how many frames to generate
          temperature: sampling temperature

        Returns:
          the new bass tokens, one per frame
        """
        model = self.bass_models[model_name]
        tk = self.tokenizer_for(model_name)  # song_chords must be in this vocabulary
        chord_at = lambda f: song_chords[f % len(song_chords)]
        bass_hist = list(bass_tokens[:frame])
        bass_hist += [tk.silence_token] * (frame - len(bass_hist))

        # History that fits next to the frames to generate
        start = max(0, frame - (self.max_frames - num_frames))
        seq = [tk.bos_token]
        for f in range(start, frame):
            bass, chord = bass_hist[f], chord_at(f)
            if f == start:  # can't hold what the model can't see
                if tk.is_note_hold(bass):
                    bass = tk.note_hold_id_to_note_id(bass)
                if tk.is_chord_hold(chord):
                    chord = tk.chord_hold_id_to_chord_id(chord)
            seq += [bass, chord]
        seq = torch.tensor([seq], dtype=torch.long, device=self.device)

        new_tokens = []
        with torch.no_grad():
            for k in range(num_frames):
                if self.mlx:
                    logits = model(seq)[:, -1].float()
                else:
                    logits = model.net(seq, return_intermediates=True)[0][:, -1].float()
                logits = filter_invalid_tokens_generate_online(
                    logits, seq, k, model_part="melody", tokenizer=tk)
                probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
                bass = int(torch.multinomial(probs, 1)[0, 0])
                new_tokens.append(bass)
                chord = chord_at(frame + k)
                if frame + k == start and tk.is_chord_hold(chord):
                    chord = tk.chord_hold_id_to_chord_id(chord)
                seq = torch.cat([seq, torch.tensor(
                    [[bass, chord]], dtype=torch.long, device=self.device)], dim=1)
        return new_tokens

    def bass_token_info(self, token: int) -> Dict:
        """{pitch, on} for a bass token: pitch None for silence, `on` true for
        a note onset (as opposed to holding the previous note)."""
        name = self.tokenizer.id_to_name[token]
        if name.startswith("NOTE_ON_"):
            return {"pitch": int(name[len("NOTE_ON_"):]), "on": True}
        if name.startswith("NOTE_"):
            return {"pitch": int(name[len("NOTE_"):]), "on": False}
        return {"pitch": None, "on": False}

    def melody_to_frame_tokens(
        self, melody_data: List[NoteInfo], end_frame: int = 0
    ) -> List[int]:
        """Given melody, convert to homophony and return frame-based tokens.

        Args:
          melody_data: list of note on/off events from frontend
          end_frame: frame to pad till or cut off at (if > 0)

        Returns:
          list of note tokens (rest/onset/hold), one for each frame
        """

        in_progress = {}
        notes = []
        for event in melody_data:
            if event["on"]:
                # Add note to in-progress notes
                in_progress[event["pitch"]] = event["frame"]
            elif event["pitch"] in in_progress:
                # Remove note from in-progress notes
                note_start_frame = in_progress.pop(event["pitch"])
                # Make sure this note's end frame doesn't exceed the overall end frame
                if end_frame > 0:
                    note_end_frame = min(end_frame, event["frame"])
                else:
                    note_end_frame = event["frame"]

                # Calculate other fields, then add note to notes list
                notes.append(
                    {
                        "onset": note_start_frame / FRAME_PER_BEAT,
                        "offset": note_end_frame / FRAME_PER_BEAT,
                        "octave": ((event["pitch"] - ZERO_OCTAVE) // 12),
                        "pitch_class": (event["pitch"] % 12),
                    }
                )

        # Truncate any notes extending past end_frame
        for pitch, note_start_frame in in_progress.items():
            if end_frame > 0:
                notes.append(
                    {
                        "onset": note_start_frame / FRAME_PER_BEAT,
                        "offset": end_frame / FRAME_PER_BEAT,
                        "octave": (pitch - ZERO_OCTAVE) // 12,
                        "pitch_class": pitch % 12,
                    }
                )

        # Determine the total sequence duration
        if end_frame > 0:
            duration = end_frame
        else:
            # If no end frame passed in, calculate it from the notes
            duration = notes[-1]["offset"] * FRAME_PER_BEAT if notes else 0

        tokens = self.tokenizer.melody_to_frames(notes, duration)
        return tokens

    def decode_chord_token(self, chord_token: int,
                           tokenizer: Optional[HooktheoryTokenizer] = None) -> ChordInfo:
        """Extract the underlying information from a given chord token.

        Args:
          chord_token: the chord token to decode
          tokenizer: the vocabulary it is in (default the old one)

        Returns:
          chord_symbol: string symbolic representation of chord
          chord_pitches: list of pitches for chord
          is_onset: if the chord is a hit or a hold
        """
        tk = tokenizer or self.tokenizer
        if tk.is_silence(chord_token):
            return "", [], True
        is_onset = tk.is_chord_on(chord_token)
        token_name = tk.id_to_name[chord_token]
        if is_onset:
            chord_symbol = token_name.replace("CHORD_ON_", "")
        else:
            chord_symbol = token_name.replace("CHORD_", "")
        pitches = self.legacy_chord_pitches(chord_symbol)
        if pitches is None:
            chord_symbol = ""
            pitches = []
        return chord_symbol, pitches, is_onset

    def legacy_chord_pitches(self, chord_symbol: str) -> Optional[List[int]]:
        """The original fixed-octave voicing rule: every chord tone at
        CHORD_OCTAVE, plus a separate bass note at BASS_OCTAVE. Shared by
        decode_chord_token and the voicing browser's fallback for chords
        the custom voicing lookup doesn't cover (e.g. most slash chords).

        Returns None if `chord_symbol` doesn't parse as a real chord.
        """
        try:
            chord_pitches = chord_symbol_pitches(chord_symbol)  # add7 = major 7th (note_seq bug A)
            bass_pitch = note_seq.chord_symbol_bass(chord_symbol)
            return [
                *[CHORD_OCTAVE * 12 + pitch for pitch in chord_pitches],
                BASS_OCTAVE * 12 + bass_pitch,
            ]
        except (note_seq.ChordSymbolError, TypeError) as e:
            logging.warning(
                "Failed to convert chord symbol %s to pitches: %s",
                chord_symbol,
                e,
            )
            return None

    def list_all_chords(self) -> List[str]:
        """Full chord vocabulary, for the voicing-browser dropdown -- NOT
        limited to the subset the custom voicing lookup covers (unlike
        VoicingSelector.list_chords()), so every chord is browsable even if
        it'll fall back to legacy_chord_pitches when selected.
        """
        return sorted(self.chord_names)

    def get_voicings_for_browser(self, chord_name: str) -> List[Dict]:
        """Voicings for one chord, for the standalone voicing browser: the
        original ReaLchords fixed-octave voicing always comes first (for
        comparison), followed by the stored custom voicings, if the lookup
        covers this chord (e.g. most slash chords are outcompeted by their
        root-position equivalent under the lookup's exact pitch-class-set
        matching -- see journal/reference/VOICING_EXTRACTION.md). A custom voicing
        identical to the fixed-octave one is not listed twice.
        """
        voicings = []
        legacy_pitches = self.legacy_chord_pitches(chord_name)
        if legacy_pitches is not None:
            voicings.append(
                {"pitches": legacy_pitches, "count": None, "num_songs": None, "legacy": True}
            )

        if self.voicing_selector is not None:
            for voicing in self.voicing_selector.get_voicings(chord_name) or []:
                if legacy_pitches is not None and sorted(voicing["pitches"]) == sorted(legacy_pitches):
                    continue
                voicings.append(voicing)
        return voicings

    def select_voicing_above_bass_line(
        self, chord_symbol: str, prev_voicing: Optional[List[int]], **weights
    ) -> Optional[List[int]]:
        """A voicing from the lookup that lies wholly above a walking bass
        line (every note above BASS_LINE_VOICING_FLOOR), voice-led from
        `prev_voicing`. A slash chord the lookup doesn't cover is voiced as
        its chord without the slash: the bass line plays the bass.

        Returns None without a selector or if neither symbol is covered.
        """
        if self.voicing_selector is None or not chord_symbol:
            return None
        for symbol in dict.fromkeys([chord_symbol, chord_symbol.split("/")[0]]):
            pitches = self.voicing_selector.select(
                symbol, prev_voicing=prev_voicing, update_state=False,
                melody_pitch=BASS_LINE_VOICING_FLOOR, melody_role="bass",
                **weights,
            )
            if pitches is not None:
                return pitches
        return None

    def decode_chord_token_custom_voicing(
        self,
        chord_token: int,
        prev_voicing: Optional[List[int]],
        vl_weight: Optional[float] = None,
        reg_weight: Optional[float] = None,
        note_count_weight: Optional[float] = None,
        density_weight: Optional[float] = None,
        target_mid: Optional[float] = None,
        density_target: Optional[float] = None,
        melody_pitch: Optional[int] = None,
        bass_line: bool = False,
        tokenizer: Optional[HooktheoryTokenizer] = None,
    ) -> Tuple[ChordInfo, Optional[List[int]]]:
        """Like decode_chord_token, but voices the chord using
        self.voicing_selector (data-driven, voice-leading-aware) instead of
        the fixed-octave rule, when available.

        Falls back to decode_chord_token's fixed-octave pitches when no
        selector is loaded, the chord symbol isn't covered by the lookup
        table, or the symbol is empty (silence/parse failure).

        Args:
          chord_token: the chord token to decode
          prev_voicing: pitches of the previously voiced chord, threaded in
            explicitly (rather than tracked as instance state) so this stays
            safe to call for multiple concurrently-jamming clients sharing
            one Agent — see VoicingSelector's own stateless `select(...,
            update_state=False)` mode.
          vl_weight, reg_weight, note_count_weight, density_weight,
          target_mid, density_target:
            Per-call overrides for VoicingSelector's scoring weights (``None``
            falls back to the selector's own constructed defaults). Threaded
            through the same way as prev_voicing, for the same concurrency
            reason — different live-jam clients can use different slider
            values against one shared selector instance.
          melody_pitch: MIDI pitch currently sounding in the melody, if any
            (see _active_melody_pitch) — passed as a hard ceiling constraint
            so chord voicings can never climb above/collide with the melody.
          bass_line: the melody lane is a walking bass line: pick a voicing
            above it instead (select_voicing_above_bass_line); melody_pitch
            is then unused.

        Returns:
          chord_info: same (chord_symbol, chord_pitches, is_onset) as
            decode_chord_token.
          new_prev_voicing: pitches to pass as prev_voicing on the NEXT call,
            for voice-leading continuity across a sequence of chords.
            Carries the previous value forward on silence/failure so a
            single missed frame doesn't reset voice-leading context.
        """
        tk = tokenizer or self.tokenizer  # the vocabulary chord_token is in
        if tk.is_silence(chord_token):
            return ("", [], True), prev_voicing

        is_onset = tk.is_chord_on(chord_token)
        token_name = tk.id_to_name[chord_token]
        chord_symbol = token_name.replace(
            "CHORD_ON_" if is_onset else "CHORD_", ""
        )

        pitches: Optional[List[int]] = None
        if bass_line:
            pitches = self.select_voicing_above_bass_line(
                chord_symbol, prev_voicing,
                vl_weight=vl_weight, reg_weight=reg_weight,
                note_count_weight=note_count_weight, density_weight=density_weight,
                target_mid=target_mid, density_target=density_target,
            )
        elif self.voicing_selector is not None and chord_symbol:
            pitches = self.voicing_selector.select(
                chord_symbol, prev_voicing=prev_voicing, update_state=False,
                vl_weight=vl_weight, reg_weight=reg_weight,
                note_count_weight=note_count_weight, density_weight=density_weight,
                melody_pitch=melody_pitch, melody_role="top",
                target_mid=target_mid, density_target=density_target,
            )

        if pitches is not None:
            return (chord_symbol, pitches, is_onset), pitches

        # Fallback: no selector, chord not covered, or empty symbol.
        chord_info = self.decode_chord_token(chord_token, tk)
        fallback_pitches = chord_info[1] if chord_info[1] else prev_voicing
        return chord_info, fallback_pitches

    def _build_interleaved_prompt(
        self,
        note_token_hist: List[int],
        chord_tokens: List[int],
        frame: int,
        max_context_len: int,
        tokenizer: Optional[HooktheoryTokenizer] = None,
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """Build the `[BOS, chord_0, note_0, chord_1, note_1, ...]` prompt the
        online model expects, covering history up to (not including) `frame`.
        `chord_tokens` must be in `tokenizer`'s vocabulary (default the old one).

        Mutates `chord_tokens` in place, filling -1 gaps with SILENCE and
        padding up to `frame` — callers rely on that (generate_live reads
        chord_tokens again afterward for its commit-window logic).

        Returns the prompt tensor plus the trimmed note/chord histories that
        went into it.
        """
        tk = tokenizer or self.tokenizer
        # Fill any gaps in chord tokens with rests
        # If there are -1 values in chord_tokens, fill them with SILENCE
        # If chord_tokens length is less than frame, pad with SILENCE
        for i in range(len(chord_tokens)):
            if chord_tokens[i] == -1:
                chord_tokens[i] = tk.name_to_id["SILENCE"]
        if len(chord_tokens) < frame:
            chord_tokens.extend(
                [tk.name_to_id["SILENCE"]]
                * (frame - len(chord_tokens))
            )

        # Create initial prompt (up to target frame)
        # If chord_tokens length is greater than frame, take the first frame tokens
        # If chord_tokens length is less than frame, take all chord_tokens
        chord_token_hist = chord_tokens[:frame]
        # Trim beginning of context to avoid surpassing max length
        note_token_hist = note_token_hist[-max_context_len:]
        chord_token_hist = chord_token_hist[-max_context_len:]

        # The first frame of context can't be a "hold" of something the model
        # can't see, so rewrite it to the equivalent non-hold token.
        if note_token_hist and tk.is_note_hold(note_token_hist[0]):
            note_token_hist[0] = tk.note_hold_id_to_note_id(
                note_token_hist[0]
            )

        if chord_token_hist and tk.is_chord_hold(
            chord_token_hist[0]
        ):
            chord_token_hist[0] = tk.chord_hold_id_to_chord_id(
                chord_token_hist[0]
            )

        # Create interleaved sequence with BOS at start, EOS at end
        interleaved_tokens = [tk.bos_token]
        for i in range(min(len(chord_token_hist), len(note_token_hist))):
            interleaved_tokens.extend([chord_token_hist[i], note_token_hist[i]])

        prompt = torch.tensor(
            [interleaved_tokens], dtype=torch.long, device=self.device
        )
        return prompt, note_token_hist, chord_token_hist

    def _onset_chord_info(self, tokenizer: Optional[HooktheoryTokenizer] = None) -> Dict[int, Dict]:
        """Pitch classes and root of every chord onset token of a vocabulary
        (default the old one), computed once per vocabulary.

        Tokens whose symbol note_seq can't parse are left out -- they could
        never be completed into sounding pitches anyway.
        """
        tk = tokenizer or self.tokenizer
        if not hasattr(self, "_onset_info_caches"):
            self._onset_info_caches = {}
        if tk.num_tokens not in self._onset_info_caches:
            info = {}
            lo, hi = tk.chord_on_token_range
            for token in range(lo, hi + 1):
                symbol = tk.id_to_name[token].replace("CHORD_ON_", "")
                try:
                    pcs = {p % 12 for p in chord_symbol_pitches(symbol)}
                    pcs.add(note_seq.chord_symbol_bass(symbol) % 12)
                    root = note_seq.chord_symbol_root(symbol) % 12
                except (note_seq.ChordSymbolError, TypeError):
                    continue
                info[token] = {"symbol": symbol, "pitchClasses": sorted(pcs),
                               "root": root}
            self._onset_info_caches[tk.num_tokens] = info
        return self._onset_info_caches[tk.num_tokens]

    def chord_candidates(
        self,
        model_name: str,
        notes: List[NoteInfo],
        chord_tokens: List[int],
        frame: int,
        top_k: int = 200,
    ) -> List[Dict]:
        """The model's most likely chord onsets at `frame`, for chord completion.

        The model is asked which chord would start now (onset tokens
        only); instead of sampling one, the renormalised onset distribution
        is returned as a ranked list. The client fetches
        it ahead of time and, when the performer plays some notes, picks the
        most probable candidate containing them -- so completion needs no
        round trip at press time.

        Returns:
          up to `top_k` dicts, most probable first, with the chord symbol,
          onset and hold token ids, renormalised probability, pitch classes
          and root pitch class.
        """
        probs = self._onset_probs(model_name, notes, chord_tokens, frame)
        tk = self.tokenizer_for(model_name)
        lo, _ = tk.chord_on_token_range
        info = self._onset_chord_info(tk)
        result = []
        for idx in torch.argsort(probs, descending=True).tolist():
            token = lo + idx
            if token not in info:
                continue
            result.append({
                **info[token],
                "onsetToken": token,
                "holdToken": tk.chord_on_id_to_chord_id(token),
                "prob": float(probs[idx]),
            })
            if len(result) >= top_k:
                break
        return result

    def _onset_probs(self, model_name, notes, chord_tokens, frame) -> torch.Tensor:
        """The renormalised distribution over chord onsets at `frame` (one
        forward pass), indexed from the model's chord_on_token_range[0].
        `chord_tokens` must be in the model's vocabulary."""
        if self.onnx:
            raise NotImplementedError(
                "chord candidates need the PyTorch or MLX backend (they read "
                "logits from a single forward pass)")
        model = self.models[model_name]
        tk = self.tokenizer_for(model_name)
        note_token_hist = self.melody_to_frame_tokens(notes, frame).tolist()
        prompt, _, _ = self._build_interleaved_prompt(
            note_token_hist, chord_tokens, frame, self.max_frames - 1, tokenizer=tk
        )
        if self.mlx:
            logits = model(prompt)  # MLX model: full-sequence logits as torch
        else:
            with torch.no_grad():
                logits, _ = model.net(prompt, return_intermediates=True)
        lo, hi = tk.chord_on_token_range
        return torch.softmax(logits[0, -1, lo:hi + 1].float(), dim=-1)

    def chord_vocab_entries(self, model_name: Optional[str] = None) -> List[Dict]:
        """Every chord onset a model (default: the old vocabulary) can be
        offered, in a fixed order (by token id): symbol, pitch classes, root,
        onset and hold token ids. The order of chord_candidate_probs."""
        tk = self.tokenizer_for(model_name)
        info = self._onset_chord_info(tk)
        return [{**info[token], "onsetToken": token,
                 "holdToken": tk.chord_on_id_to_chord_id(token)}
                for token in sorted(info)]

    def set_voicing_lookup(self, label: str) -> None:
        """Make `label` (a key of voicing_lookup_paths) the active voicing
        lookup for everything that voices chords; loads it on first use and
        drops complete mode's fit indexes, which are built from the lookup."""
        if label not in self.voicing_lookup_paths:
            raise KeyError(label)
        if label not in self._voicing_selectors:
            self._voicing_selectors[label] = VoicingSelector(self.voicing_lookup_paths[label])
            logging.info("Loaded voicing lookup %r from %s", label, self.voicing_lookup_paths[label])
        self.voicing_selector = self._voicing_selectors[label]
        self.voicing_lookup_active = label
        self._voicing_fit_indexes = {}

    def voicing_fit(self, keys: List[int], held: List[int], ceiling: Optional[int],
                    model_name: Optional[str] = None) -> List[float]:
        """For every chord of chord_vocab_entries(model_name), in that order:
        the share of its stored voicings (by count) the completion could play
        around `keys` (realjam/voicing_fit.py). All zero without a voicing
        lookup. One index per vocabulary, built on its first call (about a
        second; server.py warms them up)."""
        entries = self.chord_vocab_entries(model_name)
        if self.voicing_selector is None:
            return [0.0] * len(entries)
        tk = self.tokenizer_for(model_name)
        if not hasattr(self, "_voicing_fit_indexes"):
            self._voicing_fit_indexes = {}
        if tk.num_tokens not in self._voicing_fit_indexes:
            from realchords.realjam.voicing_fit import VoicingFitIndex
            self._voicing_fit_indexes[tk.num_tokens] = VoicingFitIndex(
                [c["symbol"] for c in entries], self.voicing_selector._lookup)
        return self._voicing_fit_indexes[tk.num_tokens].fit(keys, held, ceiling)

    def chord_candidate_probs(self, model_name, notes, chord_tokens, frame) -> List[float]:
        """The model's probability for every chord of chord_vocab_entries, in
        that order (2026-11-17: the client gets all of them, not just a top
        200 -- the top 200 cover a median 88% of the probability, and for 6% of
        typical hand positions they hold no chord containing all held notes)."""
        probs = self._onset_probs(model_name, notes, chord_tokens, frame)
        tk = self.tokenizer_for(model_name)
        lo, _ = tk.chord_on_token_range
        return [round(float(probs[token - lo]), 6) for token in sorted(self._onset_chord_info(tk))]

    def _prepare_model(self, model):
        """Prepare model for inference.
        Args:
          model: the model to prepare
        """
        if self.mlx:
            model.eval()
            return model
        model.eval()
        model.to(self.device)
        if self.compile:
            model = torch.compile(model)
        return model

    def _load_online_model(self, model_path: str, is_rl: bool = False,
                           tokenizer: Optional[HooktheoryTokenizer] = None):
        """An online (decoder-only) model in `tokenizer`'s vocabulary (default
        the old one). An RL actor (.pth) is loaded into a copy of the MLE
        model in the old vocabulary; in another vocabulary into a model built
        from the args.yml next to it (its architecture config)."""
        tk = tokenizer or self.tokenizer
        override_args = {
            "DecoderTransformer.num_tokens": tk.num_tokens,
            "DecoderTransformer.pad_value": tk.pad_token,
        }

        if self.mlx:
            if MLXDecoderTransformer is None:
                raise ImportError(
                    "MLX backend requested, but MLX could not be imported"
                ) from MLX_IMPORT_ERROR
            if is_rl:
                return MLXDecoderTransformer.from_rl_checkpoint(
                    model_path,
                    tk,
                    override_args=override_args,
                )
            return MLXDecoderTransformer.from_checkpoint(
                model_path,
                tk,
                override_args=override_args,
            )

        if is_rl:
            if tk is self.tokenizer:
                skeleton = copy.deepcopy(self.mle_model)
            else:
                model_cls = argbind.bind(DecoderTransformer)
                args = argbind.load_args(os.path.join(os.path.dirname(model_path), "args.yml"))
                args.update(override_args)
                with argbind.scope(args):
                    skeleton = model_cls()
            return load_rl_model(
                model_path,
                skeleton,
                compile=False,
            )

        return load_gen_model_from_state_dict(
            model_path,
            DecoderTransformer,
            compile=False,
            override_args=override_args,
        )

    def _load_offline_model(self, model_path: str):
        override_args = {
            "EncoderDecoderTransformer.enc_num_tokens": self.tokenizer.num_tokens,
            "EncoderDecoderTransformer.dec_num_tokens": self.tokenizer.num_tokens,
            "EncoderDecoderTransformer.pad_value": self.tokenizer.pad_token,
        }

        if self.mlx:
            if MLXEncoderDecoderTransformer is None:
                raise ImportError(
                    "MLX backend requested, but MLX could not be imported"
                ) from MLX_IMPORT_ERROR
            return MLXEncoderDecoderTransformer.from_checkpoint(
                model_path,
                self.tokenizer,
                override_args=override_args,
            )

        return load_gen_model_from_state_dict(
            model_path,
            EncoderDecoderTransformer,
            compile=False,
            override_args=override_args,
        )

    def gen_online_model(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        seq_len: int,
        temperature: float,
        filter_fn=None,
        filter_kwargs=None,
    ) -> torch.Tensor:
        """Generate tokens using the online model.

        Args:
            model: The model to use for generation
            prompt: The input prompt tensor
            seq_len: Number of new tokens to generate
            temperature: Sampling temperature
            filter_fn: Logit filter to constrain sampling. Defaults to the
                standard interleaved chord/melody validity filter.
            filter_kwargs: Keyword arguments for `filter_fn`.

        Returns:
            Generated tokens tensor
        """
        if filter_fn is None:
            filter_fn = self.online_model_filter_fn
            filter_kwargs = self.online_model_filter_kwargs
        if filter_kwargs is None:
            filter_kwargs = dict()

        if self.onnx:
            # Use ONNX implementation
            onnx_init, onnx_step = model
            gen_tokens_np, _ = generate_tokens(
                onnx_init,
                onnx_step,
                prompt,
                temperature=temperature,
                n_new=seq_len,
                filter_logits_fn=filter_fn,
                filter_kwargs=filter_kwargs,
            )
            return torch.tensor(gen_tokens_np, device=prompt.device)
        else:
            return model.generate(
                prompts=prompt,
                seq_len=seq_len,
                temperature=temperature,
                cache_kv=True,
                filter_logits_fn=filter_fn,
                filter_kwargs=filter_kwargs,
            )

    def gen_commit_online_model_postfill(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        chord_tokens: List[int],
        note_token_hist: List[int],
        chord_token_hist: List[int],
        commit_frames: int,
        frame: int,
        temperature: float,
    ) -> torch.Tensor:
        """Generate melody frames for committed period using the online model.
        We implement this by *post-filling* the committed chord tokens.

        Here we are generating for the period that already got commited,
            so the chord tokens cannot change.

        We are:
        - Condition on melody and previously generated chord tokens.
        - Generate lookahead chord & melody frames into the future.
        - Force chords to not change by filling in the committed chord tokens.
        - Effectively generating the imaginated melody frames that will be played.
        - But we are not conditioned on the correct chord tokens,
            so this is just an approximation.
        """
        if self.onnx:
            prefix = "[ONNX Online model commit]"
        else:
            prefix = "[Online model commit]"

        logging.info(
            f"{prefix} len(chord_tokens): %s",
            len(chord_tokens),
        )
        logging.info(
            f"{prefix} len(note_token_hist): %s",
            len(note_token_hist),
        )
        logging.info(
            f"{prefix} len(chord_token_hist): %s",
            len(chord_token_hist),
        )
        logging.info(
            f"{prefix} Generating commit frames: %s",
            commit_frames,
        )
        logging.info(f"{prefix} Prompt: %s", prompt)

        start_time = time.time()
        generated_commit = self.gen_online_model(
            model=model,
            prompt=prompt,
            seq_len=commit_frames * 2,
            temperature=temperature,
        )
        end_time = time.time()

        logging.info(
            f"{prefix} Time taken: %s",
            end_time - start_time,
        )
        logging.info(f"{prefix} Generated: %s", generated_commit)
        generated_commit = generated_commit.cpu().numpy()
        generated_commit[0, 0::2] = chord_tokens[frame : frame + commit_frames]
        generated_commit_tensor = torch.tensor(
            generated_commit, device=prompt.device
        )
        logging.info(
            f"{prefix} Generated after chord fill in: %s",
            generated_commit_tensor,
        )
        prompt = torch.cat([prompt, generated_commit_tensor], dim=1)
        return prompt

    def gen_commit_online_model_teacher_forcing(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        chord_tokens: List[int],
        note_token_hist: List[int],
        chord_token_hist: List[int],
        commit_frames: int,
        frame: int,
        temperature: float,
        tokenizer: Optional[HooktheoryTokenizer] = None,
    ) -> torch.Tensor:
        """Generate melody frames for committed period using the online model.
        We implement this by *teacher-forcing* the committed chord tokens.

        Here we are generating for the period that already got commited,
            so the chord tokens cannot change.

        We are:
        - Condition on melody and previously generated chord tokens.
        - Generate lookahead chord & melody frames into the future.
        - Force chords to not change by filling in the committed chord tokens *at each step*.
        - Effectively generating the imaginated melody frames that will be played *at each step*.

        in this case it actually becomes a "delay of 1" model:
            It always knows what the other part (chord) will play in present time
            and predict itself (melody) at the next time step.
        """
        if self.onnx:
            prefix = "[ONNX Online model commit]"
        else:
            prefix = "[Online model commit]"
        logging.info(
            f"{prefix} len(chord_tokens): %s",
            len(chord_tokens),
        )
        logging.info(
            f"{prefix} len(note_token_hist): %s",
            len(note_token_hist),
        )
        logging.info(
            f"{prefix} len(chord_token_hist): %s",
            len(chord_token_hist),
        )
        logging.info(
            f"{prefix} Generating commit frames: %s",
            commit_frames,
        )
        logging.info(f"{prefix} Prompt: %s", prompt)
        conditions = torch.tensor(
            [chord_tokens[frame : frame + commit_frames]],
            device=prompt.device,
        )

        logging.info(f"{prefix} Conditions: %s", conditions)

        start_time = time.time()
        if self.onnx:
            # Use ONNX implementation
            # Note: The ONNX branch uses the function `generate_tokens_online` with the parameter `prompt`.
            # This differs from the PyTorch branch, which uses `generate_online` with the parameter `prompts`.
            onnx_init, onnx_step = model
            prompt_np = prompt.cpu().numpy()
            gen_commit_np, _ = generate_tokens_online(
                onnx_init,
                onnx_step,
                prompt,
                conditions,
                n_new=commit_frames * 2,
                temperature=temperature,
                filter_logits_fn=self._filters(tokenizer or self.tokenizer)["commit"],
                filter_kwargs=self.online_model_commit_filter_kwargs,
                reverse_condition_order=True,
            )
            generated_commit = torch.tensor(gen_commit_np, device=prompt.device)
        else:
            # prompt = [bos, chord, note, chord, note, ..., chord, note]
            # conditions = [chord1, chord2, ..., chordn]
            # generated_commit = [chord1, note1, chord2, note2, ..., chordn, noten]
            generated_commit = model.generate_online(
                prompts=prompt,
                conditions=conditions,
                seq_len=commit_frames * 2,
                temperature=temperature,
                cache_kv=True,
                filter_logits_fn=self._filters(tokenizer or self.tokenizer)["commit"],
                filter_kwargs=self.online_model_commit_filter_kwargs,
                reverse_condition_order=True,
            )

        end_time = time.time()

        logging.info(
            f"{prefix} Time taken: %s",
            end_time - start_time,
        )
        logging.info(f"{prefix} Generated: %s", generated_commit)

        if not self.onnx and not self.mlx:
            generated_commit = generated_commit.cpu().numpy()
            generated_commit_tensor = torch.tensor(
                generated_commit, device=prompt.device
            )
        else:
            generated_commit_tensor = generated_commit

        prompt = torch.cat([prompt, generated_commit_tensor], dim=1)
        return prompt

    def generate_live(
        self,
        model_name: str,
        notes: List[NoteInfo],
        chord_tokens: List[int],
        frame: int,
        lookahead: int,
        commitahead: int,
        temperature: float,
        silence_till: int,
        intro_set: bool,
        use_custom_voicings: bool = False,
        prev_voicing: Optional[List[int]] = None,
        vl_weight: Optional[float] = None,
        reg_weight: Optional[float] = None,
        note_count_weight: Optional[float] = None,
        density_weight: Optional[float] = None,
        target_mid: Optional[float] = None,
        density_target: Optional[float] = None,
        bass_line: bool = False,
    ) -> Tuple[List[ChordInfo], List[int], Optional[List[int]]]:
        """Generate chords for a given frame.

        - Condition on melody and previously generated chord tokens.
        - Generate lookahead frames into the future.
        - Force chords to not change during commitahead from previous generation.
        - Generate predicted intro chords before silence cutoff.

        Args:
          model_name: model to use, from MODELS keys
          notes: list of all note events in session
          chord_tokens: list of all chord tokens in session
          frame: frame to start generating at
          lookahead: how many frames into the future to generate
          commitahead: how many frames to leave chords unchanged since last call
          temperature: model sampling temperature
          silence_till: frames before generating with online model
          intro_set: if intro chords have been filled in by offline model
          use_custom_voicings: if True, voice chords via self.voicing_selector
            (data-driven, voice-leading-aware) instead of the fixed-octave
            rule, falling back per-chord when the lookup doesn't cover it.
          prev_voicing: pitches of the last voicing the caller actually used
            (regardless of which rule produced them), for voice-leading
            continuity across separate generate_live calls. Client-tracked
            since the server holds no per-session state.
          vl_weight, reg_weight, note_count_weight, density_weight,
          target_mid, density_target:
            Per-call overrides for VoicingSelector's scoring weights, only
            relevant when use_custom_voicings is True. ``None`` falls back
            to the selector's own constructed defaults.

        Returns:
          new_chords: list of ChordInfo, one for each frame in lookahead
          new_chord_tokens: list of chord tokens, one for each frame in lookahead
          intro_chord_tokens: list of chord tokens to fill in session beginning,
            only returned when first generated right before silence cutoff frame
        """
        # Wait until at most 4 frames before start to have enough context
        gen_start_frame = silence_till - min(lookahead, 4)
        if frame < gen_start_frame or frame == 0:
            logging.info(
                "Waiting for more user input before generating; frame: %s, lookahead:"
                " %s, silence_till: %s",
                frame,
                lookahead,
                silence_till,
            )
            return [], [], None

        logging.info("[Request] Generate chords for frame %s", frame)
        logging.info("[Request] Model name: %s", model_name)
        logging.info("[Request] Notes: %s", notes)
        logging.info("[Request] Chord tokens: %s", chord_tokens)
        logging.info("[Request] Lookahead: %s", lookahead)
        logging.info("[Request] Commitahead: %s", commitahead)
        logging.info("[Request] Temperature: %s", temperature)
        logging.info("[Request] Silence till: %s", silence_till)

        model = self.models[model_name]
        tk = self.tokenizer_for(model_name)  # chord_tokens are in its vocabulary

        # Convert notes to frame token format
        note_token_hist = self.melody_to_frame_tokens(notes, frame)
        note_token_hist = note_token_hist.tolist()

        # Use non-causal model to generate likely chords for introduction section
        intro_chord_tokens = None
        if not intro_set:
            note_tokens_bos_eos = (
                [self.tokenizer.bos_token]
                + note_token_hist
                + [self.tokenizer.eos_token]
            )
            enc_inputs = (
                torch.tensor(note_tokens_bos_eos, dtype=torch.long)
                .unsqueeze(0)
                .to(self.device)
            )
            gen_inputs = torch.full(
                (enc_inputs.shape[0], 1),
                self.tokenizer.bos_token,
                dtype=torch.long,
                device=self.device,
            )

            start_time = time.time()
            intro_chord_tokens = self.non_causal_model.generate(
                seq_in=enc_inputs,
                seq_out_start=gen_inputs,
                seq_len=frame,
                cache_kv=True,
                filter_logits_fn=self.non_causal_model_filter_fn,
                filter_kwargs=self.non_causal_model_filter_kwargs,
            )
            end_time = time.time()

            intro_chord_tokens = (
                intro_chord_tokens.squeeze(0).detach().cpu().tolist()
            )
            # The offline model writes the old vocabulary: into the model's
            intro_chord_tokens = self.remap_chord_tokens(intro_chord_tokens, self.tokenizer, tk)

            chord_tokens[:frame] = intro_chord_tokens
            logging.info("[Offline model] Input: %s", enc_inputs)
            logging.info("[Offline model] Output: %s", intro_chord_tokens)
            logging.info(
                "[Offline model] Time taken: %s", end_time - start_time
            )
        prompt, note_token_hist, chord_token_hist = (
            self._build_interleaved_prompt(
                note_token_hist, chord_tokens, frame, self.max_frames - lookahead,
                tokenizer=tk,
            )
        )

        start_idx = prompt.shape[1]
        new_chord_tokens = None
        # First predict up to commit point to get predicted user notes
        # chord_tokens represents the history of generated chords + future chords from the frontend
        # If frame > length of chord_tokens, it means the current frame (time of this frontend message)
        # has exceeded previously generated chords, indicating a missed generation period
        # Therefore, if we reduce commitahead or commit_frames
        commit_frames = len(chord_tokens) - lookahead + commitahead - frame
        commit_frames = max(commit_frames, 0)  # If frame is past commit point
        logging.debug("commit_frames: %s", commit_frames)
        logging.debug("lookahead: %s", lookahead)
        if commit_frames >= lookahead:
            # Can just look up new tokens in cache if lookahead was decreased by user
            # Otherwise will try to generate <= 0 tokens and will break model api
            new_chord_tokens = np.array(
                chord_tokens[frame : frame + min(commit_frames, lookahead)]
            )
        else:
            if commit_frames > 0:
                prompt = self.gen_commit_online_model_teacher_forcing(
                    model,
                    prompt,
                    chord_tokens,
                    note_token_hist,
                    chord_token_hist,
                    commit_frames,
                    frame,
                    temperature,
                    tokenizer=tk,
                )
            # -1 because we don't need to generate the last melody token
            # So here we are saving some time by not generating the last token
            seq_len = lookahead * 2 - commit_frames * 2 - 1

            if self.onnx:
                prefix = "[ONNX Online model]"
            else:
                prefix = "[Online model]"

            logging.info(
                f"{prefix} Generating seq_len: %s, prompt: %s",
                seq_len,
                prompt,
            )
            start_time = time.time()
            generated_lookahead = self.gen_online_model(
                model=model,
                prompt=prompt,
                seq_len=seq_len,
                temperature=temperature,
                filter_fn=self._filters(tk)["online"],
                filter_kwargs=self.online_model_filter_kwargs,
            )
            end_time = time.time()
            logging.info(f"{prefix} Time taken: %s", end_time - start_time)
            logging.info(f"{prefix} Generated: %s", generated_lookahead)

            if self.onnx:
                prompt_np = prompt.cpu().numpy()
                all_chord_tokens_np = np.concatenate(
                    [prompt_np, generated_lookahead.cpu().numpy()], axis=1
                )
                new_chord_tokens = all_chord_tokens_np[0, start_idx::2]
            else:
                all_chord_tokens = torch.cat(
                    [prompt, generated_lookahead], dim=1
                )
                new_chord_tokens = (
                    all_chord_tokens[0, start_idx::2].cpu().numpy()
                )

        # Decode and return new chord tokens. Above a walking bass line the
        # voicing always comes from the lookup (a voicing clear of the line)
        if (use_custom_voicings or bass_line) and self.voicing_selector is not None:
            new_chords = []
            voicing_state = prev_voicing
            for i, chord_tok in enumerate(new_chord_tokens):
                # new_chord_tokens[i] corresponds to absolute frame `frame + i`
                # -- look up whatever melody note is sounding there so the
                # voicing selector can keep chords from climbing above it.
                melody_pitch = _active_melody_pitch(notes, frame + i)
                chord_info, voicing_state = self.decode_chord_token_custom_voicing(
                    chord_tok, voicing_state,
                    vl_weight=vl_weight, reg_weight=reg_weight,
                    note_count_weight=note_count_weight, density_weight=density_weight,
                    target_mid=target_mid, density_target=density_target,
                    melody_pitch=melody_pitch, bass_line=bass_line, tokenizer=tk,
                )
                new_chords.append(chord_info)
        else:
            new_chords = [
                self.decode_chord_token(chord_tok, tk) for chord_tok in new_chord_tokens
            ]

        logging.info("[Response] Chords: %s", new_chords)
        logging.info("[Response] Chord tokens: %s", new_chord_tokens.tolist())
        logging.info("[Response] Intro chord tokens: %s", intro_chord_tokens)
        logging.info("[Response] Frame: %s", frame)
        return new_chords, new_chord_tokens.tolist(), intro_chord_tokens
