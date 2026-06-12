"""Subclasses extending HooktheoryDataset, HooktheoryTokenizer and WeightedJointDataset to return segments instead of random crops of songs."""

import bisect
import time
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from realchords.constants import DATA_PATH, FRAME_PER_BEAT
from realchords.dataset.hooktheory_dataloader import HooktheoryDataset
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.dataset.weighted_joint_dataset import WeightedJointDataset
from realchords.utils.logging_utils import logger
from realchords.utils.sequence_utils import pad_and_get_mask


class SegmentHooktheoryTokenizer(HooktheoryTokenizer):
    """Tokenizer with frame counting for segment indexing."""

    def num_frames(self, example: dict) -> int:
        """Return encoded sequence length without building frame arrays."""
        melody = example["annotations"]["melody"]
        chord = example["annotations"]["harmony"]

        melody = sorted(melody, key=lambda x: x["onset"])
        chord = sorted(chord, key=lambda x: x["onset"])

        melody_offset_last = melody[-1]["offset"]
        chord_offset_last = chord[-1]["offset"]
        melody_onset_last = melody[-1]["onset"]
        chord_onset_last = chord[-1]["onset"]
        duration = int(
            max(melody_offset_last, chord_offset_last) * self.frame_per_beat
        )
        if int(melody_offset_last * self.frame_per_beat) == int(
            melody_onset_last * self.frame_per_beat
        ):
            duration += 1
        elif int(chord_offset_last * self.frame_per_beat) == int(
            chord_onset_last * self.frame_per_beat
        ):
            duration += 1
        return duration


class SegmentHooktheoryDataset(HooktheoryDataset):
    """Expose each song segment as its own dataset index."""

    def __init__(self, segment_stride: Optional[int] = None, **kwargs):
        """Initialize segmented dataset.

        Args:
            segment_stride: Frame stride between window starts. Defaults to
                ``max_len_per_part`` (non-overlapping tiles). Use a smaller
                value for sliding windows with overlap.
        """
        self.segment_stride = segment_stride
        super().__init__(**kwargs)
        self.tokenizer = SegmentHooktheoryTokenizer(
            frame_per_beat=self.frame_per_beat,
            chord_names=self.chord_names,
        )
        self._segment_starts: list[list[int]] = []
        self._segment_offsets: list[int] = [0]
        self._build_segment_index()

    def _effective_stride(self) -> int:
        return (
            self.segment_stride
            if self.segment_stride is not None
            else self.max_len_per_part
        )

    def _window_starts(self, num_frames: int) -> list[int]:
        """Return frame start indices for all windows in a song."""
        window_len = self.max_len_per_part
        stride = self._effective_stride()

        if num_frames <= window_len:
            return [0]

        if stride >= window_len:
            num_segments = max(1, num_frames // window_len)
            return [segment_idx * window_len for segment_idx in range(num_segments)]

        starts = list(range(0, num_frames - window_len + 1, stride))
        last_start = num_frames - window_len
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def __len__(self) -> int:
        return self._segment_offsets[-1]

    def _resolve_segment_idx(self, idx: int) -> tuple[int, int]:
        song_idx = bisect.bisect_right(self._segment_offsets, idx) - 1
        segment_idx = idx - self._segment_offsets[song_idx]
        return song_idx, segment_idx

    def _build_segment_index(self) -> None:
        """One dataset index per segment window."""
        window_len = self.max_len_per_part
        stride = self._effective_stride()
        segments_per_song: list[int] = []
        logger.info(
            "Building segment index for %d songs (window_len=%d, stride=%d)...",
            len(self.data),
            window_len,
            stride,
        )
        for song_idx in tqdm(range(len(self.data)), desc="Segment index"):
            num_frames = self.tokenizer.num_frames(self.data[song_idx])
            starts = self._window_starts(num_frames)
            self._segment_starts.append(starts)
            segments_per_song.append(len(starts))

        offsets = [0]
        for count in segments_per_song:
            offsets.append(offsets[-1] + count)
        self._segments_per_song = segments_per_song
        self._segment_offsets = offsets
        logger.info(
            "Built segment index: %d segments from %d songs",
            offsets[-1],
            len(self.data),
        )

    def crop_segment(
        self,
        melody: torch.Tensor,
        chord: torch.Tensor,
        start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Crop one segment window starting at ``start`` (frame indices)."""
        assert melody.shape[0] == chord.shape[0]

        segment_len = self.max_len_per_part
        if melody.shape[0] <= segment_len:
            return melody, chord, 0, melody.shape[0]

        end = start + segment_len
        if end > melody.shape[0]:
            start = max(0, melody.shape[0] - segment_len)
            end = start + segment_len
        return melody[start:end], chord[start:end], start, end

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        song_idx, segment_idx = self._resolve_segment_idx(idx)
        start = self._segment_starts[song_idx][segment_idx]
        item = self.process_item(self.data[song_idx])
        melody = torch.tensor(item["melody"])
        chord = torch.tensor(item["chord"])
        melody, chord, segment_start, segment_end = self.crop_segment(
            melody, chord, start
        )

        output = self.serialize(melody, chord)

        if self.model_type == "encoder_decoder":
            inputs_pad, inputs_mask = pad_and_get_mask(
                output["inputs"],
                self.max_len_per_part + 2,
            )
            targets_pad, targets_mask = pad_and_get_mask(
                output["targets"],
                self.max_len_per_part + 2,
            )
            output["inputs"] = inputs_pad
            output["inputs_mask"] = inputs_mask
            output["targets"] = targets_pad
            output["targets_mask"] = targets_mask
        elif self.model_type == "decoder_only":
            targets_pad, targets_mask = pad_and_get_mask(
                output["targets"],
                self.max_len_per_part * 2 + 2,
            )
            output["targets"] = targets_pad
            output["targets_mask"] = targets_mask
        elif self.model_type == "decoder_only_single":
            targets_pad, targets_mask = pad_and_get_mask(
                output["targets"],
                self.max_len_per_part + 2,
            )
            output["targets"] = targets_pad
            output["targets_mask"] = targets_mask

        output["song_url"] = item["song_url"]
        output["segment_idx"] = segment_idx
        output["segment_start"] = segment_start
        output["segment_end"] = segment_end
        return output


class SegmentWeightedJointDataset(WeightedJointDataset):
    """Weighted joint dataset using segmented Hooktheory datasets."""

    def __init__(self, segment_stride: Optional[int] = None, **kwargs):
        self.segment_stride = segment_stride
        super().__init__(**kwargs)

    def _load_datasets(
        self,
        max_len: int,
        model_type: str,
        model_part: str,
        chord_names_path: str,
        data_path: str,
        frame_per_beat: int,
        load_augmented_chord_names: bool,
        num_workers: int,
    ):
        """Load individual datasets with segment indexing."""
        stride = self.segment_stride
        print(f"=== Loading Segment Weighted Joint Dataset ===")
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"Weights: {self.weights}")
        print(f"Split: {self.split}")
        print(f"Augmentation: {self.data_augmentation}")
        print(f"Segment stride: {stride if stride is not None else 'non-overlapping'}")
        print()

        for i, dataset_name in enumerate(self.datasets):
            if dataset_name not in self.cache_dirs:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            default_cache_dir = self.cache_dirs[dataset_name]
            print(f"Loading {dataset_name} (weight: {self.weights[i]:.1%})...")
            start_time = time.time()

            try:
                dataset = SegmentHooktheoryDataset(
                    segment_stride=stride,
                    cache_dir=default_cache_dir,
                    split=self.split,
                    data_augmentation=self.data_augmentation,
                    max_len=max_len,
                    model_type=model_type,
                    model_part=model_part,
                    chord_names_path=chord_names_path,
                    data_path=data_path,
                    frame_per_beat=frame_per_beat,
                    load_augmented_chord_names=load_augmented_chord_names,
                    num_workers=num_workers,
                )

                self.individual_datasets.append(dataset)
                load_time = time.time() - start_time

                self.dataset_info.append(
                    {
                        "name": dataset_name,
                        "size": len(dataset),
                        "weight": self.weights[i],
                        "load_time": load_time,
                        "start_idx": sum(
                            len(d) for d in self.individual_datasets[:-1]
                        ),
                        "end_idx": sum(
                            len(d) for d in self.individual_datasets
                        ),
                    }
                )

                print(
                    f"  {dataset_name}: {len(dataset):,} items ({load_time:.1f}s)"
                )

            except Exception as e:
                print(f"  {dataset_name}: Failed to load - {e}")
                raise


def create_segment_weighted_joint_dataset(
    datasets: List[str] = ["hooktheory"],
    weights: Optional[List[float]] = None,
    chord_names_path: str = None,
    split: str = "train",
    data_augmentation: bool = True,
    max_len: int = 512,
    model_type: str = "decoder_only",
    model_part: str = "chord",
    seed: int = 42,
    data_path: str = DATA_PATH,
    frame_per_beat: int = FRAME_PER_BEAT,
    load_augmented_chord_names: bool = False,
    num_workers: int = 8,
    train_samples_multiplier: float = 100.0,
    max_train_samples: Optional[int] = None,
    sampler_chunk_size: int = 4096,
    segment_stride: Optional[int] = None,
) -> SegmentWeightedJointDataset:
    """Create a weighted joint dataset with per-segment indexing."""
    return SegmentWeightedJointDataset(
        datasets=datasets,
        weights=weights,
        chord_names_path=chord_names_path,
        split=split,
        data_augmentation=data_augmentation,
        max_len=max_len,
        model_type=model_type,
        model_part=model_part,
        seed=seed,
        data_path=data_path,
        frame_per_beat=frame_per_beat,
        load_augmented_chord_names=load_augmented_chord_names,
        num_workers=num_workers,
        train_samples_multiplier=train_samples_multiplier,
        max_train_samples=max_train_samples,
        sampler_chunk_size=sampler_chunk_size,
        segment_stride=segment_stride,
    )
