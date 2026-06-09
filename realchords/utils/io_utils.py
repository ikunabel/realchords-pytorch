import json
from typing import Any, List, Sequence


class JSONLIndexer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.offsets = []
        self._build_index()

    def _build_index(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line.encode("utf-8"))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Any:
        return self(idx)

    def __call__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(
                f"Index {idx} out of range for file with {len(self)} lines"
            )
        with open(self.filepath, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            return json.loads(line)


class CombinedJSONLIndexer:
    """Index multiple JSONL files as one contiguous dataset."""

    def __init__(self, filepaths: Sequence[str]):
        self.indexers = [JSONLIndexer(str(path)) for path in filepaths]
        self._sizes = [len(indexer) for indexer in self.indexers]
        self._starts: List[int] = []
        start = 0
        for size in self._sizes:
            self._starts.append(start)
            start += size

    def __len__(self) -> int:
        return sum(self._sizes)

    def __getitem__(self, idx: int) -> Any:
        return self(idx)

    def __call__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for combined dataset with {len(self)} lines"
            )
        for start, size, indexer in zip(self._starts, self._sizes, self.indexers):
            if idx < start + size:
                return indexer(idx - start)
        raise IndexError(
            f"Index {idx} out of range for combined dataset with {len(self)} lines"
        )


def save_jsonl(data: Any, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
