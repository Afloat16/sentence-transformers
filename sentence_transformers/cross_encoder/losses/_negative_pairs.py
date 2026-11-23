from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence

import torch


class _InBatchNegatives:
    """Snapshot candidate columns without expanding every anchor/candidate combination."""

    def __init__(self, batch_size: int, candidates: Sequence[Sequence[str]], num_negatives: int | None):
        self.batch_size = batch_size
        self.candidates = tuple(tuple(column) for column in candidates)
        self.count = (batch_size - 1) * len(self.candidates)
        self.sampled_indices: list[list[int]] | None = None
        if num_negatives is not None and num_negatives < self.count:
            self.count = num_negatives
            candidate_count = batch_size * len(self.candidates)
            candidate_rows = torch.arange(candidate_count) % batch_size
            # Bound the temporary probability matrix even with large effective batches.
            # Each row still samples uniformly without replacement, excluding its own
            # positive and hard negatives. Larger batches may draw a different seeded
            # sequence than a single full-matrix multinomial call.
            rows_per_chunk = max(1, 1_048_576 // candidate_count)
            self.sampled_indices = []
            for begin in range(0, batch_size, rows_per_chunk):
                rows = torch.arange(begin, min(begin + rows_per_chunk, batch_size))
                allowed = candidate_rows.unsqueeze(0) != rows.unsqueeze(1)
                self.sampled_indices.extend(torch.multinomial(allowed.float(), num_negatives).tolist())

    def document(self, anchor_row: int, negative_column: int) -> str:
        if self.sampled_indices is not None:
            column, row = divmod(self.sampled_indices[anchor_row][negative_column], self.batch_size)
        else:
            column, row = divmod(negative_column, self.batch_size - 1)
            row += row >= anchor_row
        return self.candidates[column][row]

    def columns(self) -> Iterator[list[str]]:
        for column in range(self.count):
            yield [self.document(row, column) for row in range(self.batch_size)]


class _CrossEncoderPairs(Sequence[tuple[str, str]]):
    """Materialize only the requested pair slice, including during GradCache replay."""

    def __init__(self, inputs: Sequence[Sequence[str]], num_negatives: int | None):
        self.anchors = tuple(inputs[0])
        self.negatives = _InBatchNegatives(len(self.anchors), inputs[1:], num_negatives)

    def __len__(self) -> int:
        return len(self.anchors) * (len(self.negatives.candidates) + self.negatives.count)

    def __getitem__(self, index: int | slice) -> tuple[str, str] | list[tuple[str, str]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("pair index out of range")
        column, row = divmod(index, len(self.anchors))
        if column == 0:
            document = self.negatives.candidates[0][row]
        elif column <= self.negatives.count:
            document = self.negatives.document(row, column - 1)
        else:
            document = self.negatives.candidates[column - self.negatives.count][row]
        return self.anchors[row], document
