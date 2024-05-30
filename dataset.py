import pickle
from math import ceil
from typing import Generator, Any

import scipy
from scipy.sparse.csr import csr_matrix
import numpy as np
from numpy.random import choice as numpy_choice
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from datasets import IterableDataset


class GloveDataset(LightningDataModule):
    train_dataset = None
    valid_dataset = None
    test_dataset = None

    cooccurrence_matrix = None

    def __init__(
        self,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        self._load_cooccurrences()

    def setup(self, stage: str) -> None:
        self.train_dataset = IterableDataset.from_generator(self._sample_generator)

    def get_max_batches(self) -> int:
        return ceil(self.cooccurrence_matrix.size / self.batch_size)

    @staticmethod
    def _get_csr_matrix_indices(S) -> Generator:
        major_dim, minor_dim = S.shape
        minor_indices = S.indices

        major_indices = np.empty(len(minor_indices), dtype=S.indices.dtype)
        scipy.sparse._sparsetools.expandptr(major_dim, S.indptr, major_indices)

        return zip(major_indices, minor_indices)

    def _sample_generator(self) -> Generator[dict, Any, None]:
        csr_indexes = list(self._get_csr_matrix_indices(self.cooccurrence_matrix))
        possible_indexes = list(range(len(csr_indexes)))
        normalized_cooccurrence_matrix = (
            self.cooccurrence_matrix / self.cooccurrence_matrix.sum()
        )
        samples_indexes = numpy_choice(
            possible_indexes,
            size=len(csr_indexes),
            p=normalized_cooccurrence_matrix.data,
        )

        for choice_index in samples_indexes:
            choice = csr_indexes[choice_index]
            occurrence = self.cooccurrence_matrix[choice]
            yield {
                "center": choice[0],
                "context": choice[1],
                "cooccurrence": occurrence,
            }

    def _load_cooccurrences(self) -> csr_matrix:
        with open("../data/multiprocess_cooc_matrix_lil.pk", "rb") as file:
            m = pickle.load(file)
        self.cooccurrence_matrix = m.tocsr()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)


if __name__ == "__main__":
    batch_size = 5
    window_size = 2
    samples = 10

    ds = GloveDataset(
        batch_size=batch_size,
    )
    ds.setup("fit")

    dl = ds.train_dataloader()
    for e in dl:
        print(f"{e=}")
        break
