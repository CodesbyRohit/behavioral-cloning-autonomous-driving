"""Streaming mini-batch loader: images are read from disk per batch, never all at once."""

from __future__ import annotations

import math

import keras
import numpy as np

from bcdrive.config import BalanceConfig, PreprocessConfig
from bcdrive.data.balance import balanced_indices
from bcdrive.data.dataset import Sample
from bcdrive.preprocessing.augment import Augmenter
from bcdrive.preprocessing.image import load_image, preprocess


class DrivingSequence(keras.utils.PyDataset):
    """Yields (images float32 (B, 66, 200, 3), steering float32 (B, 1)).

    Training: ``augmenter`` and ``balance`` set, ``shuffle=True``; the balanced subset and
    order are re-drawn each epoch. Validation/test: no augmenter, no balancing, fixed order.
    Randomness is derived from (seed, epoch, batch index) so results do not depend on
    which worker thread loads which batch.
    """

    def __init__(self, samples: list[Sample], batch_size: int, preprocess_cfg: PreprocessConfig,
                 augmenter: Augmenter | None = None, balance: BalanceConfig | None = None,
                 shuffle: bool = False, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        if not samples:
            raise ValueError("DrivingSequence needs at least one sample")
        self.samples = samples
        self.steering = np.array([s.steering for s in samples], dtype=np.float32)
        self.batch_size = batch_size
        self.preprocess_cfg = preprocess_cfg
        self.augmenter = augmenter
        self.balance = balance
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._order = self._draw_order()

    def _draw_order(self) -> np.ndarray:
        rng = np.random.default_rng((self.seed, self.epoch, 0xBA1))
        idx = (balanced_indices(self.steering, self.balance, rng)
               if self.balance is not None else np.arange(len(self.samples)))
        return rng.permutation(idx) if self.shuffle else idx

    def __len__(self) -> int:
        return math.ceil(len(self._order) / self.batch_size)

    def __getitem__(self, i: int):
        batch = self._order[i * self.batch_size:(i + 1) * self.batch_size]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        xs, ys = [], []
        for j in batch:
            s = self.samples[int(j)]
            img, steer = load_image(s.image_path), float(s.steering)
            if self.augmenter is not None:
                img, steer = self.augmenter(img, steer, rng)
            xs.append(preprocess(img, self.preprocess_cfg))
            ys.append(steer)
        return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32).reshape(-1, 1)

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._order = self._draw_order()

    @property
    def epoch_size(self) -> int:
        return len(self._order)
