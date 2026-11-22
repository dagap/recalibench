"""A 200-row synthetic dataset with a known true probability per row."""

from pathlib import Path

import numpy as np
import pandas as pd

WORDS = ("alpha", "beta", "gamma", "delta", "omega")


class ToyDataset:
    def __init__(self, name: str, n_rows: int = 200, seed: int = 0) -> None:
        self.name = name
        self.n_rows = n_rows
        self.seed = seed

    def download(self, raw_dir: Path) -> None:
        rng = np.random.default_rng(self.seed)
        signal = rng.normal(0.0, 1.5, self.n_rows)
        true_p = 1.0 / (1.0 + np.exp(-signal))
        label = (rng.random(self.n_rows) < true_p).astype(np.int64)
        words = rng.choice(np.array(WORDS), self.n_rows)
        df = pd.DataFrame(
            {
                "id": [f"toy-{i:04d}" for i in range(self.n_rows)],
                "text": [f"{w} item {i}" for i, w in enumerate(words)],
                "signal": signal,
                "label": label,
            }
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_dir / "toy.csv", index=False)

    def load(self, raw_dir: Path) -> pd.DataFrame:
        return pd.read_csv(raw_dir / "toy.csv")

    def schema(self) -> dict[str, str]:  # pandas 3 reads text as "str", not "object"
        return {"id": "str", "text": "str", "signal": "float64", "label": "int64"}
