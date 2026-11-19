"""Synthetic trips with known structure (distance, rush hours): tests and the demo model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taximon.data import TARGET

ZONES = {z: ("Manhattan" if z < 100 else "Queens") for z in range(1, 264)}


def synthetic_trips(n=4000, seed=0, speed=12.0):
    rng = np.random.default_rng(seed)
    pu = rng.integers(1, 200, n)
    do = rng.integers(1, 200, n)
    dist = rng.gamma(2.0, 1.5, n).astype("float32") + 0.1
    hour = rng.integers(0, 24, n)
    rush = np.isin(hour, [8, 9, 17, 18])
    minutes = 60 * dist / (speed * np.where(rush, 0.6, 1.0)) * rng.lognormal(0, 0.15, n)
    return pd.DataFrame(
        {
            "PULocationID": pu.astype("int32"),
            "DOLocationID": do.astype("int32"),
            "pu_borough": [ZONES[z] for z in pu],
            "do_borough": [ZONES[z] for z in do],
            "RatecodeID": np.ones(n, "int32"),
            "VendorID": rng.integers(1, 3, n).astype("int32"),
            "trip_distance": dist,
            "hour": hour.astype("int8"),
            "weekday": rng.integers(0, 7, n).astype("int8"),
            "passenger_count": rng.integers(1, 4, n).astype("float32"),
            TARGET: minutes.astype("float32"),
        }
    )


def demo_registry(root="models", n=20_000) -> None:
    """A small model trained on synthetic trips, so the service runs without the real data."""
    from taximon import registry
    from taximon.model import ETAModel

    train = synthetic_trips(n)
    registry.register(ETAModel(rounds=100).fit(train), train, ZONES, root=root, version="demo")
    registry.promote("demo", root=root)


if __name__ == "__main__":
    demo_registry()
