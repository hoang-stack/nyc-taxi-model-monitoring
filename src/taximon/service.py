"""Prediction service with built-in monitoring hooks.

- ``POST /predict``      validated trip request -> ETA in minutes, model version, request id
- ``POST /feedback``     the real duration once the trip has ended (joined later by request id)
- ``GET  /metrics``      Prometheus: request counts, latency, ETA distribution, and live PSI of
                         each feature over the last ``WINDOW`` requests against the profile
                         shipped with the model
- ``GET  /health``       model version and how many requests are in the drift window

Every prediction and every feedback is appended to a JSONL log (one file per day), which is
what ``taximon monitor`` reads to compute drift and realised error offline.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from taximon import profile as profiles
from taximon.data import CATEGORICAL, NUMERIC
from taximon.model import ETAModel

WINDOW = 1000


class Trip(BaseModel):
    pickup_datetime: datetime
    pu_location_id: int = Field(ge=1, le=263, description="TLC taxi zone of the pickup")
    do_location_id: int = Field(ge=1, le=263, description="TLC taxi zone of the drop-off")
    trip_distance: float = Field(gt=0, le=100, description="planned route length, miles")
    passenger_count: int = Field(default=1, ge=0, le=9)
    ratecode_id: int = Field(default=1, ge=1, le=6)
    vendor_id: int = Field(default=2, ge=1, le=6)


class Prediction(BaseModel):
    request_id: str
    eta_minutes: float
    model_version: str


class Feedback(BaseModel):
    request_id: str
    actual_minutes: float = Field(gt=0, le=600)


def create_app(model_dir: str | Path | None = None, log_dir: str | Path | None = None) -> FastAPI:
    model_dir = Path(model_dir or os.environ.get("TAXIMON_MODEL", "models/current"))
    log_dir = Path(log_dir or os.environ.get("TAXIMON_LOGS", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    model = ETAModel.load(model_dir)
    prof = profiles.load(model_dir / "profile.json")
    zones = {int(k): v for k, v in json.loads((model_dir / "zones.json").read_text()).items()}
    version = str(model.meta.get("version", model_dir.resolve().name))

    reg = CollectorRegistry()
    requests = Counter("taximon_predictions", "Predictions served", registry=reg)
    feedback = Counter("taximon_feedback", "Ground-truth durations received", registry=reg)
    latency = Histogram("taximon_latency_seconds", "Prediction latency", registry=reg)
    eta_hist = Histogram(
        "taximon_eta_minutes",
        "Predicted ETA",
        buckets=(2, 5, 10, 15, 20, 30, 45, 60, 90, 120),
        registry=reg,
    )
    psi_gauge = Gauge(
        "taximon_feature_psi",
        f"PSI of each feature over the last {WINDOW} requests vs the training profile",
        ["feature"],
        registry=reg,
    )
    window: deque[dict] = deque(maxlen=WINDOW)

    app = FastAPI(title="taximon ETA service", version=version)

    def write_log(kind: str, record: dict) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        with open(log_dir / f"{kind}-{day}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def to_frame(trip: Trip) -> pd.DataFrame:
        t = trip.pickup_datetime
        row = {
            "PULocationID": trip.pu_location_id,
            "DOLocationID": trip.do_location_id,
            "pu_borough": zones.get(trip.pu_location_id, "Unknown"),
            "do_borough": zones.get(trip.do_location_id, "Unknown"),
            "RatecodeID": trip.ratecode_id,
            "VendorID": trip.vendor_id,
            "trip_distance": trip.trip_distance,
            "hour": t.hour,
            "weekday": t.weekday(),
            "passenger_count": float(trip.passenger_count),
        }
        return pd.DataFrame([row])

    @app.post("/predict", response_model=Prediction)
    def predict(trip: Trip) -> Prediction:
        start = time.perf_counter()
        X = to_frame(trip)
        eta = float(model.predict(X)[0])
        latency.observe(time.perf_counter() - start)
        requests.inc()
        eta_hist.observe(eta)
        rid = uuid.uuid4().hex
        features = X.iloc[0].to_dict()
        window.append({**features, "prediction": eta})
        write_log(
            "predictions",
            {
                "request_id": rid,
                "ts": datetime.now().isoformat(),
                "model_version": version,
                "features": features,
                "eta_minutes": eta,
            },
        )
        return Prediction(request_id=rid, eta_minutes=round(eta, 2), model_version=version)

    @app.post("/feedback")
    def give_feedback(fb: Feedback) -> dict:
        feedback.inc()
        write_log("feedback", {**fb.model_dump(), "ts": datetime.now().isoformat()})
        return {"ok": True}

    @app.get("/metrics")
    def metrics() -> Response:
        if len(window) >= 100:  # too few requests make PSI meaningless
            live = pd.DataFrame(list(window))
            for c in CATEGORICAL + NUMERIC + ["prediction"]:
                psi_gauge.labels(feature=c).set(profiles.psi(prof, c, live[c]))
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict:
        if model is None:
            raise HTTPException(503, "no model")
        return {"status": "ok", "model_version": version, "drift_window": len(window)}

    return app


def main() -> None:  # pragma: no cover - exercised through `taximon serve`
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
