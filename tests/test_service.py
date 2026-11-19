import numpy as np
import pytest

from taximon import monitor_logs, registry
from taximon.data import TARGET
from taximon.model import ETAModel
from taximon.synthetic import ZONES, synthetic_trips


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    train = synthetic_trips()
    model = ETAModel(rounds=30).fit(train)
    registry.register(model, train, ZONES, root=root, version="v1")
    registry.promote("v1", root=root)
    return root


def test_model_learns_distance_and_rush_hour():
    train, test = synthetic_trips(seed=0), synthetic_trips(seed=1)
    pred = ETAModel(rounds=60).fit(train).predict(test)
    rel = np.mean(np.abs(pred - test[TARGET]) / test[TARGET])
    assert rel < 0.2


def test_registry_promote_and_rollback(deployed, tmp_path):
    train = synthetic_trips(seed=2)
    registry.register(ETAModel(rounds=5).fit(train), train, ZONES, root=deployed, version="v2")
    registry.promote("v2", root=deployed)
    assert [v["version"] for v in registry.versions(deployed) if v["current"]] == ["v2"]
    registry.promote("v1", root=deployed)  # rollback
    assert [v["version"] for v in registry.versions(deployed) if v["current"]] == ["v1"]
    with pytest.raises(FileNotFoundError):
        registry.promote("v9", root=deployed)


def test_service_predicts_logs_and_exposes_drift(deployed, tmp_path):
    from fastapi.testclient import TestClient

    from taximon.service import create_app

    client = TestClient(create_app(deployed / "current", tmp_path))
    assert client.get("/health").json()["model_version"] == "v1"
    bad = client.post(
        "/predict",
        json={
            "pickup_datetime": "2020-04-01T08:00",
            "pu_location_id": 999,
            "do_location_id": 1,
            "trip_distance": 2,
        },
    )
    assert bad.status_code == 422  # schema validation rejects an impossible zone
    rng = np.random.default_rng(5)
    for i in range(250):  # long trips only: the distance feature drifts
        r = client.post(
            "/predict",
            json={
                "pickup_datetime": f"2020-04-01T{i % 24:02d}:00",
                "pu_location_id": int(rng.integers(1, 200)),
                "do_location_id": int(rng.integers(1, 200)),
                "trip_distance": float(rng.uniform(15, 30)),
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["eta_minutes"] > 0
        client.post("/feedback", json={"request_id": body["request_id"], "actual_minutes": 60.0})
    text = client.get("/metrics").text
    assert "taximon_predictions_total 250.0" in text
    psi = {
        line.split('"')[1]: float(line.split()[-1])
        for line in text.splitlines()
        if line.startswith("taximon_feature_psi{")
    }
    assert psi["trip_distance"] > 1.0 > 0.1 > psi["hour"]  # hours are uniform in both

    report = monitor_logs.run(tmp_path, deployed / "current", mae_alert=1000)
    assert report["rows"] == 250 and report["labelled"] == 250
    assert any("trip_distance" in a for a in report["alerts"])
    assert not any(a.startswith("MAE") for a in report["alerts"])
