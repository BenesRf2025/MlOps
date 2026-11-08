from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_sol_sec_declenche_irrigation():
    r = client.post("/predict", json={
        "culture": "tomate", "soil_moisture_pct": 20, "air_temp_c": 33,
        "air_humidity_pct": 30, "solar_radiation_wm2": 800,
        "rainfall_mm": 0, "hour": 13,
    })
    assert r.status_code == 200
    assert r.json()["irrigation_needed"] is True


def test_sol_humide_pas_d_irrigation():
    r = client.post("/predict", json={
        "culture": "mais", "soil_moisture_pct": 65, "air_temp_c": 22,
        "air_humidity_pct": 70, "solar_radiation_wm2": 0,
        "rainfall_mm": 2, "hour": 3,
    })
    assert r.json()["irrigation_needed"] is False
