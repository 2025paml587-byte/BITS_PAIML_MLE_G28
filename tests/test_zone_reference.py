from src.data.zone_reference import load_high_traffic_zones, save_high_traffic_zones


def test_high_traffic_zones_roundtrip_through_json(tmp_path):
    path = tmp_path / "high_traffic_zones.json"
    zones = {"40.75_-73.98", "40.76_-73.97"}

    save_high_traffic_zones(zones, path)
    loaded = load_high_traffic_zones(path)

    assert loaded == zones


def test_load_high_traffic_zones_returns_empty_set_when_missing(tmp_path):
    assert load_high_traffic_zones(tmp_path / "does_not_exist.json") == set()
