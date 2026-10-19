from src.qm.features import (
    derive_edit_fields,
    journey_to_features,
    MODEL_FEATURES,
    age_band,
    age_band_code,
)


def _q(age, exp, acc, mileage, premium, t):
    return {
        "driver_age": age, "driver_experience": exp, "previous_accidents": acc,
        "annual_mileage": mileage, "car_age": 5, "premium": premium, "quote_time": t,
    }


# --- edit detection ---

def test_mileage_drop_is_material_and_suspicious():
    prev = _q(30, 10, 0, 18.0, 900, "2026-03-01T09:00:00")
    curr = _q(30, 10, 0, 14.0, 800, "2026-03-01T09:10:00")
    changed, material, suspicious = derive_edit_fields(prev, curr)
    assert "annual_mileage" in changed
    assert "annual_mileage" in material      # 4.0 >= 1.5
    assert "annual_mileage" in suspicious     # decrease lowers premium


def test_small_mileage_change_is_not_material():
    prev = _q(30, 10, 0, 10.0, 900, "2026-03-01T09:00:00")
    curr = _q(30, 10, 0, 10.5, 900, "2026-03-01T09:10:00")  # 0.5 < 1.5
    changed, material, _ = derive_edit_fields(prev, curr)
    assert "annual_mileage" in changed
    assert "annual_mileage" not in material


def test_accident_removal_is_suspicious():
    prev = _q(30, 10, 2, 10.0, 900, "2026-03-01T09:00:00")
    curr = _q(30, 10, 1, 10.0, 800, "2026-03-01T09:10:00")
    _, material, suspicious = derive_edit_fields(prev, curr)
    assert "previous_accidents" in material     # any accident change is material
    assert "previous_accidents" in suspicious


def test_age_band_crossing_is_material_and_suspicious():
    prev = _q(24, 5, 0, 10.0, 900, "2026-03-01T09:00:00")
    curr = _q(25, 5, 0, 10.0, 800, "2026-03-01T09:10:00")
    _, material, suspicious = derive_edit_fields(prev, curr)
    assert age_band(24) != age_band(25)
    assert age_band_code(25) < age_band_code(24)
    assert "driver_age" in material
    assert "driver_age" in suspicious


def test_experience_increase_is_suspicious():
    prev = _q(30, 4, 0, 10.0, 900, "2026-03-01T09:00:00")
    curr = _q(30, 6, 0, 10.0, 820, "2026-03-01T09:10:00")
    _, _, suspicious = derive_edit_fields(prev, curr)
    assert "driver_experience" in suspicious


# --- journey_to_features ---

def test_produces_all_31_features():
    j = [_q(24, 2, 2, 18.0, 1300, "2026-03-01T09:00:00"),
         _q(25, 4, 0, 10.0, 820, "2026-03-01T09:15:00")]
    out = journey_to_features(j)
    assert set(out["feature_vector"].keys()) == set(MODEL_FEATURES)
    assert len(out["feature_vector"]) == 31


def test_single_quote_journey_has_zero_edits():
    out = journey_to_features([_q(40, 20, 0, 9.0, 560, "2026-03-01T09:00:00")])
    ev = out["evidence_fields"]
    assert ev["n_quotes"] == 1
    assert ev["material_edit_count"] == 0
    assert ev["suspicious_step_ratio"] == 0.0


def test_manipulated_journey_accumulates_signals():
    j = [_q(24, 2, 2, 18.0, 1300, "2026-03-01T09:00:00"),
         _q(24, 3, 1, 14.0, 1100, "2026-03-01T09:12:00"),
         _q(25, 4, 0, 8.0, 760, "2026-03-01T09:21:00")]
    ev = journey_to_features(j)["evidence_fields"]
    assert ev["material_edit_count"] > 0
    assert ev["mileage_decrease_count"] >= 2
    assert ev["accident_decrease_count"] >= 1
    assert ev["first_to_lowest_premium_drop_pct"] > 0


def test_empty_journey_raises():
    try:
        journey_to_features([])
        assert False, "expected ValueError"
    except ValueError:
        pass
