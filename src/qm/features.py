"""Turn a raw quote journey into the 31 model features and case evidence.

The journey-level feature maths (`engineer_customer_features`) and the behavioural
helpers are lifted verbatim from the V3 notebook so the served path and the trained
model see identically-computed features. The only addition is `derive_edit_fields`,
which reconstructs the per-step edit flags (changed / material / premium-lowering)
from consecutive submitted quotes, using the same thresholds the simulator applied.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

__all__ = ["MODEL_FEATURES", "journey_to_features", "derive_edit_fields"]

# The 31 features the model expects, in training order (notebook cell 12).
MODEL_FEATURES = [
    "n_quotes",
    "premium_drop_pct",
    "first_to_lowest_premium_drop_pct",
    "cumulative_absolute_premium_change",
    "cumulative_premium_reduction",
    "largest_single_drop_pct",
    "mileage_decrease_count",
    "accident_decrease_count",
    "experience_increase_count",
    "age_edit_count",
    "age_band_cross_count",
    "consistency_violation_count",
    "new_consistency_violation_events",
    "premium_lowering_edit_count",
    "premium_lowering_edit_ratio",
    "material_edit_count",
    "material_edit_step_count",
    "material_edit_ratio",
    "suspicious_field_edit_count",
    "suspicious_edit_step_count",
    "suspicious_step_ratio",
    "avg_suspicious_fields_per_edit_step",
    "unique_fields_changed",
    "unique_material_fields_changed",
    "final_premium_is_lowest",
    "monotonic_premium_drop",
    "premium_recovery_count",
    "avg_minutes_between_quotes",
    "min_minutes_between_quotes",
    "max_minutes_between_quotes",
    "quote_velocity_score",
]


# --- behavioural helpers (verbatim from the notebook) ---

def age_band(age: float) -> str:
    if age < 25:
        return "17-24_high"
    if age < 40:
        return "25-39_medium"
    if age < 65:
        return "40-64_lower"
    return "65+_moderate"


def age_band_code(age: float) -> int:
    if age < 25:
        return 3
    if age < 40:
        return 2
    if age < 65:
        return 1
    return 2


def years_since_legal_driving_age(age: float) -> int:
    return max(int(round(age - 17)), 0)


def consistency_margin(age: float, experience: float) -> float:
    return years_since_legal_driving_age(age) - experience


def has_consistency_violation(age: float, experience: float) -> int:
    return int(experience > years_since_legal_driving_age(age))


# --- edit detection (same thresholds the simulator applied) ---

def derive_edit_fields(prev: Dict[str, Any], curr: Dict[str, Any]):
    """Compare two consecutive quotes and return (changed, material, suspicious).

    Mirrors the edit classification inside the notebook's journey simulator:
    a premium-lowering ('suspicious') edit is mileage down, accidents down,
    experience up, or an age edit that drops the rating band.
    """
    changed: List[str] = []
    material: List[str] = []
    suspicious: List[str] = []

    # annual mileage
    if curr["annual_mileage"] != prev["annual_mileage"]:
        changed.append("annual_mileage")
        if abs(curr["annual_mileage"] - prev["annual_mileage"]) >= 1.5:
            material.append("annual_mileage")
        if curr["annual_mileage"] < prev["annual_mileage"]:
            suspicious.append("annual_mileage")

    # previous accidents (any change is material)
    if curr["previous_accidents"] != prev["previous_accidents"]:
        changed.append("previous_accidents")
        material.append("previous_accidents")
        if curr["previous_accidents"] < prev["previous_accidents"]:
            suspicious.append("previous_accidents")

    # driving experience
    if curr["driver_experience"] != prev["driver_experience"]:
        changed.append("driver_experience")
        if abs(curr["driver_experience"] - prev["driver_experience"]) >= 1:
            material.append("driver_experience")
        if curr["driver_experience"] > prev["driver_experience"]:
            suspicious.append("driver_experience")

    # driver age (material only if the rating band changes)
    if curr["driver_age"] != prev["driver_age"]:
        changed.append("driver_age")
        if age_band(curr["driver_age"]) != age_band(prev["driver_age"]):
            material.append("driver_age")
        if age_band_code(curr["driver_age"]) < age_band_code(prev["driver_age"]):
            suspicious.append("driver_age")

    return changed, material, suspicious


# --- feature engineering (verbatim from the notebook) ---

def engineer_customer_features(quotes_df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []

    for customer_id, grp in quotes_df.groupby("customer_id"):
        grp = grp.sort_values("quote_number").reset_index(drop=True)

        n_quotes = len(grp)
        first_premium = grp.loc[0, "premium"]
        last_premium = grp.loc[n_quotes - 1, "premium"]
        lowest_premium = grp["premium"].min()
        max_premium = grp["premium"].max()

        premium_drop_pct = ((first_premium - last_premium) / first_premium) * 100
        first_to_lowest_premium_drop_pct = ((first_premium - lowest_premium) / first_premium) * 100
        cumulative_absolute_premium_change = grp["premium"].diff().abs().fillna(0).sum()
        cumulative_premium_reduction = grp["premium"].diff().fillna(0).clip(upper=0).abs().sum()
        largest_single_drop = abs(grp["premium"].diff().fillna(0).clip(upper=0).min())
        largest_single_drop_pct = (largest_single_drop / first_premium) * 100 if first_premium else 0

        mileage_changes = grp["annual_mileage"].diff().fillna(0)
        accident_changes = grp["previous_accidents"].diff().fillna(0)
        exp_changes = grp["driver_experience"].diff().fillna(0)
        age_changes = grp["driver_age"].diff().fillna(0)
        premium_changes = grp["premium"].diff().fillna(0)
        consistency_changes = grp["consistency_violation"].diff().fillna(0)

        mileage_decrease_count = int((mileage_changes < 0).sum())
        accident_decrease_count = int((accident_changes < 0).sum())
        experience_increase_count = int((exp_changes > 0).sum())
        age_edit_count = int((age_changes != 0).sum())
        age_band_cross_count = int((grp["age_band"].astype(str) != grp["age_band"].astype(str).shift(1)).sum() - 1)
        consistency_violation_count = int(grp["consistency_violation"].sum())
        new_consistency_violation_events = int((consistency_changes > 0).sum())

        premium_lowering_edit_count = int((premium_changes < 0).sum())
        premium_lowering_edit_ratio = premium_lowering_edit_count / (n_quotes - 1) if n_quotes > 1 else 0
        final_premium_is_lowest = int(last_premium == lowest_premium)
        monotonic_premium_drop = int(grp["premium"].is_monotonic_decreasing)
        premium_recovery_count = int(((premium_changes.shift(-1) < 0) & (premium_changes > 0)).sum())

        changed_fields_lists = grp["changed_fields"].tolist()
        material_fields_lists = grp["material_fields"].tolist()
        suspicious_field_lists = grp["suspicious_premium_lowering_fields"].tolist()

        total_edit_steps = sum(len(x) > 0 for x in changed_fields_lists[1:])

        material_edit_count = sum(len(x) for x in material_fields_lists[1:])
        suspicious_field_edit_count = sum(len(x) for x in suspicious_field_lists[1:])

        material_edit_step_count = sum(len(x) > 0 for x in material_fields_lists[1:])
        suspicious_edit_step_count = sum(len(x) > 0 for x in suspicious_field_lists[1:])

        unique_fields_changed = len(set(item for sublist in changed_fields_lists for item in sublist))
        unique_material_fields_changed = len(set(item for sublist in material_fields_lists for item in sublist))

        edit_step_count = max(n_quotes - 1, 1)
        material_edit_ratio = material_edit_step_count / edit_step_count
        suspicious_step_ratio = suspicious_edit_step_count / edit_step_count
        avg_suspicious_fields_per_edit_step = suspicious_field_edit_count / edit_step_count

        time_diffs = grp["quote_time"].diff().dropna()
        avg_minutes_between_quotes = time_diffs.dt.total_seconds().mean() / 60 if len(time_diffs) > 0 else np.nan
        min_minutes_between_quotes = time_diffs.dt.total_seconds().min() / 60 if len(time_diffs) > 0 else np.nan
        max_minutes_between_quotes = time_diffs.dt.total_seconds().max() / 60 if len(time_diffs) > 0 else np.nan
        quote_velocity_score = 1 / (avg_minutes_between_quotes + 1) if pd.notnull(avg_minutes_between_quotes) else 0

        feature_rows.append({
            "customer_id": customer_id,
            "n_quotes": n_quotes,
            "premium_drop_pct": premium_drop_pct,
            "first_to_lowest_premium_drop_pct": first_to_lowest_premium_drop_pct,
            "cumulative_absolute_premium_change": cumulative_absolute_premium_change,
            "cumulative_premium_reduction": cumulative_premium_reduction,
            "largest_single_drop_pct": largest_single_drop_pct,
            "mileage_decrease_count": mileage_decrease_count,
            "accident_decrease_count": accident_decrease_count,
            "experience_increase_count": experience_increase_count,
            "age_edit_count": age_edit_count,
            "age_band_cross_count": max(age_band_cross_count, 0),
            "consistency_violation_count": consistency_violation_count,
            "new_consistency_violation_events": new_consistency_violation_events,
            "premium_lowering_edit_count": premium_lowering_edit_count,
            "premium_lowering_edit_ratio": premium_lowering_edit_ratio,
            "total_edit_steps": total_edit_steps,
            "material_edit_count": material_edit_count,
            "material_edit_step_count": material_edit_step_count,
            "material_edit_ratio": material_edit_ratio,
            "suspicious_field_edit_count": suspicious_field_edit_count,
            "suspicious_edit_step_count": suspicious_edit_step_count,
            "suspicious_step_ratio": suspicious_step_ratio,
            "avg_suspicious_fields_per_edit_step": avg_suspicious_fields_per_edit_step,
            "unique_fields_changed": unique_fields_changed,
            "unique_material_fields_changed": unique_material_fields_changed,
            "final_premium_is_lowest": final_premium_is_lowest,
            "monotonic_premium_drop": monotonic_premium_drop,
            "premium_recovery_count": premium_recovery_count,
            "avg_minutes_between_quotes": avg_minutes_between_quotes if pd.notnull(avg_minutes_between_quotes) else 0,
            "min_minutes_between_quotes": min_minutes_between_quotes if pd.notnull(min_minutes_between_quotes) else 0,
            "max_minutes_between_quotes": max_minutes_between_quotes if pd.notnull(max_minutes_between_quotes) else 0,
            "quote_velocity_score": quote_velocity_score,
        })

    return pd.DataFrame(feature_rows)


# --- public entry point ---

def _build_quote_frame(quotes: List[Dict[str, Any]], customer_id: int) -> pd.DataFrame:
    """Expand a list of submitted quote steps into the per-quote frame the feature
    engineering expects, deriving the edit-flag lists between consecutive steps."""
    rows = []
    for idx, q in enumerate(quotes):
        if idx == 0:
            changed, material, suspicious = [], [], []
        else:
            changed, material, suspicious = derive_edit_fields(quotes[idx - 1], q)
        rows.append({
            "customer_id": customer_id,
            "quote_number": idx + 1,
            "quote_time": pd.to_datetime(q["quote_time"]),
            "driver_age": q["driver_age"],
            "driver_experience": q["driver_experience"],
            "previous_accidents": q["previous_accidents"],
            "annual_mileage": q["annual_mileage"],
            "premium": q["premium"],
            "age_band": age_band(q["driver_age"]),
            "consistency_violation": has_consistency_violation(q["driver_age"], q["driver_experience"]),
            "changed_fields": changed,
            "material_fields": material,
            "suspicious_premium_lowering_fields": suspicious,
        })
    return pd.DataFrame(rows)


def journey_to_features(quotes: List[Dict[str, Any]], customer_id: int = 0) -> Dict[str, Any]:
    """Convert a submitted journey into model features, plus the pieces /explain needs.

    Returns:
        feature_vector: ordered dict of the 31 model features (model input)
        evidence_fields: the subset of counts CaseEvidence needs
        quote_sequence: per-step records for the evidence payload
    """
    if not quotes:
        raise ValueError("journey must contain at least one quote")

    frame = _build_quote_frame(quotes, customer_id)
    row = engineer_customer_features(frame).iloc[0].to_dict()

    feature_vector = {name: row[name] for name in MODEL_FEATURES}

    evidence_fields = {
        "n_quotes": int(row["n_quotes"]),
        "first_to_lowest_premium_drop_pct": round(float(row["first_to_lowest_premium_drop_pct"]), 2),
        "largest_single_drop_pct": round(float(row["largest_single_drop_pct"]), 2),
        "material_edit_count": int(row["material_edit_count"]),
        "suspicious_field_edit_count": int(row["suspicious_field_edit_count"]),
        "suspicious_step_ratio": round(float(row["suspicious_step_ratio"]), 2),
        "mileage_decrease_count": int(row["mileage_decrease_count"]),
        "accident_decrease_count": int(row["accident_decrease_count"]),
        "experience_increase_count": int(row["experience_increase_count"]),
        "age_band_cross_count": int(row["age_band_cross_count"]),
        "consistency_violation_count": int(row["consistency_violation_count"]),
    }

    quote_sequence = []
    for idx, q in enumerate(quotes):
        if idx == 0:
            changed, suspicious = [], []
        else:
            changed, _, suspicious = derive_edit_fields(quotes[idx - 1], q)
        quote_sequence.append({
            "quote_number": idx + 1,
            "driver_age": q["driver_age"],
            "driver_experience": q["driver_experience"],
            "previous_accidents": q["previous_accidents"],
            "annual_mileage": q["annual_mileage"],
            "premium": q["premium"],
            "changed_fields": changed,
            "suspicious_premium_lowering_fields": suspicious,
        })

    return {
        "feature_vector": feature_vector,
        "evidence_fields": evidence_fields,
        "quote_sequence": quote_sequence,
    }
