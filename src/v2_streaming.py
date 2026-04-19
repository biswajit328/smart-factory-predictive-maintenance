from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .config import (
    RANDOM_STATE,
    SENSOR_GROUP_MAP,
    SIMULATED_SENSOR_COLUMNS,
    V2_FREQ_MINUTES,
    V2_HORIZON_STEPS,
    V2_NUM_MACHINES,
    V2_NUM_STEPS,
)
from .preprocess import load_raw_data


@dataclass
class SensorEvent:
    machine_id: str
    machine_type: str
    timestamp: str
    step: int
    sensor_name: str
    sensor_group: str
    sensor_value: float
    breakdown_event: int
    failure_next_horizon: int


def build_type_profiles(raw_df: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    source_df = load_raw_data() if raw_df is None else raw_df.copy()

    grouped = source_df.groupby("Type")
    profiles = {}
    for machine_type, group in grouped:
        profiles[machine_type] = {
            "air_temp_mean": float(group["Air temperature [K]"].mean()),
            "air_temp_std": float(group["Air temperature [K]"].std(ddof=0)),
            "process_gap_mean": float(
                (group["Process temperature [K]"] - group["Air temperature [K]"]).mean()
            ),
            "rpm_mean": float(group["Rotational speed [rpm]"].mean()),
            "rpm_std": float(group["Rotational speed [rpm]"].std(ddof=0)),
            "torque_mean": float(group["Torque [Nm]"].mean()),
            "torque_std": float(group["Torque [Nm]"].std(ddof=0)),
            "wear_mean": float(group["Tool wear [min]"].mean()),
        }
    return profiles


def _simulate_machine_records(
    machine_id: str,
    machine_type: str,
    steps: int,
    horizon_steps: int,
    start_timestamp: pd.Timestamp,
    freq_minutes: int,
    profile: dict[str, float],
    rng: np.random.Generator,
    machine_index: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    wear = float(rng.uniform(0.0, max(8.0, profile["wear_mean"] * 0.35)))
    min_onset = max(horizon_steps + 2, steps // 3)
    max_onset = max(min_onset + 1, steps - max(horizon_steps + 6, 10))
    onset_step = int(rng.integers(min_onset, max_onset))
    max_duration = max(7, min(20, steps - onset_step - 3))
    pre_failure_duration = int(rng.integers(6, max_duration))
    failure_step = min(steps - 6, onset_step + pre_failure_duration)

    for step in range(steps):
        timestamp = start_timestamp + pd.Timedelta(minutes=freq_minutes * step)
        fault_progress = 0.0
        if onset_step <= step <= failure_step:
            fault_progress = (step - onset_step) / max(pre_failure_duration, 1)

        post_failure = step > failure_step
        if post_failure:
            wear *= 0.72

        load = 0.70 + 0.25 * np.sin(step / 12 + machine_index / 3) + rng.normal(0, 0.035)
        load = float(np.clip(load, 0.25, 1.25))
        wear_increment = 0.9 + 0.8 * load + 1.6 * fault_progress + rng.normal(0, 0.2)
        wear = max(0.0, wear + wear_increment)

        air_temp = (
            profile["air_temp_mean"]
            + 1.3 * np.sin(step / 18)
            + rng.normal(0, max(profile["air_temp_std"], 0.35) * 0.35)
        )
        process_temp = (
            air_temp
            + profile["process_gap_mean"]
            + 1.7 * load
            + 3.8 * fault_progress
            + rng.normal(0, 0.45)
        )
        rotational_speed = (
            profile["rpm_mean"]
            + 85 * np.sin(step / 10)
            - 110 * fault_progress
            + rng.normal(0, max(profile["rpm_std"], 12.0) * 0.12)
        )
        torque = (
            profile["torque_mean"]
            + 7.0 * load
            + 10.0 * fault_progress
            + rng.normal(0, max(profile["torque_std"], 1.0) * 0.15)
        )
        vibration = (
            1.1
            + 0.024 * torque
            + 0.0035 * wear
            + 2.6 * fault_progress
            + rng.normal(0, 0.06)
        )
        pressure = (
            5.2
            + 0.0014 * rotational_speed
            + 0.022 * torque
            + 0.75 * fault_progress
            + rng.normal(0, 0.07)
        )
        current = (
            8.5
            + (torque * max(rotational_speed, 200.0)) / 1800.0
            + 2.8 * fault_progress
            + rng.normal(0, 0.18)
        )
        acoustic = (
            54.0
            + 6.8 * vibration
            + 2.5 * fault_progress
            + rng.normal(0, 0.35)
        )
        humidity = 46.0 + 6.5 * np.sin((step + machine_index) / 20) + rng.normal(0, 1.0)
        breakdown_event = int(step == failure_step)

        rows.append(
            {
                "machine_id": machine_id,
                "machine_type": machine_type,
                "timestamp": timestamp.isoformat(),
                "step": step,
                "air_temp_k": round(float(air_temp), 3),
                "process_temp_k": round(float(process_temp), 3),
                "rotational_speed_rpm": round(float(rotational_speed), 3),
                "torque_nm": round(float(torque), 3),
                "tool_wear_min": round(float(wear), 3),
                "vibration_mm_s": round(float(vibration), 3),
                "pressure_bar": round(float(pressure), 3),
                "current_a": round(float(current), 3),
                "acoustic_db": round(float(acoustic), 3),
                "humidity_pct": round(float(humidity), 3),
                "fault_progress": round(float(fault_progress), 4),
                "breakdown_event": breakdown_event,
            }
        )

    return rows


def _label_failure_horizon(
    stream_df: pd.DataFrame,
    horizon_steps: int,
) -> pd.DataFrame:
    labeled = stream_df.copy()
    labeled["failure_next_horizon"] = 0

    for _, group in labeled.groupby("machine_id"):
        breakdown = group["breakdown_event"].to_numpy(dtype=int)
        labels = np.zeros(len(group), dtype=int)
        for index in range(len(group)):
            labels[index] = int(breakdown[index + 1 : index + 1 + horizon_steps].any())
        labeled.loc[group.index, "failure_next_horizon"] = labels

    return labeled


def simulate_factory_stream(
    num_machines: int = V2_NUM_MACHINES,
    steps: int = V2_NUM_STEPS,
    horizon_steps: int = V2_HORIZON_STEPS,
    freq_minutes: int = V2_FREQ_MINUTES,
    start_timestamp: str = "2026-01-01T08:00:00",
    seed: int = RANDOM_STATE,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    profiles = build_type_profiles(raw_df)
    machine_types = sorted(profiles.keys())
    rng = np.random.default_rng(seed)
    start_time = pd.Timestamp(start_timestamp)

    rows: list[dict[str, object]] = []
    for machine_index in range(num_machines):
        machine_type = machine_types[machine_index % len(machine_types)]
        machine_id = f"M{machine_index:03d}"
        rows.extend(
            _simulate_machine_records(
                machine_id=machine_id,
                machine_type=machine_type,
                steps=steps,
                horizon_steps=horizon_steps,
                start_timestamp=start_time,
                freq_minutes=freq_minutes,
                profile=profiles[machine_type],
                rng=rng,
                machine_index=machine_index,
            )
        )

    stream_df = pd.DataFrame(rows).sort_values(["timestamp", "machine_id"]).reset_index(drop=True)
    return _label_failure_horizon(stream_df, horizon_steps=horizon_steps)


def to_sensor_events(stream_df: pd.DataFrame) -> pd.DataFrame:
    event_df = stream_df.melt(
        id_vars=[
            "machine_id",
            "machine_type",
            "timestamp",
            "step",
            "breakdown_event",
            "failure_next_horizon",
        ],
        value_vars=SIMULATED_SENSOR_COLUMNS,
        var_name="sensor_name",
        value_name="sensor_value",
    )
    event_df["sensor_group"] = event_df["sensor_name"].map(SENSOR_GROUP_MAP)
    return event_df.sort_values(["timestamp", "machine_id", "sensor_name"]).reset_index(drop=True)


def iter_sensor_events(stream_df: pd.DataFrame) -> Iterator[SensorEvent]:
    for row in to_sensor_events(stream_df).itertuples(index=False):
        yield SensorEvent(
            machine_id=row.machine_id,
            machine_type=row.machine_type,
            timestamp=row.timestamp,
            step=int(row.step),
            sensor_name=row.sensor_name,
            sensor_group=row.sensor_group,
            sensor_value=float(row.sensor_value),
            breakdown_event=int(row.breakdown_event),
            failure_next_horizon=int(row.failure_next_horizon),
        )
