from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import V2_DASHBOARD_PATH, V2_LIVE_PREDICTIONS_PATH, V2_SIMULATED_STREAM_PATH


def build_dashboard_report(
    stream_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: str | Path = V2_DASHBOARD_PATH,
) -> Path:
    if predictions_df.empty:
        raise ValueError("Predictions dataframe is empty. Run the live demo first.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latest_risk = (
        predictions_df.sort_values("timestamp")
        .groupby("machine_id", as_index=False)
        .tail(1)
        .sort_values("failure_probability", ascending=False)
    )
    focus_machine_ids = latest_risk["machine_id"].head(3).tolist()
    focus_machine_id = focus_machine_ids[0]

    merged_focus = stream_df.merge(
        predictions_df[
            [
                "machine_id",
                "timestamp",
                "failure_probability",
                "risk_band",
                "classification_flag",
            ]
        ],
        on=["machine_id", "timestamp"],
        how="left",
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Failure Probability by Machine",
            f"Sensor Trends for {focus_machine_id}",
            "Latest Maintenance Priority",
            "Risk vs Vibration Snapshot",
        ),
        vertical_spacing=0.14,
    )

    for machine_id in focus_machine_ids:
        machine_predictions = predictions_df[predictions_df["machine_id"] == machine_id]
        fig.add_trace(
            go.Scatter(
                x=machine_predictions["timestamp"],
                y=machine_predictions["failure_probability"],
                mode="lines",
                name=f"{machine_id} risk",
            ),
            row=1,
            col=1,
        )

    focus_stream = merged_focus[merged_focus["machine_id"] == focus_machine_id]
    fig.add_trace(
        go.Scatter(
            x=focus_stream["timestamp"],
            y=focus_stream["vibration_mm_s"],
            mode="lines",
            name="vibration_mm_s",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=focus_stream["timestamp"],
            y=focus_stream["current_a"],
            mode="lines",
            name="current_a",
        ),
        row=1,
        col=2,
    )
    failure_points = focus_stream[focus_stream["breakdown_event"] == 1]
    if not failure_points.empty:
        fig.add_trace(
            go.Scatter(
                x=failure_points["timestamp"],
                y=failure_points["vibration_mm_s"],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name="breakdown_event",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Bar(
            x=latest_risk["machine_id"],
            y=latest_risk["maintenance_priority"],
            name="priority_score",
            marker_color="#d55e00",
        ),
        row=2,
        col=1,
    )

    snapshot = merged_focus.dropna(subset=["failure_probability"]).copy()
    fig.add_trace(
        go.Scatter(
            x=snapshot["vibration_mm_s"],
            y=snapshot["failure_probability"],
            mode="markers",
            marker=dict(
                color=snapshot["classification_flag"].map({True: 1, False: 0}),
                colorscale=[[0, "#1b9e77"], [1, "#d95f02"]],
                showscale=False,
                size=7,
            ),
            text=snapshot["machine_id"],
            name="risk_points",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Smart Factory Neural Predictive Maintenance Dashboard",
        template="plotly_white",
        height=850,
    )
    fig.update_yaxes(title_text="Failure Probability", row=1, col=1)
    fig.update_yaxes(title_text="Sensor Value", row=1, col=2)
    fig.update_yaxes(title_text="Priority Score", row=2, col=1)
    fig.update_yaxes(title_text="Failure Probability", row=2, col=2)
    fig.update_xaxes(title_text="Timestamp", row=1, col=1)
    fig.update_xaxes(title_text="Timestamp", row=1, col=2)
    fig.update_xaxes(title_text="Machine", row=2, col=1)
    fig.update_xaxes(title_text="Vibration (mm/s)", row=2, col=2)

    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a smart factory dashboard report.")
    parser.add_argument(
        "--stream-csv",
        type=str,
        default=str(V2_SIMULATED_STREAM_PATH),
        help="CSV file containing the simulated fused sensor stream.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=str(V2_LIVE_PREDICTIONS_PATH),
        help="CSV file containing live predictions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(V2_DASHBOARD_PATH),
        help="Destination HTML file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stream_df = pd.read_csv(args.stream_csv)
    predictions_df = pd.read_csv(args.predictions_csv)
    output_path = build_dashboard_report(stream_df, predictions_df, args.output)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
