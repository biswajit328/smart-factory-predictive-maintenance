from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .anomaly import score_anomalies
from .config import MODEL_BUNDLE_PATH, OUTPUT_DIR, RAW_DATA_PATH, RAW_INPUT_COLUMNS
from .features import prepare_model_frame


class PredictiveMaintenanceService:
    def __init__(self, bundle_path: str | Path = MODEL_BUNDLE_PATH):
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. Run 'python -m src.train' first."
            )

        self.bundle = joblib.load(bundle_path)
        self.pipeline = self.bundle["pipeline"]
        self.preprocessor = self.pipeline.named_steps["preprocess"]
        self.anomaly_model = self.bundle["anomaly_model"]
        self.probability_threshold = float(self.bundle["probability_threshold"])
        self.anomaly_threshold = float(self.bundle["anomaly_threshold"])

    def _build_recommendation(self, probability: float, anomaly_flag: bool) -> tuple[str, str]:
        if anomaly_flag or probability >= self.probability_threshold:
            return "high", "Inspect this machine soon and plan maintenance."
        if probability >= self.probability_threshold * 0.5:
            return "medium", "Monitor this machine closely and recheck the next readings."
        return "low", "Keep the normal maintenance schedule."

    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        feature_frame = prepare_model_frame(raw_df)
        probabilities = self.pipeline.predict_proba(feature_frame)[:, 1]
        transformed_matrix = self.preprocessor.transform(feature_frame)
        anomaly_scores = score_anomalies(self.anomaly_model, transformed_matrix)

        result = raw_df.copy()
        result["failure_probability"] = probabilities
        result["classification_flag"] = result["failure_probability"] >= self.probability_threshold
        result["anomaly_score"] = anomaly_scores
        result["anomaly_flag"] = result["anomaly_score"] >= self.anomaly_threshold

        risk_bands = []
        recommendations = []
        for probability, anomaly_flag in zip(
            result["failure_probability"],
            result["anomaly_flag"],
        ):
            risk_band, recommendation = self._build_recommendation(
                float(probability),
                bool(anomaly_flag),
            )
            risk_bands.append(risk_band)
            recommendations.append(recommendation)

        result["risk_band"] = risk_bands
        result["recommended_action"] = recommendations

        for column in ["failure_probability", "anomaly_score"]:
            result[column] = result[column].round(4)

        return result

    def predict_one(self, raw_record: dict) -> dict:
        input_frame = pd.DataFrame([raw_record])
        scored = self.score_dataframe(input_frame).iloc[0]

        return {
            "failure_probability": float(scored["failure_probability"]),
            "decision_threshold": round(self.probability_threshold, 4),
            "classification_flag": bool(scored["classification_flag"]),
            "anomaly_score": float(scored["anomaly_score"]),
            "anomaly_threshold": round(self.anomaly_threshold, 4),
            "anomaly_flag": bool(scored["anomaly_flag"]),
            "risk_band": str(scored["risk_band"]),
            "recommended_action": str(scored["recommended_action"]),
        }


def run_demo(bundle_path: str | Path = MODEL_BUNDLE_PATH) -> None:
    service = PredictiveMaintenanceService(bundle_path=bundle_path)

    demo_rows = [
        {
            "Type": "L",
            "Air temperature [K]": 298.0,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        },
        {
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 312.0,
            "Rotational speed [rpm]": 1350.0,
            "Torque [Nm]": 60.0,
            "Tool wear [min]": 180.0,
        },
        {
            "Type": "H",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 317.0,
            "Rotational speed [rpm]": 1220.0,
            "Torque [Nm]": 72.0,
            "Tool wear [min]": 230.0,
        },
    ]

    for index, row in enumerate(demo_rows, start=1):
        prediction = service.predict_one(row)
        print(
            f"Example {index}: risk={prediction['risk_band']} "
            f"prob={prediction['failure_probability']:.4f} "
            f"anomaly={prediction['anomaly_flag']} "
            f"action={prediction['recommended_action']}"
        )


def score_csv_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    bundle_path: str | Path = MODEL_BUNDLE_PATH,
) -> Path:
    service = PredictiveMaintenanceService(bundle_path=bundle_path)
    raw_df = pd.read_csv(input_path)
    scored_df = service.score_dataframe(raw_df)

    destination = Path(output_path) if output_path else OUTPUT_DIR / "scored_predictions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(destination, index=False)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for predictive maintenance.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Score three example machine readings.",
    )
    parser.add_argument(
        "--score-csv",
        type=str,
        default=None,
        help="Path to a CSV file with raw machine readings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for the scored CSV output.",
    )
    parser.add_argument(
        "--bundle-path",
        type=str,
        default=str(MODEL_BUNDLE_PATH),
        help="Path to the saved model bundle.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.demo:
        run_demo(bundle_path=args.bundle_path)
        return

    if args.score_csv:
        output_path = score_csv_file(
            input_path=args.score_csv,
            output_path=args.output,
            bundle_path=args.bundle_path,
        )
        print(f"Scored file written to {output_path}")
        return

    if Path(RAW_DATA_PATH).exists():
        sample_df = pd.read_csv(RAW_DATA_PATH).head(5)[RAW_INPUT_COLUMNS]
        scored_df = PredictiveMaintenanceService(args.bundle_path).score_dataframe(sample_df)
        print(scored_df.to_string(index=False))


if __name__ == "__main__":
    main()
