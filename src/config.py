import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = _env_path("PM_OUTPUT_DIR", ROOT_DIR / "outputs")
V2_OUTPUT_DIR = OUTPUT_DIR / "v2"


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)

RAW_DATA_PATH = _env_path("PM_RAW_DATA_PATH", DATA_DIR / "predictive_maintenance.csv")
MODEL_BUNDLE_PATH = OUTPUT_DIR / "model_bundle.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
PRECISION_RECALL_CURVE_PATH = OUTPUT_DIR / "precision_recall_curve.png"
ROC_CURVE_PATH = OUTPUT_DIR / "roc_curve.png"
CALIBRATION_CURVE_PATH = OUTPUT_DIR / "calibration_curve.png"
THRESHOLD_ANALYSIS_PATH = OUTPUT_DIR / "threshold_analysis.csv"
PROBABILITY_DISTRIBUTION_PATH = OUTPUT_DIR / "probability_distribution.png"
FEATURE_IMPORTANCE_PLOT_PATH = OUTPUT_DIR / "feature_importance.png"
V2_MODEL_PATH = V2_OUTPUT_DIR / "temporal_fusion_model.keras"
V2_SCALERS_PATH = V2_OUTPUT_DIR / "branch_scalers.joblib"
V2_METADATA_PATH = V2_OUTPUT_DIR / "neural_metadata.json"
V2_METRICS_PATH = V2_OUTPUT_DIR / "neural_metrics.json"
V2_SIMULATED_STREAM_PATH = V2_OUTPUT_DIR / "simulated_stream.csv"
V2_SENSOR_EVENTS_PATH = V2_OUTPUT_DIR / "sensor_events.csv"
V2_TEST_PREDICTIONS_PATH = V2_OUTPUT_DIR / "test_predictions.csv"
V2_LIVE_PREDICTIONS_PATH = V2_OUTPUT_DIR / "live_predictions.csv"
V2_TRAINING_HISTORY_PATH = V2_OUTPUT_DIR / "training_history.png"
V2_ROC_CURVE_PATH = V2_OUTPUT_DIR / "roc_curve.png"
V2_PR_CURVE_PATH = V2_OUTPUT_DIR / "precision_recall_curve.png"
V2_CALIBRATION_CURVE_PATH = V2_OUTPUT_DIR / "calibration_curve.png"
V2_THRESHOLD_ANALYSIS_PATH = V2_OUTPUT_DIR / "threshold_analysis.csv"
V2_BRANCH_IMPORTANCE_PATH = V2_OUTPUT_DIR / "branch_importance.csv"
V2_BRANCH_IMPORTANCE_PLOT_PATH = V2_OUTPUT_DIR / "branch_importance.png"
V2_DASHBOARD_PATH = V2_OUTPUT_DIR / "smart_factory_dashboard.html"

TARGET_COL = "Machine failure"
CATEGORICAL_COLS = ["Type"]
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ID_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RAW_INPUT_COLUMNS = CATEGORICAL_COLS + SENSOR_COLS

RANDOM_STATE = _env_int("PM_RANDOM_STATE", 42)
TEST_SIZE = _env_float("PM_TEST_SIZE", 0.20)
VAL_SIZE = _env_float("PM_VAL_SIZE", 0.10)
THRESHOLD_PRECISION_FLOOR = _env_float("PM_THRESHOLD_PRECISION_FLOOR", 0.60)
THRESHOLD_BETA = _env_float("PM_THRESHOLD_BETA", 2.0)
ANOMALY_QUANTILE = _env_float("PM_ANOMALY_QUANTILE", 0.95)

API_HOST = os.getenv("PM_API_HOST", "127.0.0.1")
API_PORT = _env_int("PM_API_PORT", _env_int("PORT", 8000))
DASHBOARD_HOST = os.getenv("PM_DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = _env_int("PM_DASHBOARD_PORT", _env_int("PORT", 8501))
LOG_LEVEL = os.getenv("PM_LOG_LEVEL", "INFO").upper()
DATABASE_URL = os.getenv(
    "PM_DATABASE_URL",
    "postgresql://maintenance:maintenance@postgres:5432/maintenance",
)
REDIS_URL = os.getenv("PM_REDIS_URL", "redis://redis:6379/0")
MQTT_BROKER_HOST = os.getenv("PM_MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = _env_int("PM_MQTT_BROKER_PORT", 1883)
MQTT_SENSOR_TOPIC = os.getenv("PM_MQTT_SENSOR_TOPIC", "factory/sensors")
API_PREDICT_EVENTS_URL = os.getenv(
    "PM_API_PREDICT_EVENTS_URL",
    f"http://{API_HOST}:{API_PORT}/predict/events",
)

SIMULATED_SENSOR_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mm_s",
    "pressure_bar",
    "current_a",
    "acoustic_db",
    "humidity_pct",
]
TYPE_FEATURE_COLUMNS = ["type_H", "type_L", "type_M"]

THERMAL_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "humidity_pct",
]
MECHANICAL_FEATURES = [
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mm_s",
    *TYPE_FEATURE_COLUMNS,
]
ELECTRICAL_FEATURES = [
    "pressure_bar",
    "current_a",
    "acoustic_db",
]
FEATURE_GROUPS = {
    "thermal": THERMAL_FEATURES,
    "mechanical": MECHANICAL_FEATURES,
    "electrical": ELECTRICAL_FEATURES,
}
SENSOR_GROUP_MAP = {
    "air_temp_k": "thermal",
    "process_temp_k": "thermal",
    "humidity_pct": "thermal",
    "rotational_speed_rpm": "mechanical",
    "torque_nm": "mechanical",
    "tool_wear_min": "mechanical",
    "vibration_mm_s": "mechanical",
    "pressure_bar": "electrical",
    "current_a": "electrical",
    "acoustic_db": "electrical",
}

V2_WINDOW_SIZE = _env_int("PM_V2_WINDOW_SIZE", 18)
V2_HORIZON_STEPS = _env_int("PM_V2_HORIZON_STEPS", 8)
V2_FREQ_MINUTES = _env_int("PM_V2_FREQ_MINUTES", 5)
V2_NUM_MACHINES = _env_int("PM_V2_NUM_MACHINES", 10)
V2_NUM_STEPS = _env_int("PM_V2_NUM_STEPS", 120)
V2_EPOCHS = _env_int("PM_V2_EPOCHS", 5)
V2_BATCH_SIZE = _env_int("PM_V2_BATCH_SIZE", 16)
V2_THRESHOLD_PRECISION_FLOOR = _env_float("PM_V2_THRESHOLD_PRECISION_FLOOR", 0.25)
V2_THRESHOLD_BETA = _env_float("PM_V2_THRESHOLD_BETA", 3.0)
