from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
V2_OUTPUT_DIR = ROOT_DIR / "outputs" / "v2"


def build_project_preview() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ASSETS_DIR / "project-preview.png"
    panels = [
        ("Training history", V2_OUTPUT_DIR / "training_history.png"),
        ("Precision-recall", V2_OUTPUT_DIR / "precision_recall_curve.png"),
        ("Calibration", V2_OUTPUT_DIR / "calibration_curve.png"),
        ("Sensor branch importance", V2_OUTPUT_DIR / "branch_importance.png"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#f6f7f2")
    fig.suptitle(
        "Smart Factory Predictive Maintenance - Model Evidence",
        fontsize=18,
        fontweight="bold",
        color="#17211a",
    )

    for ax, (title, image_path) in zip(axes.ravel(), panels):
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_title(title, fontsize=12, color="#17211a", pad=8)
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.4)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_api_contract() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ASSETS_DIR / "api-contract.png"
    endpoints = [
        ("GET", "/health", "service and artifact check"),
        ("GET", "/metadata", "model schema and settings"),
        ("POST", "/predict/fused", "score one complete machine reading"),
        ("POST", "/predict/events", "buffer sensor events into fused readings"),
        ("POST", "/stream/reset", "clear live machine buffers"),
        ("POST", "/simulate/run", "run a full smart-factory replay"),
        ("GET", "/examples/fused-reading", "return a valid sample payload"),
    ]

    fig, ax = plt.subplots(figsize=(13, 7), facecolor="#10201b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.91,
        "FastAPI service layer",
        fontsize=24,
        fontweight="bold",
        color="#f4f1e8",
    )
    ax.text(
        0.05,
        0.85,
        "The model is served as a small backend, not only a local training script.",
        fontsize=13,
        color="#c7d4c6",
    )

    y = 0.74
    for method, path, description in endpoints:
        method_color = "#6fd08c" if method == "GET" else "#f7b267"
        ax.text(0.07, y, method, fontsize=12, fontweight="bold", color=method_color)
        ax.text(0.18, y, path, fontsize=13, fontweight="bold", color="#f4f1e8")
        ax.text(0.52, y, description, fontsize=12, color="#c7d4c6")
        ax.plot([0.06, 0.94], [y - 0.035, y - 0.035], color="#28443b", linewidth=1)
        y -= 0.085

    ax.text(
        0.05,
        0.08,
        "Run: python -m src.v2_api  |  Docs: http://127.0.0.1:8000/docs",
        fontsize=12,
        color="#f4f1e8",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    preview = build_project_preview()
    api_contract = build_api_contract()
    print(f"Created {preview}")
    print(f"Created {api_contract}")


if __name__ == "__main__":
    main()
