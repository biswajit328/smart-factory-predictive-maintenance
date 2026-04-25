from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
V2_OUTPUT_DIR = ROOT_DIR / "outputs" / "v2"
sys.path.insert(0, str(ROOT_DIR))


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
    from src.v2_api import app

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ASSETS_DIR / "api-contract.png"
    descriptions = {
        "/health": "service and artifact check",
        "/infrastructure": "Redis, Postgres, and MQTT reachability",
        "/metadata": "model schema and settings",
        "/predict/fused": "score one complete machine reading",
        "/predict/events": "buffer sensor events into fused readings",
        "/stream/reset": "clear live machine buffers",
        "/simulate/run": "run a full smart-factory replay",
        "/examples/fused-reading": "return a valid sample payload",
    }
    schema = app.openapi()
    endpoints = []
    for path, methods in schema["paths"].items():
        for method in methods:
            method_upper = method.upper()
            if method_upper in {"GET", "POST"} and path in descriptions:
                endpoints.append((method_upper, path, descriptions[path]))

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

    y = 0.76
    for method, path, description in endpoints:
        method_color = "#6fd08c" if method == "GET" else "#f7b267"
        ax.text(0.07, y, method, fontsize=12, fontweight="bold", color=method_color)
        ax.text(0.18, y, path, fontsize=13, fontweight="bold", color="#f4f1e8")
        ax.text(0.52, y, description, fontsize=12, color="#c7d4c6")
        ax.plot([0.06, 0.94], [y - 0.035, y - 0.035], color="#28443b", linewidth=1)
        y -= 0.075

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


def build_demo_gif() -> Path:
    output_path = ASSETS_DIR / "demo-walkthrough.gif"
    source_paths = [
        ASSETS_DIR / "project-preview.png",
        ASSETS_DIR / "api-contract.png",
    ]
    frames = []
    for source_path in source_paths:
        image = Image.open(source_path).convert("RGB")
        image.thumbnail((1280, 720))
        canvas = Image.new("RGB", (1280, 720), "#f6f7f2")
        x = (canvas.width - image.width) // 2
        y = (canvas.height - image.height) // 2
        canvas.paste(image, (x, y))
        frames.append(canvas)

    title_frame = Image.new("RGB", (1280, 720), "#10201b")
    draw = ImageDraw.Draw(title_frame)
    font = ImageFont.load_default()
    draw.text((90, 300), "Smart Factory Predictive Maintenance", fill="#f4f1e8", font=font)
    draw.text((90, 330), "Model evidence + API service preview", fill="#c7d4c6", font=font)
    frames.insert(0, title_frame)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[900, 1300, 1300],
        loop=0,
    )
    return output_path


def main() -> None:
    preview = build_project_preview()
    api_contract = build_api_contract()
    demo_gif = build_demo_gif()
    print(f"Created {preview}")
    print(f"Created {api_contract}")
    print(f"Created {demo_gif}")


if __name__ == "__main__":
    main()
