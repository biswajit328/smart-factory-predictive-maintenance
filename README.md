# Smart Factory Predictive Maintenance

A two-layer predictive maintenance project:

- **v1 (baseline):** leakage-free tabular ML pipeline for failure risk + anomaly detection.
- **v2 (advanced):** neural, multi-sensor smart-factory simulation with streaming inference, API, and dashboard.

## Project Goal

Predict equipment failure before breakdown using machine telemetry, so maintenance can be planned earlier and downtime reduced.

## Repository Structure

```text
project_root/
├── data/
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── v2_streaming.py
│   ├── v2_neural.py
│   ├── v2_train.py
│   ├── v2_inference.py
│   ├── v2_dashboard.py
│   └── v2_api.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## v1: Baseline Pipeline

Train and save artifacts:

```bash
python -m src.train
```

Run demo inference:

```bash
python -m src.inference --demo
```

v1 outputs are saved under `outputs/` (model bundle, metrics, reports, and plots).

## v2: Neural Smart-Factory Pipeline

Train the v2 model:

```bash
python -m src.v2_train
```

Run live simulation inference:

```bash
python -m src.v2_inference --live-demo
```

Generate dashboard from saved predictions:

```bash
python -m src.v2_dashboard
```

v2 outputs are saved under `outputs/v2/`.

## Run API (v2)

Start FastAPI service:

```bash
python -m src.v2_api
```

Open docs:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict/fused`
- `POST /predict/events`
- `POST /stream/reset`
- `POST /simulate/run`
- `GET /examples/fused-reading`

## Testing

Run test suite:

```bash
python -m unittest discover -s tests -v
```

If API tests fail with `starlette.testclient requires httpx`, install:

```bash
pip install httpx
```

## Notes

- v2 performance metrics come from a simulator-based environment, not a real industrial live stream.
- This project is useful for showcasing both ML fundamentals (v1) and ML systems thinking (v2).
