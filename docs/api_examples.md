# API Examples

Start the API after training the v2 model:

```bash
python -m src.v2_train
python -m src.v2_api
```

The interactive docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Infrastructure Check

```bash
curl http://127.0.0.1:8000/infrastructure
```

## Example Fused Reading

```bash
curl http://127.0.0.1:8000/examples/fused-reading
```

## Predict From A Full Reading

The model needs a rolling window per machine. The first few calls return `warming_up`; after enough readings for the same machine, the API returns a prediction.

```bash
curl -X POST http://127.0.0.1:8000/predict/fused \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M000",
    "machine_type": "H",
    "timestamp": "2026-01-01T08:00:00",
    "step": 0,
    "air_temp_k": 299.8,
    "process_temp_k": 311.2,
    "rotational_speed_rpm": 1405.0,
    "torque_nm": 49.2,
    "tool_wear_min": 22.0,
    "vibration_mm_s": 2.1,
    "pressure_bar": 8.7,
    "current_a": 44.8,
    "acoustic_db": 68.2,
    "humidity_pct": 47.5,
    "breakdown_event": 0,
    "failure_next_horizon": 0
  }'
```

## Predict From Sensor Events

This endpoint is closer to the smart-factory idea. It accepts individual sensor events, buffers them, and creates one fused reading when all required sensors arrive for a timestamp.

```bash
curl -X POST http://127.0.0.1:8000/predict/events \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "machine_id": "M000",
        "machine_type": "H",
        "timestamp": "2026-01-01T08:00:00",
        "step": 0,
        "sensor_name": "air_temp_k",
        "sensor_value": 299.8,
        "breakdown_event": 0,
        "failure_next_horizon": 0
      }
    ]
  }'
```

## Reset Stream State

```bash
curl -X POST http://127.0.0.1:8000/stream/reset
```

## Run A Full Simulation From The API

```bash
curl -X POST http://127.0.0.1:8000/simulate/run \
  -H "Content-Type: application/json" \
  -d '{}'
```
