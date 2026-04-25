# Deployment Notes

This project is still a prototype, but the API is packaged so it can be run in a container.

## Local Docker Flow

Build the image:

```bash
docker compose build
```

Train the v2 model inside the container:

```bash
docker compose run --rm api python -m src.v2_train
```

Start the API:

```bash
docker compose up api
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PM_API_HOST` | `127.0.0.1` | API host outside Docker |
| `PM_API_PORT` | `8000` | API port |
| `PM_OUTPUT_DIR` | `outputs` | Where models, metrics, and reports are saved |
| `PM_LOG_LEVEL` | `INFO` | JSON log level |
| `PM_V2_NUM_MACHINES` | `10` | Number of simulated machines |
| `PM_V2_NUM_STEPS` | `120` | Number of timesteps per machine |
| `PM_V2_EPOCHS` | `5` | Neural training epochs |

## Production Gaps

For a real deployment, I would not keep stream state only inside the API process.

The next production version should use:

- Redis or another external store for machine buffers
- MQTT or Kafka for sensor ingestion
- Postgres or another database for prediction history
- model versioning for deployed artifacts
- drift monitoring for incoming sensor distributions
- alert ownership and maintenance workflow tracking
