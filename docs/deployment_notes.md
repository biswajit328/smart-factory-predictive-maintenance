# Deployment Notes

This project is still a prototype, but the API and dashboard are packaged so they can run as a small local stack.

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

Start the full local stack:

```bash
docker compose up api dashboard redis postgres mqtt mqtt-bridge
```

Open:

```text
API docs:  http://127.0.0.1:8000/docs
Dashboard: http://127.0.0.1:8501
```

Replay simulated sensor messages through MQTT:

```bash
docker compose --profile replay up mqtt-replay
```

## Hosted Demo Path

The repo includes a GitHub Pages workflow:

```text
.github/workflows/pages.yml
```

It builds a static demo site from:

- `assets/project-preview.png`
- `assets/api-contract.png`
- `outputs/v2/smart_factory_dashboard.html`

After pushing to `main`, GitHub Actions can publish the static demo site through GitHub Pages.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PM_API_HOST` | `127.0.0.1` | API host outside Docker |
| `PM_API_PORT` | `8000` | API port |
| `PM_OUTPUT_DIR` | `outputs` | Where models, metrics, and reports are saved |
| `PM_LOG_LEVEL` | `INFO` | JSON log level |
| `PM_REDIS_URL` | `redis://redis:6379/0` | Redis service URL used by the local stack |
| `PM_DATABASE_URL` | `postgresql://maintenance:maintenance@postgres:5432/maintenance` | Postgres service URL used by the local stack |
| `PM_MQTT_BROKER_HOST` | `localhost` | MQTT broker host |
| `PM_MQTT_BROKER_PORT` | `1883` | MQTT broker port |
| `PM_MQTT_SENSOR_TOPIC` | `factory/sensors` | MQTT topic for sensor events |
| `PM_V2_NUM_MACHINES` | `10` | Number of simulated machines |
| `PM_V2_NUM_STEPS` | `120` | Number of timesteps per machine |
| `PM_V2_EPOCHS` | `5` | Neural training epochs |

## Production Gaps

For a real deployment, I would not keep stream state only inside the API process.

The current Compose stack includes Redis, Postgres, and MQTT to show the deployment shape. The API does not yet fully persist state into those services.

The next production version should use:

- Redis or another external store for machine buffers
- MQTT or Kafka for sensor ingestion
- Postgres or another database for prediction history
- model versioning for deployed artifacts
- drift monitoring for incoming sensor distributions
- alert ownership and maintenance workflow tracking
