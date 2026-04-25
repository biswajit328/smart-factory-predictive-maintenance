# Data Card

## Data Sources

The project has two data layers.

### Original dataset

The v1 baseline uses `data/predictive_maintenance.csv`.

This is a tabular predictive maintenance dataset with machine type, temperature, rotational speed, torque, tool wear, and failure labels.

### Simulated smart-factory stream

The v2 system creates a simulated stream using ranges and relationships grounded in the original dataset.

The simulator adds:

- multiple machine IDs
- timestamps
- vibration
- pressure
- current
- acoustic signal
- humidity
- breakdown events
- failure-horizon labels

## Why Simulation Is Used

The original dataset is not a true industrial time-series stream. It does not contain live sensor events per machine over time.

Because the core project idea is about real-time multi-sensor fusion, v2 builds a realistic simulation layer around the original data distribution.

This is useful for system design, but it is not the same as real factory telemetry.

## Label Definition

### v1

Target: `Machine failure`

### v2

Target: `failure_next_horizon`

This means: a machine is labeled positive if a breakdown event occurs within the configured future horizon.

## Splitting Strategy

v1 uses train/validation/test splits on the tabular dataset.

v2 splits by machine ID, so test windows come from machines not used during training.

This is stricter than randomly mixing windows because it checks whether the model generalizes to different machines in the simulation.

## Known Biases And Gaps

- The simulator is based on assumptions about how sensor values change before failure.
- Real factories may have different sensor noise, failure modes, missing values, and maintenance interventions.
- There are no real operators, maintenance logs, or downtime costs in the data.
- The class balance in simulation is controlled by the simulator design.
- The original data has limited sensor variety compared with real industrial systems.

## Recommended Next Dataset Upgrade

To make this project stronger, test it on a real industrial time-series dataset with:

- multiple machines
- timestamped sensor readings
- maintenance events
- failure timestamps
- operating regimes
- missing or delayed sensor data

That would move the project from a strong prototype toward a serious applied ML system.
