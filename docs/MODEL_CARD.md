# Model Card

## Project

Neural Predictive Maintenance System for Smart Factories using Multi-Sensor Data Fusion.

## Model Versions

### v1 baseline

- Model: `RandomForestClassifier`
- Input: original tabular machine readings plus engineered physics-style features
- Output: failure probability, thresholded failure flag, risk band, and recommendation
- Purpose: honest tabular baseline for the original dataset

### v2 neural fusion model

- Model: temporal sensor-fusion CNN
- Input: rolling windows of simulated multi-sensor streams
- Branches:
  - thermal: air temperature, process temperature, humidity
  - mechanical: rotational speed, torque, tool wear, vibration, machine type
  - electrical: pressure, current, acoustic signal
- Output: failure probability, risk band, maintenance priority, and recommendation

## Current Metrics

### v1 baseline

| Metric | Value |
|---|---:|
| ROC-AUC | `0.9848` |
| PR-AUC | `0.8619` |
| Precision | `0.9322` |
| Recall | `0.8088` |
| Brier score | `0.0097` |

### v2 neural simulation

| Metric | Value |
|---|---:|
| ROC-AUC | `0.9928` |
| PR-AUC | `0.8890` |
| Precision | `0.6667` |
| Recall | `1.0000` |
| Brier score | `0.0397` |

## Thresholding

The models do not use a hard-coded `0.5` decision threshold.

The threshold is selected on validation data using an F-beta strategy because predictive maintenance usually cares more about missing failures than about sending a few extra alerts.

Saved threshold artifacts:

- `outputs/threshold_analysis.csv`
- `outputs/v2/threshold_analysis.csv`

## Explainability

v1 uses feature importance from the tree model.

v2 uses branch ablation. Each sensor branch is removed and the drop in PR-AUC is measured. In the current run, the electrical branch has the largest PR-AUC drop, so the model depends most on that branch in the simulation.

Saved explainability artifacts:

- `outputs/feature_importance.csv`
- `outputs/v2/branch_importance.csv`
- `outputs/v2/branch_importance.png`

## Calibration

The project saves calibration curves and Brier scores because the output is used as a risk score, not only as a class label.

Saved calibration artifacts:

- `outputs/calibration_curve.png`
- `outputs/v2/calibration_curve.png`

## Intended Use

This project is intended as a prototype and portfolio system for predictive maintenance workflows.

It is useful for:

- demonstrating leakage-free training
- testing sensor-stream fusion ideas
- showing API-based model serving
- explaining threshold and alert tradeoffs

It is not ready for real maintenance decisions without real telemetry validation.

## Limitations

- v2 is trained and tested on simulated streams, not real factory time-series data.
- The API keeps machine stream state in memory.
- There is no production alert workflow or maintenance ticket integration.
- The model has not been tested for drift across real machine types, operating conditions, or factories.
- The cost of false positives and false negatives is not yet tied to a real maintenance budget.

## Risk Notes

False negatives could miss upcoming failures. False positives could create unnecessary maintenance work.

Before real use, this model would need:

- real telemetry validation
- calibration review
- drift monitoring
- operator feedback loop
- maintenance cost model
- rollback and model versioning
