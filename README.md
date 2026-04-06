# DeadNeurons

**A self-improving neural decoder with full MLOps lifecycle. Because dead neurons don't retrain themselves.**

## What This Is

An end-to-end MLOps pipeline that decodes mouse decision-making from real Neuropixels brain recordings (Steinmetz 2019). The core model is a neural network built in pure NumPy that monitors its own internals, detects dead and saturated neurons, and fixes them automatically during training.

## The Problem

Given the spike activity of 60-1769 neurons recorded during a visual decision task, predict which direction the mouse chose. Binary classification on real neural population data across 26 recording sessions from 7 mice.

## Results

| Metric | Value |
|--------|-------|
| Mean test accuracy | 84.3% |
| Best session | 98.0% |
| Sessions above chance | 25/26 |
| Total sessions | 26 |
| Total usable trials | 4,869 |

## Self-Improving Framework

Standard neural networks accumulate dead neurons over training. Weights drift into regions where ReLU outputs zero for every input. The neuron stops learning permanently. Nobody notices until accuracy plateaus.

This framework adds three phases after every training epoch:

1. **Self-Observation** records per-neuron activation statistics (mean, variance) across the training batch
2. **Self-Diagnosis** detects dead neurons (zero output for all inputs) and saturated neurons (max output for all inputs)
3. **Self-Correction** reinitializes dead neurons with He initialization and rescales saturated neurons

The network watches itself and heals itself. No external intervention needed.

## Architecture

```
Steinmetz Spike Data (neurons x trials x time_bins)
    |
    v
Feature Extraction (mean firing rates per neuron per time window)
    |
    v
PCA (reduce to 50 principal components)
    |
    v
Self-Improving Neural Network (input -> 32 hidden ReLU -> 1 sigmoid)
    |
    v
Decision Prediction (left or right)
```

## Project Structure

```
DeadNeurons/
├── src/
│   ├── features/       extractor.py (spike data loading and feature extraction)
│   ├── model/          classifier.py (self-improving neural network)
│   ├── tracking/       experiment tracker (Phase 2)
│   ├── registry/       model registry (Phase 3)
│   ├── validation/     data validation (Phase 4)
│   ├── monitoring/     drift detection (Phase 4)
│   └── api/            FastAPI serving (Phase 5)
├── dashboard/          Streamlit monitoring dashboard (Phase 6)
├── notebooks/          exploration and analysis
├── .github/workflows/  CI/CD pipelines (Phase 7)
├── train.py            main training script
├── Dockerfile          container definition (Phase 5)
└── requirements.txt    dependencies
```

## Quick Start

```bash
git clone https://github.com/Rekhii/DeadNeurons.git
cd DeadNeurons
pip install -r requirements.txt

# Train single session
python train.py --session 0 --epochs 150 --hidden 32

# Train all sessions
python train.py --epochs 150 --hidden 32
```

## Dataset

Steinmetz 2019 Neuropixels recordings. 26 sessions, 7 mice, 4869 usable trials. Each session records 474-1769 neurons simultaneously across multiple brain regions while mice perform a visual contrast discrimination task.

Not included in repo due to size. Download from the original source and place in `data/synthetic/`.

## Tech Stack

- **Core model**: Pure NumPy (no deep learning frameworks)
- **API**: FastAPI (Phase 5)
- **Dashboard**: Streamlit (Phase 6)
- **Model registry**: Hugging Face Hub (Phase 3)
- **CI/CD**: GitHub Actions (Phase 7)
- **Deployment**: Render.com + Docker (Phase 5)
- **Cost**: Zero

## MLOps Pipeline (In Progress)

- [x] Phase 1: Feature extraction + self-improving classifier
- [x] Phase 2: Experiment tracking
- [x] Phase 3: Model registry with versioning
- [ ] Phase 4: Data validation + drift detection
- [ ] Phase 5: FastAPI serving + Docker
- [ ] Phase 6: Monitoring dashboard
- [ ] Phase 7: CI/CD + automated retraining

## Author

[Rekhii](https://github.com/Rekhii)
