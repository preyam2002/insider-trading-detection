# Insider Trading Detection

ML-based system for detecting insider trading patterns in financial markets using TensorFlow, XGBoost, and custom feature engineering.

## Features

- **Feature engineering** — financial ML features (price momentum, volume anomalies, options activity, timing patterns)
- **Multiple models** — TensorFlow neural network + XGBoost gradient boosting ensemble
- **FastAPI server** — REST API for real-time prediction
- **C++ components** — high-performance feature processor for production workloads
- **Configurable** — YAML-based configuration for model parameters and thresholds

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML | TensorFlow, XGBoost, scikit-learn |
| Data | Pandas, NumPy |
| API | FastAPI |
| Performance | C++17 (CMake) |
| Config | YAML |

## Quick Start

```bash
pip install -r requirements.txt
python src/training/train.py     # Train models
python src/api/main.py           # Start API server
```

## Architecture

```
src/
├── features/
│   └── feature_engineering.py   # Financial feature extraction
├── training/
│   └── train.py                 # Model training pipeline
├── detection/
│   ├── detector.py              # Basic detector
│   └── enhanced_detector.py     # Multi-factor enhanced detection
├── api/
│   └── main.py                  # FastAPI server
└── utils/
    └── data_collector.py        # Market data collection

cpp/
├── detector.hpp                 # C++ detection engine
└── main.cpp                     # CLI + benchmarks
```

## License

MIT — Built by [Preyam](https://github.com/preyam2002)
