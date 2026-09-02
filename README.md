# AAPL-stock-prediction-model

Predict AAPL (Apple Inc.) stock prices using time-series models and deep learning. This repository contains data preprocessing, feature engineering, model training, and evaluation pipelines used for experimenting with stock price prediction.

## Contents

- data/ — (optional) raw and processed datasets
- src/ — code for preprocessing, feature engineering, model training, and inference
- notebooks/ — exploratory notebooks and experiments
- models/ — saved model checkpoints

## Quick start

1. Clone the repo:
   ```bash
   git clone https://github.com/KhushiKeswani/AAPL-stock-prediction-model.git
   cd AAPL-stock-prediction-model
   ```
2. Create and activate a Python environment (Python 3.8+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Prepare data and run training (example):
   ```bash
   python src/preprocess.py --input data/raw/aapl.csv --output data/preprocessed
   python src/train.py --data data/preprocessed --epochs 50
   ```

## Notes

- Replace dataset paths with your local files if not included.
- This project is intended for learning and experimentation and does not constitute financial advice.

## License

MIT
