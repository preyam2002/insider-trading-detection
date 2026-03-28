import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_collector import StockDataFetcher, fetch_all_data
from src.features.feature_engineering import FeatureEngineer, get_feature_columns
from src.models.insider_models import (
    RandomForestModel, XGBoostModel, SVMModel, 
    print_evaluation, evaluate_model
)
from src.models.model_exporter import export_rf_to_binary


def load_config(config_path: str = "config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config: dict):
    tickers = config.get('TICKERS', ['AAPL', 'GOOGL', 'MSFT'])
    train_start = config.get('TRAIN_START_DATE', '2018-01-01')
    train_end = config.get('TRAIN_END_DATE', '2022-12-31')
    test_start = config.get('TEST_START_DATE', '2023-01-01')
    test_end = config.get('TEST_END_DATE', '2023-12-31')
    
    print(f"Fetching training data from {train_start} to {train_end}")
    train_data = fetch_all_data(tickers, train_start, train_end)
    
    print(f"Fetching test data from {test_start} to {test_end}")
    test_data = fetch_all_data(tickers, test_start, test_end)
    
    return train_data, test_data


def engineer_features(stock_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_windows = config.get('FEATURE_WINDOWS', [7, 14, 30, 60])
    threshold = config.get('INSIDER_TRADING_THRESHOLD', 0.05)
    
    engineer = FeatureEngineer(feature_windows=feature_windows)
    
    processed = engineer.engineer_all_features(
        stock_data=stock_data,
        insider_transactions=pd.DataFrame(),
        fundamental_data={},
        market_returns=None
    )
    
    return processed


def train_and_evaluate(X_train, y_train, X_test, y_test, config: dict):
    results = {}
    
    print("\n=== Training Random Forest ===")
    rf_model = RandomForestModel(n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print_evaluation(y_test, rf_pred)
    results['random_forest'] = evaluate_model(y_test, rf_pred)
    
    print("\n=== Training XGBoost ===")
    xgb_model = XGBoostModel(n_estimators=100, max_depth=6)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    print_evaluation(y_test, xgb_pred)
    results['xgboost'] = evaluate_model(y_test, xgb_pred)
    
    print("\n=== Training SVM ===")
    svm_model = SVMModel(kernel='rbf', C=1.0)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    print_evaluation(y_test, svm_pred)
    results['svm'] = evaluate_model(y_test, svm_pred)
    
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest model: {best_model[0]} with F1: {best_model[1]['f1']:.4f}")
    
    return rf_model, xgb_model, svm_model, results


def save_models(rf_model, xgb_model, svm_model, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    rf_path = os.path.join(output_dir, 'random_forest.pkl')
    xgb_path = os.path.join(output_dir, 'xgboost.pkl')
    svm_path = os.path.join(output_dir, 'svm.pkl')
    
    rf_model.save(rf_path)
    xgb_model.save(xgb_path)
    svm_model.save(svm_path)
    
    print(f"Models saved to {output_dir}")
    
    return rf_path, xgb_path, svm_path


def main():
    parser = argparse.ArgumentParser(description='Train insider trading detection models')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--output', default='models', help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("=== Loading Data ===")
    train_data, test_data = prepare_data(config)
    
    print("\n=== Engineering Features ===")
    train_features = engineer_features(train_data, config)
    test_features = engineer_features(test_data, config)
    
    feature_cols = get_feature_columns()
    
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    X_test = test_features[feature_cols].values
    y_test = test_features['label'].values
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training labels: {np.sum(y_train)} positive, {len(y_train) - np.sum(y_train)} negative")
    print(f"Test labels: {np.sum(y_test)} positive, {len(y_test) - np.sum(y_test)} negative")
    
    rf_model, xgb_model, svm_model, results = train_and_evaluate(
        X_train, y_train, X_test, y_test, config
    )
    
    save_models(rf_model, xgb_model, svm_model, args.output)
    
    export_rf_to_binary(rf_model.model, rf_model.scaler, 
                        os.path.join(args.output, 'random_forest.bin'))
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
