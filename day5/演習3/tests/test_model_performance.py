import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")

# ベースラインモデルのパス（存在していなければ現在のモデルがコピーされる）
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model_baseline.pkl")

# 推論性能を評価するための閾値
ACCURACY_THRESHOLD = 0.75
PRECISION_THRESHOLD = 0.7
RECALL_THRESHOLD = 0.65
F1_THRESHOLD = 0.7
INFERENCE_TIME_THRESHOLD = 0.5  # 秒


@pytest.fixture
def load_dataset():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"データファイルが存在しません: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Survived", axis=1)
    y = df["Survived"].astype(int)
    return X, y


@pytest.fixture
def load_current_model():
    """現在のモデルを読み込む"""
    if not os.path.exists(CURRENT_MODEL_PATH):
        pytest.skip(f"現在のモデルファイルが存在しません: {CURRENT_MODEL_PATH}")
    
    with open(CURRENT_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@pytest.fixture
def load_baseline_model(load_current_model):
    """ベースラインモデルを読み込む。存在しない場合は現在のモデルをコピーする"""
    if not os.path.exists(BASELINE_MODEL_PATH):
        # ベースラインモデルがない場合は、現在のモデルをコピーする
        with open(CURRENT_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        with open(BASELINE_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        
        print(f"ベースラインモデルが存在しないため、現在のモデルをコピーしました: {BASELINE_MODEL_PATH}")
    
    with open(BASELINE_MODEL_PATH, "rb") as f:
        baseline_model = pickle.load(f)
    
    return baseline_model


def get_model_metrics(model, X, y):
    """モデルの評価指標を計算する"""
    # 推論時間の計測
    start_time = time.time()
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    inference_time = time.time() - start_time

    # 各種評価指標の計算
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "inference_time": inference_time,
    }
    
    return metrics


def test_model_performance_metrics(load_dataset, load_current_model):
    """モデルのパフォーマンス指標を検証する"""
    X, y = load_dataset
    model = load_current_model
    
    metrics = get_model_metrics(model, X, y)
    
    print(f"\n現在のモデルの評価指標:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 各指標が閾値を超えていることを確認
    assert metrics["accuracy"] >= ACCURACY_THRESHOLD, f"精度が閾値を下回っています: {metrics['accuracy']:.4f} < {ACCURACY_THRESHOLD}"
    assert metrics["precision"] >= PRECISION_THRESHOLD, f"適合率が閾値を下回っています: {metrics['precision']:.4f} < {PRECISION_THRESHOLD}"
    assert metrics["recall"] >= RECALL_THRESHOLD, f"再現率が閾値を下回っています: {metrics['recall']:.4f} < {RECALL_THRESHOLD}"
    assert metrics["f1"] >= F1_THRESHOLD, f"F1スコアが閾値を下回っています: {metrics['f1']:.4f} < {F1_THRESHOLD}"
    assert metrics["inference_time"] <= INFERENCE_TIME_THRESHOLD, f"推論時間が閾値を超えています: {metrics['inference_time']:.4f} > {INFERENCE_TIME_THRESHOLD}"


def test_compare_with_baseline(load_dataset, load_current_model, load_baseline_model):
    """現在のモデルとベースラインモデルのパフォーマンスを比較する"""
    X, y = load_dataset
    current_model = load_current_model
    baseline_model = load_baseline_model
    
    # 両方のモデルの評価指標を計算
    current_metrics = get_model_metrics(current_model, X, y)
    baseline_metrics = get_model_metrics(baseline_model, X, y)
    
    print(f"\n現在のモデルの評価指標:")
    for metric_name, value in current_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print(f"\nベースラインモデルの評価指標:")
    for metric_name, value in baseline_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 性能指標の比較
    performance_degradation = False
    degradation_messages = []
    
    # 精度、適合率、再現率、F1スコアは大きい方が良い
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if current_metrics[metric] < 0.95 * baseline_metrics[metric]:  # 5%以上の性能劣化を検出
            performance_degradation = True
            degradation_messages.append(
                f"{metric}の性能劣化: {current_metrics[metric]:.4f} < {baseline_metrics[metric]:.4f} (5%超の劣化)"
            )
    
    # 推論時間は小さい方が良い
    if current_metrics["inference_time"] > 1.2 * baseline_metrics["inference_time"]:  # 20%以上の性能劣化を検出
        performance_degradation = True
        degradation_messages.append(
            f"推論時間の劣化: {current_metrics['inference_time']:.4f} > {baseline_metrics['inference_time']:.4f} (20%超の劣化)"
        )
    
    # 性能劣化がある場合はテストを失敗させる
    if performance_degradation:
        degradation_message = "\n".join(degradation_messages)
        assert False, f"⚠️ モデルの性能劣化が検出されました:\n{degradation_message}"
    else:
        print("\n✅ モデルの性能劣化は検出されませんでした")