from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import pandas as pd

# Base features (original set)
BASE_FEATURES = ["RSI", "EMA_10", "EMA_20", "MACD"]

# Extended features (includes new technical indicators)
# Using only reliable, well-tested features
EXTENDED_FEATURES = [
    "RSI", "EMA_10", "EMA_20", "MACD", "MACD_Signal",
    "BB_Width", "Stoch", "ATR",
    "Price_to_EMA10", "Price_to_EMA20"
]

def train_model(data, use_extended_features=False, random_state=42):
    """
    Train XGBoost model with optional extended feature set
    Balanced hyperparameters for better accuracy
    
    Args:
        data: DataFrame with features and Signal column
        use_extended_features: If True, use EXTENDED_FEATURES, else BASE_FEATURES
        random_state: Random seed for reproducibility
    """
    features = EXTENDED_FEATURES if use_extended_features else BASE_FEATURES
    # Filter to only available features
    available_features = [f for f in features if f in data.columns]
    
    if not available_features:
        raise ValueError("No valid features found in data")
    
    label_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}
    X = data[available_features].astype(float)
    y = data["Signal"].map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Remove NaN rows for better training
    valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if len(X_train) == 0:
        raise ValueError("No valid training data after removing NaNs")
    
    # Original hyperparameters that worked better - simpler is often better
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        use_label_encoder=False, 
        eval_metric="mlogloss", 
        verbosity=0,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    return model, available_features

def predict_signal(model, data, features=None):
    if features is None:
        features = BASE_FEATURES
    
    # Filter to available features
    available_features = [f for f in features if f in data.columns]
    label_reverse_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
    pred_numeric = model.predict([data[available_features].iloc[-1]])[0]
    return label_reverse_mapping[pred_numeric]

def predict_signal_with_probability(model, data, features=None):
    """Return prediction with probability confidence"""
    if features is None:
        features = BASE_FEATURES
    
    available_features = [f for f in features if f in data.columns]
    label_reverse_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
    pred_proba = model.predict_proba([data[available_features].iloc[-1]])[0]
    pred_numeric = model.predict([data[available_features].iloc[-1]])[0]
    confidence = max(pred_proba) * 100
    return label_reverse_mapping[pred_numeric], round(confidence, 2)

def get_feature_importance(model, features=None):
    """Extract and return feature importance"""
    if features is None:
        features = BASE_FEATURES
    importance = model.feature_importances_
    return dict(zip(features, importance))


def predict_latest_signal(data):
    """Train on all historical rows up to the last row (no leakage) and predict the label for the latest row.

    This function trains an XGB model on data.iloc[:-1] and predicts for data.iloc[-1].
    Returns the string label ("BUY","HOLD","SELL") or None if not enough data.
    Uses BASE features only - simpler is better for this problem.
    """
    # Use BASE features only - the original set that worked
    features = BASE_FEATURES
    available_features = [f for f in features if f in data.columns]
    
    # Must have all base features
    if len(available_features) < len(BASE_FEATURES):
        return None
        
    label_reverse_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}

    if data is None or len(data) < 30:
        return None

    train = data.iloc[:-1]
    test_row = data.iloc[-1]

    X_train = train[available_features].astype(float)
    y_train = train["Signal"].map({"SELL": 0, "HOLD": 1, "BUY": 2})
    
    # Remove any rows with NaN
    valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if len(X_train) < 20 or len(y_train) < 20:
        return None
    
    # Check class distribution
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) < 2:  # Need at least 2 classes
        return None

    # Original simpler hyperparameters that worked better
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        use_label_encoder=False, 
        eval_metric="mlogloss", 
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    # Check for NaN in test row
    test_features = test_row[available_features].astype(float)
    if test_features.isna().any():
        return None

    pred_numeric = model.predict([test_features.values])[0]
    return label_reverse_mapping.get(pred_numeric)
