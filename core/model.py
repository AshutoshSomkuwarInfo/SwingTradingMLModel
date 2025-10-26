from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Base features (original set)
BASE_FEATURES = ["RSI", "EMA_10", "EMA_20", "MACD"]

# Extended features (includes new technical indicators)
EXTENDED_FEATURES = [
    "RSI", "EMA_10", "EMA_20", "MACD",
    "BB_Width", "Stoch", "ATR",
    "Price_to_EMA10", "Price_to_EMA20"
]

def train_model(data, use_extended_features=False, random_state=42):
    """
    Train XGBoost model with optional extended feature set
    
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
    
    # Improved hyperparameters for better performance
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
    """
    # Use extended features if available, otherwise base features
    features = BASE_FEATURES
    if all(f in data.columns for f in ["BB_Width", "Stoch", "ATR"]):
        features = EXTENDED_FEATURES
    
    available_features = [f for f in features if f in data.columns]
    label_reverse_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}

    if data is None or len(data) < 5:
        return None

    train = data.iloc[:-1]
    test_row = data.iloc[-1]

    X_train = train[available_features].astype(float)
    y_train = train["Signal"].map({"SELL": 0, "HOLD": 1, "BUY": 2})

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

    pred_numeric = model.predict([test_row[available_features].astype(float).values])[0]
    return label_reverse_mapping.get(pred_numeric)
