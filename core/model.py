from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train_model(data):
    features = ["RSI", "EMA_10", "EMA_20", "MACD"]
    label_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}
    X = data[features].astype(float)
    y = data["Signal"].map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    model.fit(X_train, y_train)
    return model

def predict_signal(model, data):
    features = ["RSI", "EMA_10", "EMA_20", "MACD"]
    label_reverse_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
    pred_numeric = model.predict([data[features].iloc[-1]])[0]
    return label_reverse_mapping[pred_numeric]


def predict_latest_signal(data):
    """Train on all historical rows up to the last row (no leakage) and predict the label for the latest row.

    This function trains an XGB model on data.iloc[:-1] and predicts for data.iloc[-1].
    Returns the string label ("BUY","HOLD","SELL") or None if not enough data.
    """
    features = ["RSI", "EMA_10", "EMA_20", "MACD"]
    label_reverse_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}

    if data is None or len(data) < 5:
        return None

    train = data.iloc[:-1]
    test_row = data.iloc[-1]

    X_train = train[features].astype(float)
    y_train = train["Signal"].map({"SELL": 0, "HOLD": 1, "BUY": 2})

    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    model.fit(X_train, y_train)

    pred_numeric = model.predict([test_row[features].astype(float).values])[0]
    return label_reverse_mapping.get(pred_numeric)
