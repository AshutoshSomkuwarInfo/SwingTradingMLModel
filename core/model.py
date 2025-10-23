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
