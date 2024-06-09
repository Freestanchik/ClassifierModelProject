from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

app = Flask(__name__)

MODEL_FILE = 'model.joblib'
SCALER_FILE = 'scaler.joblib'

# Load the model from a file if it exists
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = None


@app.route('/train', methods=['POST'])
def train():
    global model, scaler
    data = request.get_json()
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Save the model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return jsonify({'message': 'Model trained and saved successfully', 'accuracy': accuracy}), 200


if __name__ == '__main__':
    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model is not trained yet'}), 400

    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({'error': 'Input data should be a list of JSON objects'}), 400

    df = pd.DataFrame(data).drop(columns=['reportDate'])

    scaler = joblib.load(SCALER_FILE)

    X_scaled = scaler.transform(df)

    predictions = model.predict(X_scaled)

    return jsonify({'predictions': predictions.tolist()}), 200


if __name__ == '__main__':
    app.run(debug=True)