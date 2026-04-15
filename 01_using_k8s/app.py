from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load trained model
model_filepath = "model/model.pkl"

with open(model_filepath, "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

FEATURE_COLUMNS = [
    "Hours_Studied",
    "Sleep_Hours",
    "Attendance",
]

TARGET_COLUMN = "Exam_Score"


@app.route("/")
def home():
    return "Flask API is running!"


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "Hours_Studied": float(data.get("Hours_Studied", 0)),
            "Sleep_Hours": float(data.get("Sleep_Hours", 0)),
            "Attendance": float(data.get("Attendance", 0)),
        }])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_exam_score": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)