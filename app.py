import os
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and supporting files
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))
feature_order = pickle.load(open("model/feature_order.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        input_data = {
            "loan_amnt": float(request.form["loan_amnt"]),
            "revenue": float(request.form["revenue"]),
            "dti_n": float(request.form["dti_n"]),
            "fico_n": float(request.form["fico_n"]),
            "experience_c": float(request.form["experience_c"]),
            "emp_length": request.form["emp_length"],
            "purpose": request.form["purpose"],
            "home_ownership_n": request.form["home_ownership_n"],
            "addr_state": request.form["addr_state"]
        }

        # Prepare DataFrame
        df_input = pd.DataFrame([input_data])

        # Encode categorical columns
        for col, le in encoders.items():
            if col in df_input.columns:
                if df_input[col].values[0] not in le.classes_:
                    # Handle unseen label
                    le.classes_ = np.append(le.classes_, df_input[col].values[0])
                df_input[col] = le.transform(df_input[col])

        # Reorder columns to match training
        df_input = df_input.reindex(columns=feature_order)

        # Scale numeric values
        scaled_input = scaler.transform(df_input)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        result = "Default" if prediction == 1 else "No Default"
        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT environment variable
    app.run(host="0.0.0.0", port=port, debug=True)
