from flask import Flask, render_template, request
import pickle
import pandas as pd

import os
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

# Load model
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
columns = data["columns"]

@app.route("/")
def home():
    return render_template("index.html", columns=columns)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {}

    for col in columns:
        input_data[col] = float(request.form.get(col, 0))

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    return render_template("index.html",
                           columns=columns,
                           prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)