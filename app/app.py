import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from utils import generate_bubble_chart

app = Flask(__name__)

# Safe Path Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Load Models
regression_data = joblib.load(os.path.join(MODELS_DIR, "regression_model.pkl"))
classification_data = joblib.load(os.path.join(MODELS_DIR, "classification_model.pkl"))
clustering_data = joblib.load(os.path.join(MODELS_DIR, "clustering_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
q1, q2 = joblib.load(os.path.join(MODELS_DIR, "class_thresholds.pkl"))

regression_model = regression_data["model"]
regression_type = regression_data["type"]
regression_feature = regression_data["feature"]

classification_model = classification_data["model"]
clustering_model = clustering_data["model"]
cluster_features = clustering_data["features"]

# ROUTES

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "MedInc": float(request.form.get("MedInc", 0)),
        "HouseAge": float(request.form.get("HouseAge", 0)),
        "AveRooms": float(request.form.get("AveRooms", 0)),
        "AveBedrms": float(request.form.get("AveBedrms", 0)),
        "Population": float(request.form.get("Population", 0)),
        "AveOccup": float(request.form.get("AveOccup", 0)),
        "Latitude": float(request.form.get("Latitude", 0)),
        "Longitude": float(request.form.get("Longitude", 0)),
    }

    df_input = pd.DataFrame([input_data])

    df_scaled = pd.DataFrame(
        scaler.transform(df_input),
        columns=df_input.columns,
        index=df_input.index
    )

    # Regression
    if regression_type == "multiple":
        price_pred = regression_model.predict(df_scaled)[0]
    else:
        price_pred = regression_model.predict(
            df_scaled[[regression_feature]]
        )[0]

    # Classification
    category_pred = classification_model.predict(df_scaled)[0]

    category_map = {0: "Low", 1: "Medium", 2: "High"}
    category_label = category_map.get(category_pred, "Unknown")

    # Clustering
    cluster_input = df_scaled[cluster_features]
    cluster_pred = clustering_model.predict(cluster_input)[0]

    cluster_descriptions = {
        0: "Lower-income inland regions with moderate population density.",
        1: "Medium-income suburban regions with balanced demographics.",
        2: "High-income coastal regions with dense population."
    }

    cluster_description = cluster_descriptions.get(
        cluster_pred,
        "Cluster profile unavailable."
    )

    # Bubble chart
    chart_path = generate_bubble_chart(
        user_income=input_data["MedInc"],
        user_price=price_pred,
        user_population=input_data["Population"]
    )

    return render_template(
        "result.html",
        price_formatted=f"${price_pred * 100000:,.2f}",
        category=category_label,
        cluster=cluster_pred,
        cluster_description=cluster_description,
        chart_path=chart_path
    )

if __name__ == "__main__":
    app.run(debug=True)