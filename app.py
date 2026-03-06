from flask import Flask, render_template, request
import pickle
import csv
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("diabetes_model.sav", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        gender = request.form["Gender"]
        age = float(request.form["Age"])

        if gender == "Male":
            pregnancies = 0
        else:
            pregnancies = float(request.form["Pregnancies"])

        glucose = float(request.form["Glucose"])
        bp = float(request.form["BloodPressure"])
        skin = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        dpf = float(request.form["DPF"])

        # Create dataframe (prevents sklearn warning)
        input_data = pd.DataFrame(
            [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        )

        # Prediction
        result = model.predict(input_data)
        probability = model.predict_proba(input_data)

        prob = round(probability[0][1] * 100, 2)

        if result[0] == 1:
            prediction = "The patient has diabetes"
            advice = "Consult a doctor and maintain a healthy lifestyle."
            result_text = "Diabetes"
        else:
            prediction = "The patient does not have diabetes"
            advice = "Maintain healthy diet and regular exercise."
            result_text = "No Diabetes"

        # Save prediction history
        with open("diabetes_history.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([gender, age, glucose, bmi, prob, result_text])

        return render_template(
            "result.html",
            prediction=prediction,
            prob=prob,
            advice=advice
        )

    return render_template("index.html")


@app.route("/history")
def history():

    data = pd.read_csv("diabetes_history.csv")

    return render_template(
        "history.html",
        tables=data.values,
        columns=data.columns
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)