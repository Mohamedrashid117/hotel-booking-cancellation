from pathlib import Path

import cloudpickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hotel_cancellation_pipeline.pkl"

with MODEL_PATH.open("rb") as file:
    model_package = cloudpickle.load(file)

model = model_package["pipeline"]
input_columns = model_package["input_columns"]

# Read category choices from the trained encoder.
preprocessor = model.named_steps["preprocessing"]

categorical_columns = next(
    columns
    for name, transformer, columns in preprocessor.transformers_
    if name == "categorical"
)

encoder = preprocessor.named_transformers_["categorical"]

category_options = {
    column: [str(value) for value in values]
    for column, values in zip(
        categorical_columns, encoder.categories_
    )
}

# Prepare form fields in the model's expected order.
fields = [
    {
        "name": column,
        "label": column.replace("_", " ").title(),
        "options": category_options.get(column)
    }
    for column in input_columns
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()

        try:
            booking = {}

            for column in input_columns:
                value = form_values.get(column, "").strip()

                if not value:
                    raise ValueError(
                        f"Please fill in {column.replace('_', ' ')}."
                    )

                if column in category_options:
                    if value not in category_options[column]:
                        raise ValueError(
                            f"Please select a valid value for {column}."
                        )

                    # The cleaning pipeline expects numeric IDs or missing values.
                    if column in ["agent", "company"]:
                        booking[column] = (
                            float("nan")
                            if value in ["No_Agent", "No_Company"]
                            else int(value)
                        )
                    else:
                        booking[column] = value

                else:
                    try:
                        number = int(value)
                    except ValueError:
                        raise ValueError(
                            f"{column.replace('_', ' ')} must be a whole number."
                        ) from None

                    if number < 0:
                        raise ValueError(
                            f"{column.replace('_', ' ')} cannot be negative."
                        )

                    booking[column] = number

            if booking["is_repeated_guest"] not in [0, 1]:
                raise ValueError("Repeated guest must be 0 or 1.")

            if not 1 <= booking["arrival_date_day_of_month"] <= 31:
                raise ValueError("Arrival day must be between 1 and 31.")

            if not 1 <= booking["arrival_date_week_number"] <= 53:
                raise ValueError("Arrival week must be between 1 and 53.")

            if not 1 <= booking["arrival_date_year"] <= 9999:
                raise ValueError("Please enter a valid arrival year.")

            if (
                booking["adults"]
                + booking["children"]
                + booking["babies"]
                == 0
            ):
                raise ValueError("At least one guest is required.")

            input_data = pd.DataFrame(
                [booking], columns=input_columns
            )

            prediction = int(model.predict(input_data)[0])

            class_index = list(model.classes_).index(1)
            probability = model.predict_proba(input_data)[0, class_index]

            result = {
                "label": (
                    "Likely to cancel"
                    if prediction == 1
                    else "Likely not to cancel"
                ),
                "probability": round(float(probability) * 100, 1)
            }

        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        fields=fields,
        form_values=form_values,
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=False)