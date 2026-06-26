"""Airline passenger satisfaction classifier — Gradio app for Hugging Face Spaces.

This is a serving app, not a training app.
No model is trained in this file.

The fitted pipelines were produced offline by `train_and_save_model.ipynb`.
This file only loads the saved `model.joblib` artifact and uses it for prediction.

The app contains three tabs:

1. EDA         — descriptive statistics and simple feature plots.
2. Model Card  — comparison of the candidate models trained offline.
3. Predict     — prediction for a new passenger profile.

"""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import plotly.express as px


SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR / "data" / "airline_passenger_satisfaction_sample.csv"
MODEL_PATH = SCRIPT_DIR / "model.joblib"
COMPARISON_PATH = SCRIPT_DIR / "model_comparison.csv"


# ---------------------------------------------------------------------------
# Load data and fitted model bundle
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

_BUNDLE = joblib.load(MODEL_PATH)

PIPELINES = _BUNDLE["pipelines"]
FEATURES = _BUNDLE["feature_names"]
TARGET = _BUNDLE["target_name"]
POSITIVE = _BUNDLE["positive_label"]
NEGATIVE = _BUNDLE["negative_label"]
WINNER = _BUNDLE["winner_name"]
JUSTIFICATION = _BUNDLE["justification"]

ALGORITHMS = list(PIPELINES)

COMPARISON = pd.read_csv(COMPARISON_PATH)


# ---------------------------------------------------------------------------
# Tab 1 — EDA
# ---------------------------------------------------------------------------

def eda_summary():
    """Return summary statistics for the bundled sample."""
    summary = df[FEATURES].describe(include="all").T.reset_index()
    summary = summary.rename(columns={"index": "feature"})
    return summary


def eda_target_plot():
    """Create a bar chart for the target distribution."""
    counts = df[TARGET].value_counts().reset_index()
    counts.columns = [TARGET, "count"]

    fig = px.bar(
        counts,
        x=TARGET,
        y="count",
        color=TARGET,
        text="count",
        title=f"Target distribution ({TARGET})",
        height=320,
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)

    return fig


def eda_feature_plot(feature):
    """Create a simple feature plot grouped by the target variable."""
    fig = px.histogram(
        df,
        x=feature,
        color=TARGET,
        barmode="overlay",
        nbins=30,
        title=f"{feature} by {TARGET}",
        height=360,
        opacity=0.65,
    )

    return fig


# ---------------------------------------------------------------------------
# Tab 3 — Predict
# ---------------------------------------------------------------------------

def predict_one(
    algorithm,
    gender,
    customer_type,
    age,
    type_of_travel,
    travel_class,
    flight_distance,
    inflight_wifi_service,
    departure_arrival_time_convenient,
    ease_of_online_booking,
    gate_location,
    food_and_drink,
    online_boarding,
    seat_comfort,
    inflight_entertainment,
    on_board_service,
    leg_room_service,
    baggage_handling,
    checkin_service,
    inflight_service,
    cleanliness,
    departure_delay,
    arrival_delay,
):
    """Predict passenger satisfaction for one new passenger profile."""

    pipeline = PIPELINES[algorithm]

    row = pd.DataFrame([{
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": inflight_wifi_service,
        "Departure/Arrival time convenient": departure_arrival_time_convenient,
        "Ease of Online booking": ease_of_online_booking,
        "Gate location": gate_location,
        "Food and drink": food_and_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": on_board_service,
        "Leg room service": leg_room_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
    }])

    # Keep the exact same feature order used during training
    row = row[FEATURES]

    prediction = pipeline.predict(row)[0]

    probabilities = pipeline.predict_proba(row)[0]
    classes = list(pipeline.classes_)

    confidence = probabilities[classes.index(prediction)]

    note = " *(F1 winner)*" if algorithm == WINNER else ""

    return (
        f"**{algorithm}**{note}\n\n"
        f"Prediction: **{prediction}**\n\n"
        f"Confidence: **{confidence:.3f}**"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Airline Passenger Satisfaction Classifier") as demo:
    gr.Markdown(
        "## Airline Passenger Satisfaction Classifier\n\n"
        "Advanced optional assignment for the Python course.\n\n"
        "The model was trained offline in `train_and_save_model.ipynb`. "
        "This Gradio app only loads the saved `model.joblib` artifact and serves predictions."
    )

    with gr.Tabs():

        # -------------------------------------------------------------------
        # EDA tab
        # -------------------------------------------------------------------
        with gr.Tab("EDA"):
            gr.Markdown("### Summary statistics of the bundled sample")
            gr.Dataframe(
                value=eda_summary(),
                interactive=False,
            )

            with gr.Row():
                gr.Plot(
                    value=eda_target_plot(),
                    label="Target distribution",
                )

                feature_dropdown = gr.Dropdown(
                    choices=FEATURES,
                    value="Age",
                    label="Feature to plot",
                )

            feature_plot = gr.Plot(
                value=eda_feature_plot("Age"),
                label="Feature plot",
            )

            feature_dropdown.change(
                fn=eda_feature_plot,
                inputs=feature_dropdown,
                outputs=feature_plot,
            )

        # -------------------------------------------------------------------
        # Model Card tab
        # -------------------------------------------------------------------
        with gr.Tab("Model Card"):
            gr.Markdown(
                f"### Model selection — chosen: **{WINNER}**\n\n"
                f"{JUSTIFICATION}\n\n"
                "All candidate models were trained offline on the same train/test split "
                "in `train_and_save_model.ipynb`. The table below is loaded from "
                "`model_comparison.csv`."
            )

            gr.Dataframe(
                value=COMPARISON,
                label="Candidate model comparison",
                interactive=False,
            )

        # -------------------------------------------------------------------
        # Predict tab
        # -------------------------------------------------------------------
        with gr.Tab("Predict"):
            gr.Markdown(
                "### Predict satisfaction for a new passenger\n\n"
                "Choose one of the fitted algorithms, enter the passenger and flight "
                "information, and click **Predict**. The default algorithm is the F1 winner "
                "from the Model Card."
            )

            algorithm_dropdown = gr.Dropdown(
                choices=ALGORITHMS,
                value=WINNER,
                label="Algorithm",
            )

            with gr.Row():
                gender = gr.Radio(
                    choices=["Female", "Male"],
                    value="Female",
                    label="Gender",
                )

                customer_type = gr.Radio(
                    choices=["Loyal Customer", "disloyal Customer"],
                    value="Loyal Customer",
                    label="Customer Type",
                )

            age = gr.Slider(
                minimum=7,
                maximum=85,
                value=40,
                step=1,
                label="Age",
            )

            with gr.Row():
                type_of_travel = gr.Radio(
                    choices=["Business travel", "Personal Travel"],
                    value="Business travel",
                    label="Type of Travel",
                )

                travel_class = gr.Radio(
                    choices=["Business", "Eco", "Eco Plus"],
                    value="Business",
                    label="Class",
                )

            flight_distance = gr.Slider(
                minimum=31,
                maximum=4983,
                value=843,
                step=1,
                label="Flight Distance",
            )

            gr.Markdown("### Service ratings")

            with gr.Row():
                inflight_wifi_service = gr.Slider(0, 5, value=3, step=1, label="Inflight wifi service")
                departure_arrival_time_convenient = gr.Slider(0, 5, value=3, step=1, label="Departure/Arrival time convenient")
                ease_of_online_booking = gr.Slider(0, 5, value=3, step=1, label="Ease of Online booking")

            with gr.Row():
                gate_location = gr.Slider(0, 5, value=3, step=1, label="Gate location")
                food_and_drink = gr.Slider(0, 5, value=3, step=1, label="Food and drink")
                online_boarding = gr.Slider(0, 5, value=3, step=1, label="Online boarding")

            with gr.Row():
                seat_comfort = gr.Slider(0, 5, value=4, step=1, label="Seat comfort")
                inflight_entertainment = gr.Slider(0, 5, value=4, step=1, label="Inflight entertainment")
                on_board_service = gr.Slider(0, 5, value=4, step=1, label="On-board service")

            with gr.Row():
                leg_room_service = gr.Slider(0, 5, value=4, step=1, label="Leg room service")
                baggage_handling = gr.Slider(1, 5, value=4, step=1, label="Baggage handling")
                checkin_service = gr.Slider(0, 5, value=3, step=1, label="Checkin service")

            with gr.Row():
                inflight_service = gr.Slider(0, 5, value=4, step=1, label="Inflight service")
                cleanliness = gr.Slider(0, 5, value=3, step=1, label="Cleanliness")

            gr.Markdown("### Delay information")

            with gr.Row():
                departure_delay = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=0,
                    step=1,
                    label="Departure Delay in Minutes",
                )

                arrival_delay = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=0,
                    step=1,
                    label="Arrival Delay in Minutes",
                )

            predict_button = gr.Button("Predict", variant="primary")

            prediction_output = gr.Markdown()

            predict_button.click(
                fn=predict_one,
                inputs=[
                    algorithm_dropdown,
                    gender,
                    customer_type,
                    age,
                    type_of_travel,
                    travel_class,
                    flight_distance,
                    inflight_wifi_service,
                    departure_arrival_time_convenient,
                    ease_of_online_booking,
                    gate_location,
                    food_and_drink,
                    online_boarding,
                    seat_comfort,
                    inflight_entertainment,
                    on_board_service,
                    leg_room_service,
                    baggage_handling,
                    checkin_service,
                    inflight_service,
                    cleanliness,
                    departure_delay,
                    arrival_delay,
                ],
                outputs=prediction_output,
            )


if __name__ == "__main__":
    demo.launch()