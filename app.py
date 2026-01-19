# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# gradio app

import gradio as gr
import pandas as pd
import pickle
import numpy as np
from joblib import load

# 1. Load the Model
model=load("gb_final.joblib")


def predict_productivity(wake_up_hour, exercise, meditation, breakfast,
                         screen_time_before_work_min, early_riser):

    early_riser = 1 if early_riser == "Yes" else 0
    input_df = pd.DataFrame([[
        wake_up_hour, exercise, meditation, breakfast,
        screen_time_before_work_min, early_riser
    ]], columns=[
        'wake_up_hour', 'exercise', 'meditation', 'breakfast',
        'screen_time_before_work_min', 'early_riser'
    ])

    prediction = model.predict(input_df)[0]
    return f"Productivity Score: {(prediction):.2f}"

# 3. The App Interface
inputs = [
    gr.Number(
        label="Wake-up Time (Hour)",
        value=6.5,
        precision=2,
        step=0.01
    ),

    gr.Radio(
        ["Yes", "No"],
        label="Exercise"
    ),

    gr.Radio(
        ["Yes", "No"],
        label="Meditation"
    ),

    gr.Radio(
        ["Yes", "No"],
        label="Breakfast"
    ),

    gr.Number(
        label="Screen Time Before Work (minutes)",
        value=30
    ),
    gr.Radio(
        ["Yes", "No"],
        label="Early Riser"
    ),
]


app = gr.Interface(
    fn=predict_productivity,
    inputs=inputs,
    outputs="text",
    title="Productivity Prediction")

app.launch(share=True)