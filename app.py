
import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load the model
with open('mobile_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_price_range(
    battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
    m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w,
    talk_time, three_g, touch_screen, wifi
):
    # Create engineered features
    pixel_area = px_height * px_width
    screen_area = sc_h * sc_w
    total_camera_mp = fc + pc
    has_advanced_features = four_g + three_g + touch_screen + wifi

    # Feature names (must match training order)
    feature_names = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 
                     'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 
                     'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 
                     'touch_screen', 'wifi', 'pixel_area', 'screen_area', 
                     'total_camera_mp', 'has_advanced_features']

    input_data = pd.DataFrame([[
        battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
        m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w,
        talk_time, three_g, touch_screen, wifi,
        pixel_area, screen_area, total_camera_mp, has_advanced_features
    ]], columns=feature_names)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    price_ranges = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    result = f"Predicted Price Range: {price_ranges[prediction]} (Class {prediction})"
    confidence = f"Confidence: {probabilities[prediction]:.2%}"
    all_probs = "\n".join([
        f"{price_ranges[i]}: {prob:.2%}" for i, prob in enumerate(probabilities)
    ])

    return f"{result}\n{confidence}\n\nAll Probabilities:\n{all_probs}"

iface = gr.Interface(
    fn=predict_price_range,
    inputs=[
        gr.Number(label="Battery Power (mAh)", value=1500),
        gr.Radio([0, 1], label="Bluetooth", value=1),
        gr.Slider(0.5, 3.0, step=0.1, label="Clock Speed (GHz)", value=1.5),
        gr.Radio([0, 1], label="Dual SIM", value=1),
        gr.Number(label="Front Camera (MP)", value=5),
        gr.Radio([0, 1], label="4G", value=1),
        gr.Number(label="Internal Memory (GB)", value=32),
        gr.Slider(0.1, 1.0, step=0.1, label="Mobile Depth (cm)", value=0.5),
        gr.Number(label="Mobile Weight (g)", value=150),
        gr.Slider(1, 8, step=1, label="Number of Cores", value=4),
        gr.Number(label="Primary Camera (MP)", value=12),
        gr.Number(label="Pixel Height", value=1080),
        gr.Number(label="Pixel Width", value=1920),
        gr.Number(label="RAM (MB)", value=2048),
        gr.Number(label="Screen Height (cm)", value=12),
        gr.Number(label="Screen Width (cm)", value=6),
        gr.Number(label="Talk Time (hours)", value=10),
        gr.Radio([0, 1], label="3G", value=1),
        gr.Radio([0, 1], label="Touch Screen", value=1),
        gr.Radio([0, 1], label="WiFi", value=1)
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=8),
    title="Mobile Price Range Predictor",
    description="Enter mobile phone specifications to predict the price range",
    examples=[
        [1500, 1, 2.0, 1, 5, 1, 32, 0.5, 150, 4, 12, 1080, 1920, 2048, 12, 6, 10, 1, 1, 1],
        [800, 0, 0.5, 0, 2, 0, 16, 0.3, 180, 2, 5, 720, 1280, 512, 10, 5, 5, 1, 0, 0]
    ]
)

iface.launch()
