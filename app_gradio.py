"""Gradio app: assign a credit card customer to a segment.
Run:  python app_gradio.py   (needs: pip install gradio)
Run the notebook first so models/segmentation_model.joblib exists.
"""
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr

bundle = joblib.load("models/segmentation_model.joblib")
scaler, kmeans = bundle["scaler"], bundle["kmeans"]
columns, names = bundle["columns"], bundle["segment_names"]
medians = json.load(open("models/feature_medians.json"))

ADVICE = {
    "Premium Big Spenders": "Low risk. Offer rewards, a premium card and a limit increase.",
    "Installment Buyers": "Low to medium risk. Offer EMI deals and cashback on instalments.",
    "High-Balance Revolvers": "High risk. Monitor closely and offer a balance-transfer or repayment plan.",
    "Cash-Advance Dependents": "Highest risk. Add to the early-warning list; offer counselling and cap cash advances.",
    "Dormant / Low Activity": "Low risk but a churn risk. Run a re-activation campaign.",
}

def predict_segment(balance, purchases, oneoff, installments, cash_advance,
                    credit_limit, payments, purchases_freq, cash_adv_freq, full_payment_pct):
    row = dict(medians)  # the features not asked for use the dataset median
    row.update({
        "BALANCE": balance, "PURCHASES": purchases, "ONEOFF_PURCHASES": oneoff,
        "INSTALLMENTS_PURCHASES": installments, "CASH_ADVANCE": cash_advance,
        "CREDIT_LIMIT": credit_limit, "PAYMENTS": payments,
        "PURCHASES_FREQUENCY": purchases_freq, "CASH_ADVANCE_FREQUENCY": cash_adv_freq,
        "PRC_FULL_PAYMENT": full_payment_pct,
    })
    x = pd.DataFrame([row])[columns]
    seg = int(kmeans.predict(scaler.transform(np.log1p(x)))[0])
    name = names[str(seg)] if str(seg) in names else names[seg]
    return f"{name}\n\n{ADVICE[name]}"

demo = gr.Interface(
    fn=predict_segment,
    inputs=[
        gr.Number(label="Balance", value=1500), gr.Number(label="Purchases", value=1000),
        gr.Number(label="One-off purchases", value=500), gr.Number(label="Instalment purchases", value=400),
        gr.Number(label="Cash advance", value=0), gr.Number(label="Credit limit", value=4500),
        gr.Number(label="Payments", value=1700),
        gr.Slider(0, 1, value=0.5, label="Purchases frequency"),
        gr.Slider(0, 1, value=0.0, label="Cash advance frequency"),
        gr.Slider(0, 1, value=0.15, label="Share of months paid in full"),
    ],
    outputs=gr.Textbox(label="Predicted segment and suggested action"),
    title="Credit Card Customer Segmentation",
    description="Enter a customer's card usage to see which segment they belong to.",
)

if __name__ == "__main__":
    demo.launch()
