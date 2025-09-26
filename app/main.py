from __future__ import annotations
from nicegui import ui, app as niceapp
from fastapi import FastAPI
from pathlib import Path
import pandas as pd, joblib, os

CATS = ['Food','Housing','Transport','Health','Recreation','Misc']
FEATS = CATS + [f"w_{c}" for c in CATS]
BASE = Path(__file__).resolve().parent.parent
ART = BASE / "artifacts"
DAT = BASE / "data" / "cpih_wide_yoy.csv"

# Load artifacts
cls = joblib.load(ART / "cls_model.joblib")
reg = joblib.load(ART / "reg_model.joblib")
CLF, THR = cls["model"], float(cls["threshold"])
REG = reg["model"]

def latest_row():
    df = pd.read_csv(DAT, parse_dates=['date']).sort_values('date')
    return df.iloc[-1]

def normalize_weights(raw):
    s = sum(max(0.0, float(raw.get(c,0))) for c in CATS) or 1.0
    return {c: max(0.0, float(raw.get(c,0)))/s for c in CATS}

def make_features(latest, w_norm):
    row = {c: float(latest[c]) for c in CATS}
    row.update({f"w_{c}": w_norm[c] for c in CATS})
    return pd.DataFrame([row])[FEATS]

# FastAPI
fastapi: FastAPI = niceapp.native

@fastapi.post("/predict")
def predict(payload: dict):
    # payload: {"Housing":35, "Food":25, ...} (unnormalized ok)
    w = normalize_weights(payload)
    latest = latest_row()
    X = make_features(latest, w)

    proba = float(CLF.predict_proba(X)[:,1][0])
    flag  = "HIGH" if proba >= THR else "LOW"
    yhat  = float(REG.predict(X)[0])

    return {
        "forecast_personal_inflation_pct": round(yhat,2),
        "risk_probability": round(proba,4),
        "threshold": round(THR,3),
        "risk_flag": flag,
        "weights_normalized": w
    }

# UI
ui.dark_mode().enable()
ui.page_title('Personal Inflation Impact (UK)')
ui.label('Personal Inflation Impact (UK)').style('font-size: 1.6rem; font-weight: 600;')

with ui.row().style('gap:24px; width:100%; align-items:flex-start;'):
    with ui.column().style('flex:1; min-width:360px;'):
        sliders = {}
        ui.label('Your budget (we auto-normalize)').style('font-weight:600;')
        total_lbl = ui.label()

        def update_total():
            tot = sum(s.value for s in sliders.values())
            total_lbl.text = f'Total (unnormalized): {tot:.1f}'

        defaults = {"Housing":35,"Food":25,"Transport":15,"Health":5,"Recreation":8,"Misc":12}
        for c in CATS:
            s = ui.slider(min=0, max=100, value=defaults.get(c,10), step=1, on_change=update_total)
            with ui.row():
                ui.label(c).style('min-width:140px;')
                s.props('label-always').style('width:300px;')
            sliders[c] = s
        update_total()

        out_forecast = ui.label()
        out_proba    = ui.label()
        out_flag     = ui.label()

        def run():
            weights = {k: sliders[k].value for k in CATS}
            latest = latest_row()
            X = make_features(latest, normalize_weights(weights))
            proba = float(CLF.predict_proba(X)[:,1][0])
            flag  = "HIGH" if proba >= THR else "LOW"
            yhat  = float(REG.predict(X)[0])

            out_forecast.text = f"3-month forecast: {yhat:.2f}%"
            out_proba.text    = f"Risk probability: {proba:.3f} (thr={THR:.3f})"
            out_flag.text     = f"Risk flag: {flag}"

        ui.button('Calculate', on_click=run, color='primary').style('margin-top:12px;')

ui.run(host='0.0.0.0', port=int(os.getenv('PORT', '8501')))
