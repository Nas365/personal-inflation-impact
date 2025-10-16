# app/main.py
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import joblib

from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from nicegui import ui

# ---------------- CONSTANTS ---------------- #
CATS = ['Food', 'Housing', 'Transport', 'Health', 'Recreation', 'Misc']
LABELS = {
    'Food': 'Food',
    'Housing': 'Housing',
    'Transport': 'Transport',
    'Health': 'Health',
    'Recreation': 'Recreation',
    'Misc': 'Miscellaneous',
}
FEATS = CATS + [f'w_{c}' for c in CATS]

BASE = Path(__file__).resolve().parent.parent
ART = BASE / 'artifacts'
DATA_W = BASE / 'data' / 'cpih_wide_yoy.csv'
ASSETS = BASE / 'app' / 'assets'  # harmless if missing

# ---------------- LOAD ARTIFACTS ---------------- #
_cls = joblib.load(ART / 'cls_model.joblib')
_reg = joblib.load(ART / 'reg_model.joblib')
CLF, THR = _cls['model'], float(_cls['threshold'])
REG = _reg['model']

# ---------------- HELPERS ---------------- #
def latest_row():
    df = pd.read_csv(DATA_W, parse_dates=['date']).sort_values('date')
    return df.iloc[-1]

def normalize_weights(raw: dict) -> dict:
    raw = raw or {}
    s = sum(max(0.0, float(raw.get(c, 0))) for c in CATS) or 1.0
    return {c: max(0.0, float(raw.get(c, 0))) / s for c in CATS}

def make_features(latest, w_norm):
    row = {c: float(latest[c]) for c in CATS}
    row.update({f'w_{c}': w_norm[c] for c in CATS})
    return pd.DataFrame([row])[FEATS]

# ---------------- FASTAPI BACKEND ---------------- #
fastapi = FastAPI(title='Personal Inflation Impact API')
fastapi.mount('/assets', StaticFiles(directory=str(ASSETS)), name='assets')

@fastapi.get('/healthz')
def healthz():
    return {'status': 'ok'}

@fastapi.post('/predict')
def predict(payload: dict):
    w = normalize_weights(payload)
    last = latest_row()
    X = make_features(last, w)
    proba = float(CLF.predict_proba(X)[:, 1][0])
    flag = 'HIGH' if proba >= THR else 'LOW'
    yhat = float(REG.predict(X)[0])
    return {
        'latest_headline_cpih_pct': round(float(last.get('Headline', float('nan'))), 2),
        'latest_month': last['date'].strftime('%b %Y'),
        'forecast_personal_inflation_pct': round(yhat, 2),
        'risk_probability': round(proba, 4),
        'threshold': round(THR, 3),
        'risk_flag': flag,
        'weights_normalized': w,
    }

# ---------------- NICEGUI FRONTEND ---------------- #
ui.page_title('Personal Inflation Impact (UK)')
ui.dark_mode().enable()

# 1) Intro (always first)
with ui.column().style('max-width: 1100px; margin: 12px auto; gap: 8px; padding: 12px;'):
    ui.label('Personal Inflation Impact (UK)').style('font-size: 1.4rem; font-weight: 700;')
    ui.label(
        'Inflation affects every household differently. This tool uses official UK ONS CPIH data and your budget '
        'to estimate your personal inflation and flag potential risk over the next three months. '
        'Adjust the sliders to reflect your spending mix.'
    )

# 2) About (always second)
with ui.column().style('max-width: 1100px; margin: 6px auto; gap: 8px; padding: 12px;'):
    ui.label('About this tool').style('font-weight:700;')
    ui.label('Source: UK Office for National Statistics (ONS) CPIH divisions (monthly).')
    ui.label('What you do: move sliders to reflect your typical spending mix; values need not sum to 100 — we normalize.')
    ui.label('What you get: a 3-month personal inflation forecast and a risk probability/flag versus the headline CPIH.')

# 3) Tool / Inputs (always after About)
with ui.column().style('max-width: 1100px; margin: 6px auto; gap: 10px; padding: 12px;'):
    ui.label('Your budget (we auto-normalize)').style('font-weight:600;')
    total_lbl = ui.label()

    sliders = {}
    defaults = {"Housing": 35, "Food": 25, "Transport": 15, "Health": 5, "Recreation": 8, "Misc": 12}

    def update_total():
        total_lbl.text = f"Total (unnormalized): {sum(s.value for s in sliders.values()):.1f}"

    for c in CATS:
        with ui.row().style('width:100%; align-items:center; gap:12px;'):
            ui.label(LABELS[c]).style('min-width:160px;')
            s = ui.slider(min=0, max=100, value=defaults.get(c, 10), step=1).props('label-always').style('width:100%')
            sliders[c] = s
    update_total()

    last = latest_row()
    latest_headline = float(last.get('Headline', float('nan')))
    latest_month = last['date'].strftime('%b %Y')
    ui.separator()
    ui.label(f'Latest published UK CPIH (headline): {latest_headline:.2f}%  [{latest_month}]')\
      .style('margin-top:4px; font-weight:600;')

    out_forecast = ui.label()
    out_proba = ui.label()
    out_flag = ui.label()

    def run():
        weights = {k: sliders[k].value for k in CATS}
        X = make_features(latest_row(), normalize_weights(weights))
        p = float(CLF.predict_proba(X)[:, 1][0])
        flag = 'HIGH' if p >= THR else 'LOW'
        y = float(REG.predict(X)[0])
        out_forecast.text = f'Your 3-month forecast: {y:.2f}%   (vs headline {latest_headline:.2f}% now)'
        out_proba.text    = f'Risk probability: {p:.3f}   (threshold={THR:.3f})'
        out_flag.text     = f'Risk flag: {flag}'

    ui.button('CALCULATE', on_click=run, color='primary').style('margin-top:6px; width:160px;')

# Footer
with ui.row().style('width:100%; justify-content:center; margin: 10px 0 20px 0;'):
    ui.label('© Nasir Abubakar').style('opacity:0.8;')

# ---------------- START APP ---------------- #
ui.run_with(fastapi)
ui.run(host='0.0.0.0', port=int(os.getenv('PORT', '8080')))
