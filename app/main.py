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
ASSETS = BASE / 'app' / 'assets'  # not required anymore, but harmless if present

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

# (Optional) static mount left in place; safe to remove if you prefer
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
ui.add_head_html("""
<style>
  html, body, #q-app { height: 100%; }

  /* Minimal, asset-free background */
  #q-app {
    background:
      radial-gradient(1200px 600px at 15% 5%, rgba(255,255,255,0.06), transparent 60%),
      linear-gradient(135deg, #0b1020 0%, #1b2235 50%, #101522 100%);
  }

  .panel {
    background: rgba(0,0,0,0.55);
    padding: 18px 20px;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    backdrop-filter: blur(2px);
  }

  .title { font-size: 1.65rem; font-weight: 700; }
  .intro { line-height: 1.5; }
  .bullet { margin: 6px 0; line-height: 1.45; }
  .footer { opacity: 0.8; font-size: 0.95rem; }
</style>
""")

ui.page_title('Personal Inflation Impact (UK)')
ui.dark_mode().enable()

# Intro banner
with ui.column().classes('panel').style('max-width: 1100px; margin: 12px auto;'):
    ui.label('Personal Inflation Impact (UK)').classes('title')
    ui.label(
        'Inflation affects every household differently. This tool uses official UK ONS CPIH data and your budget '
        'to estimate your personal inflation and flag potential risk over the next three months. '
        'Fine-tune your spending mix with the sliders to see how changes could support better financial stability.'
    ).classes('intro')

# Main layout: inputs (left) and about (right)
with ui.row().style('gap:28px; width:100%; max-width:1100px; margin: 0 auto; align-items:flex-start;'):
    # LEFT
    with ui.column().classes('panel').style('flex:1; min-width:420px;'):
        ui.label('Your budget (we auto-normalize)').style('margin-top:6px; font-weight:600;')
        total_lbl = ui.label()

        sliders = {}
        defaults = {"Housing": 35, "Food": 25, "Transport": 15, "Health": 5, "Recreation": 8, "Misc": 12}

        def update_total():
            total_lbl.text = f"Total (unnormalized): {sum(s.value for s in sliders.values()):.1f}"

        for c in CATS:
            with ui.row().style('width:100%; align-items:center; gap:12px;'):
                ui.label(LABELS[c]).style('min-width:160px;')
                s = (ui.slider(min=0, max=100, value=defaults.get(c, 10), step=1)
                       .props('label-always').style('width:100%'))
                sliders[c] = s
        update_total()

        last = latest_row()
        latest_headline = float(last.get('Headline', float('nan')))
        latest_month = last['date'].strftime('%b %Y')

        ui.separator()
        ui.label(
            f'Latest published UK CPIH (headline): {latest_headline:.2f}%  [{latest_month}]'
        ).style('margin:4px 0 0 0; font-weight:600;')

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

        ui.button('CALCULATE', on_click=run, color='primary').style('margin-top:10px; width:160px;')

    # RIGHT
    with ui.column().classes('panel').style('flex:1; min-width:440px;'):
        ui.label('About this tool').style('font-weight:700;')
        ui.label('Source: UK Office for National Statistics (ONS) CPIH divisions (monthly).').classes('bullet')
        ui.label('What you do: move sliders to reflect your typical spending mix; values need not sum to 100 — we normalize.').classes('bullet')
        ui.label('What you get: a 3-month personal inflation forecast and a risk probability/flag if your personal rate may exceed the headline CPIH by a policy margin (e.g., +2pp).').classes('bullet')
        ui.separator()
        ui.label(
            'Why it matters: being conscious of inflation pressures across your own basket helps you plan, adjust spending, '
            'and avoid a disproportionate squeeze on your budget.'
        ).classes('bullet')

# Footer
with ui.row().style('width:100%; justify-content:center; margin: 10px 0 20px 0;'):
    ui.label('© Nasir Abubakar').classes('footer')

# ---------------- START APP ---------------- #
ui.run_with(fastapi)
ui.run(host='0.0.0.0', port=int(os.getenv('PORT', '8080')))
