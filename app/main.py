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

# ================================================================
# NICEGUI FRONTEND
# ================================================================
ui.page_title('Personal Inflation Impact (UK)')

# Set Quasar primary colour to government blue
ui.colors(primary='#003078')

# ── Global styles & Google Fonts ─────────────────────────────────────
ui.add_head_html('''
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  /* ── Design tokens ── */
  :root {
    --gov-blue:        #003078;
    --gov-blue-light:  #1d70b8;
    --gov-blue-mid:    #1a4480;
    --bg-page:         #f0f2f5;
    --bg-card:         #ffffff;
    --text-primary:    #0b0c0c;
    --text-muted:      #505a5f;
    --accent-warning:  #f47738;
    --accent-success:  #00703c;
    --accent-warn-bg:  #fff7ed;
    --accent-ok-bg:    #f0fdf4;
    --border:          #b1b4b6;
  }

  /* ── Base ── */
  body, .q-page, .nicegui-content {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
    background: var(--bg-page) !important;
    color: var(--text-primary) !important;
    margin: 0;
    padding: 0 !important;
  }

  /* ── Cold-start banner ── */
  .cold-banner {
    background: #dbeafe;
    border-bottom: 2px solid #93c5fd;
    color: #1e3a5f;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    font-size: 0.875rem;
    line-height: 1.4;
    position: sticky;
    top: 0;
    z-index: 999;
  }

  /* ── Hero header ── */
  .gov-header {
    background: linear-gradient(135deg, var(--gov-blue) 0%, var(--gov-blue-mid) 100%);
    color: #ffffff;
    padding: 32px 24px 28px;
  }
  .gov-header-inner {
    max-width: 1100px;
    margin: 0 auto;
  }
  .ons-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.7rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 12px;
  }
  .gov-header h1 {
    font-size: clamp(1.4rem, 3vw, 1.9rem);
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.3px;
    line-height: 1.2;
  }
  .gov-header p {
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.88;
    max-width: 740px;
    line-height: 1.6;
  }

  /* ── Card containers ── */
  .app-card {
    background: var(--bg-card);
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04);
    padding: 22px 26px;
    margin-bottom: 18px;
  }

  /* ── Section titles ── */
  .section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--gov-blue);
    border-bottom: 3px solid var(--gov-blue-light);
    padding-bottom: 7px;
    margin-bottom: 16px;
  }

  /* ── Slider rows ── */
  .slider-row {
    padding: 9px 0;
    border-bottom: 1px solid #f0f0f0;
  }
  .slider-row:last-child { border-bottom: none; }
  .cat-label {
    font-weight: 500;
    color: var(--text-primary);
    min-width: 160px;
    font-size: 0.93rem;
  }
  .pct-badge {
    background: var(--gov-blue);
    color: #ffffff;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    min-width: 52px;
    text-align: center;
    letter-spacing: 0.3px;
  }

  /* ── Metric box ── */
  .metric-box {
    background: linear-gradient(135deg, var(--gov-blue) 0%, var(--gov-blue-mid) 100%);
    color: #ffffff;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    min-width: 180px;
  }
  .metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.5px;
  }
  .metric-label {
    font-size: 0.7rem;
    opacity: 0.8;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
  }
  .metric-month {
    font-size: 0.8rem;
    opacity: 0.65;
    margin-top: 3px;
  }

  /* ── Calculate button ── */
  .calc-btn {
    background: var(--gov-blue) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 11px 28px !important;
    border-radius: 6px !important;
    letter-spacing: 0.4px;
    transition: background 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 6px rgba(0,48,120,0.25);
  }
  .calc-btn:hover {
    background: #001f52 !important;
    box-shadow: 0 4px 12px rgba(0,48,120,0.35) !important;
  }

  /* ── Result card ── */
  .result-card {
    border-radius: 10px;
    padding: 20px 24px;
    border-left: 5px solid var(--border);
    background: #f9fafb;
    transition: background 0.3s ease, border-color 0.3s ease;
  }
  .result-card.high {
    background: var(--accent-warn-bg);
    border-left-color: var(--accent-warning);
  }
  .result-card.low {
    background: var(--accent-ok-bg);
    border-left-color: var(--accent-success);
  }
  .result-forecast {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
  }
  .result-proba {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-top: 6px;
  }
  .result-flag {
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .result-flag.high { color: var(--accent-warning); }
  .result-flag.low  { color: var(--accent-success); }

  /* ── Footer ── */
  .app-footer {
    background: var(--gov-blue);
    color: rgba(255,255,255,0.8);
    padding: 24px 24px;
    font-size: 0.84rem;
    text-align: center;
    margin-top: 10px;
    line-height: 1.8;
  }
  .app-footer a {
    color: #93c5fd;
    text-decoration: none;
  }
  .app-footer a:hover { text-decoration: underline; }
  .app-footer strong { color: #ffffff; }

  /* ── About two-col grid ── */
  .about-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 640px) {
    .about-grid { grid-template-columns: 1fr; }
    .gov-header { padding: 22px 16px 18px; }
    .app-card { padding: 16px 14px; }
    .metric-box { min-width: unset; width: 100%; }
  }
</style>
''')

# ── 1. Cold-start banner ──────────────────────────────────────────────
with ui.element('div').classes('cold-banner') as _banner:
    ui.html(
        '<span>⏳ <strong>First load may take up to 60 seconds</strong> — '
        'this app runs on Render\'s free tier and pulls live ONS data. '
        'Thank you for your patience!</span>'
    )
    ui.button('✕', on_click=lambda: _banner.set_visibility(False)) \
      .props('flat dense size=sm') \
      .style('color:#1e3a5f; font-weight:700; padding:0 8px; min-height:unset;')

# ── 2. Hero header ────────────────────────────────────────────────────
ui.html('''
<div class="gov-header">
  <div class="gov-header-inner">
    <div class="ons-badge">ONS CPIH Data</div>
    <h1>Personal Inflation Impact Calculator (UK)</h1>
    <p>
      Inflation affects every household differently. This tool uses official UK ONS CPIH data
      and your spending mix to estimate your <strong>personal 3-month inflation forecast</strong>
      and flag your risk level versus the national headline rate.
    </p>
  </div>
</div>
''')

# ── 3. Main content column ────────────────────────────────────────────
with ui.column().style('max-width:1100px; margin:24px auto 0; padding:0 16px 40px; gap:0;'):

    # ── About card ────────────────────────────────────────────────
    with ui.element('div').classes('app-card'):
        ui.html('<div class="section-title">About This Tool</div>')
        ui.html('''
        <div class="about-grid">
          <div>
            <p style="margin:0 0 10px 0;">
              <strong>Source:</strong> UK Office for National Statistics (ONS)
              CPIH divisions — monthly data via public API.
            </p>
            <p style="margin:0 0 10px 0;">
              <strong>What you do:</strong> Move sliders to reflect your typical
              spending mix as a percentage of your household budget.
            </p>
          </div>
          <div>
            <p style="margin:0 0 10px 0;">
              <strong>What you get:</strong> A 3-month personal inflation forecast
              and a risk probability/flag versus the headline CPIH.
            </p>
            <p style="margin:0; font-size:0.85rem; color:#505a5f;">
              🔒 <em>Privacy: no personal data is collected or stored.
              Your inputs stay in the browser and are used only to compute
              the on-page forecast.</em>
            </p>
          </div>
        </div>
        ''')

    # ── Sliders card ──────────────────────────────────────────────
    with ui.element('div').classes('app-card'):
        ui.html('<div class="section-title">Your Budget Breakdown</div>')
        ui.html(
            '<p style="margin:0 0 14px 0; font-size:0.88rem; color:#505a5f;">'
            'Adjust each slider to match your typical monthly spending. '
            'Percentages are normalized automatically so they always sum to 100%.'
            '</p>'
        )

        total_lbl = ui.label() \
                      .style('font-size:0.82rem; color:#666; margin-bottom:10px;')

        sliders: dict[str, ui.slider] = {}
        pct_labels: dict[str, ui.label] = {}
        defaults = {"Housing": 35, "Food": 25, "Transport": 15, "Health": 5, "Recreation": 8, "Misc": 12}

        def refresh_totals_and_percents() -> None:
            total = sum(s.value for s in sliders.values())
            total_lbl.text = f"Total (unnormalized): {total:.1f}"
            w_norm = normalize_weights({k: sliders[k].value for k in CATS})
            for k in CATS:
                pct_labels[k].text = f"{w_norm[k]*100:.0f}%"

        for c in CATS:
            with ui.row().classes('slider-row') \
                         .style('width:100%; align-items:center; gap:12px; flex-wrap:nowrap;'):
                ui.html(f'<span class="cat-label">{LABELS[c]}</span>')
                s = ui.slider(min=0, max=100, value=defaults.get(c, 10), step=1) \
                       .props('label-always color=primary') \
                       .style('flex:1; min-width:80px;')
                sliders[c] = s
                pct_labels[c] = ui.label('0%').classes('pct-badge')
                s.on_value_change(lambda e: refresh_totals_and_percents())

        refresh_totals_and_percents()

    # ── Latest ONS data ───────────────────────────────────────────
    last = latest_row()
    latest_headline = float(last.get('Headline', float('nan')))
    latest_month = last['date'].strftime('%b %Y')

    with ui.element('div').classes('app-card'):
        ui.html('<div class="section-title">Latest ONS Data &amp; Your Forecast</div>')

        with ui.row().style('gap:20px; align-items:flex-start; flex-wrap:wrap;'):

            # Headline metric box
            ui.html(f'''
            <div class="metric-box">
              <div class="metric-value">{latest_headline:.2f}%</div>
              <div class="metric-label">UK Headline CPIH</div>
              <div class="metric-month">{latest_month}</div>
            </div>
            ''')

            with ui.column().style('gap:10px; flex:1; min-width:200px; justify-content:center;'):
                ui.html(
                    '<p style="margin:0; font-size:0.9rem; color:#505a5f; line-height:1.5;">'
                    'This is the latest published UK CPIH inflation rate. '
                    'Your personal rate may be higher or lower depending on your spending patterns. '
                    'Click <strong>Calculate</strong> to see your estimate.'
                    '</p>'
                )

        ui.separator().style('margin:18px 0 14px;')

        # ── Result card (populated on click) ──────────────────────
        result_container = ui.element('div').classes('result-card')
        with result_container:
            out_forecast = ui.label('Run the calculator to see your personal inflation estimate.') \
                             .classes('result-forecast') \
                             .style('color:#505a5f; font-weight:400;')
            out_proba    = ui.label('').classes('result-proba')
            out_flag     = ui.label('').classes('result-flag')

        def run():
            weights = {k: sliders[k].value for k in CATS}
            X = make_features(latest_row(), normalize_weights(weights))
            p = float(CLF.predict_proba(X)[:, 1][0])
            flag = 'HIGH' if p >= THR else 'LOW'
            y = float(REG.predict(X)[0])

            out_forecast.text = (
                f'Predicted personal inflation rate (3-month): {y:.2f}%'
                f'   \u2014   vs headline {latest_headline:.2f}% now'
            )
            out_proba.text = f'Risk probability: {p:.3f}   (threshold = {THR:.3f})'
            out_flag.text  = f'Inflation risk flag: {flag}'

            out_forecast.classes(replace='result-forecast')
            out_proba.classes(replace='result-proba')

            if flag == 'HIGH':
                result_container.classes(replace='result-card high')
                out_flag.classes(replace='result-flag high')
                out_flag.text = f'⚠️  Inflation risk flag: HIGH'
                ui.notify(
                    f'⚠️ HIGH inflation risk detected — probability {p:.1%}',
                    type='warning', timeout=5000
                )
            else:
                result_container.classes(replace='result-card low')
                out_flag.classes(replace='result-flag low')
                out_flag.text = f'✅  Inflation risk flag: LOW'
                ui.notify(
                    f'Your inflation risk is LOW — probability {p:.1%}',
                    type='positive', timeout=4000
                )

        ui.button('CALCULATE MY INFLATION', on_click=run) \
          .classes('calc-btn') \
          .props('no-caps') \
          .style('margin-top:16px;')

# ── 4. Footer ─────────────────────────────────────────────────────────
ui.html('''
<div class="app-footer">
  <div style="max-width:1100px; margin:0 auto;">
    <div>
      Built by <strong>Nasir Abubakar</strong>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <a href="https://github.com/NasirAbubakar" target="_blank" rel="noopener">GitHub</a>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <a href="https://linkedin.com/in/nasirab" target="_blank" rel="noopener">LinkedIn</a>
    </div>
    <div style="margin-top:4px;">
      Data source:
      <a href="https://www.ons.gov.uk/economy/inflationandpriceindices" target="_blank" rel="noopener">
        Office for National Statistics (ONS) CPIH
      </a>
      &nbsp;&mdash;&nbsp; updated automatically each month.
    </div>
    <div style="margin-top:6px; font-size:0.78rem; opacity:0.65;">
      No personal data is collected or stored.
      Uses live ONS CPIH data via public API.
      For educational and informational purposes only.
    </div>
  </div>
</div>
''')

# ---------------- START APP ---------------- #
ui.run_with(fastapi)
ui.run(host='0.0.0.0', port=int(os.getenv('PORT', '8080')))
