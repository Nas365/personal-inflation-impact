# Personal Inflation Impact (UK)

An end-to-end machine learning application that estimates an individual’s personal inflation rate and compares it with the national average (UK CPIH).
This project combines Economics and Data Science to make inflation data personal, interactive, and actionable.

## Overview

Following my previous projects on London House Price Prediction and Loan Default Prediction, this project examines the next logical phase of financial decision-making — managing the cost of living.

After purchasing a house and obtaining a loan, the next question for any household is:
“How do I manage my living expenses and stay financially stable amid rising inflation?”

The Personal Inflation Impact (UK) app answers that question by allowing users to simulate their inflation exposure based on their spending habits and compare it with the official CPIH rate published by the Office for National Statistics (ONS).

## Motivation

Inflation is often reported as a national average, but in reality, every household experiences it differently depending on their spending mix.
For example, someone who spends more on transport and housing will experience inflation differently from someone whose major expenses are health and recreation.

This project was developed to quantify that difference, helping individuals visualize how their personal inflation compares with the headline CPIH and whether they are at high risk or low risk financially.

## Data Source

Source: Office for National Statistics (ONS) – CPIH Divisional Index (UK)

Covers core CPIH categories: Food, Housing, Transport, Health, Recreation, Miscellaneous

Updated monthly and used as the basis for all predictions

Processed into a tidy CSV (data/cpih_wide_yoy.csv) for training and inference

## Methodology

The system uses two integrated models:

Regression Model – forecasts the user’s 3-month forward personal inflation rate

Classification Model – predicts whether the forecasted rate exceeds the headline CPIH by 2 percentage points or more, classifying it as High Risk or Low Risk

## Technical Architecture

Language: Python

Libraries: pandas, scikit-learn, LightGBM

Frameworks: FastAPI (backend), NiceGUI (frontend)

Version Control: Git & GitHub

Automation: GitHub Actions (CI/CD)

Hosting: Render

Artifacts: Stored under artifacts/ as serialized joblib models

Data File: data/cpih_wide_yoy.csv

## Why Compare Personal Inflation?

Comparing your personal inflation rate with the national CPIH provides insight into how your household is positioned relative to overall economic trends.
If your personal inflation is consistently above the national average, it suggests that your spending mix exposes you to faster-rising costs — prompting adjustments to your budget or savings strategy.

This approach bridges macroeconomic data and personal finance, empowering individuals to plan and act based on evidence rather than averages.

## Deployment

Deployed on Render with unified FastAPI + NiceGUI interface

Every push to the main branch triggers a GitHub Actions workflow that rebuilds and redeploys automatically

No Docker or AWS configuration is required in this version — deployment is streamlined for simplicity

## Live Demo

Live App: [https://personal-inflation-impact.onrender.com](https://personal-inflation-impact.onrender.com)

GitHub Repository: https://github.com/Nas365/personal-inflation-impact
