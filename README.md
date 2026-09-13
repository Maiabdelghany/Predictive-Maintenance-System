# Machine Health Monitoring — Streamlit Deployment

## Files in this folder
- `app.py` — the dashboard code
- `requirements.txt` — required libraries
- The 5 `.pkl` files — the trained models (keep them as they are)

## Steps to deploy on Streamlit Community Cloud (free and permanent)

### 1. Create a GitHub account (if you don't have one)
Go to github.com and sign up for a free account.

### 2. Create a new Repository
- From GitHub, click "New repository"
- Name it something like: `machine-health-monitoring`
- Set it to Public
- Upload all the files in this folder (the 7 files) — via "Add file" → "Upload files"

### 3. Sign in to Streamlit Community Cloud
Go to share.streamlit.io and sign in with the same GitHub account (completely free).

### 4. Deploy
- Click "New app"
- Select the repository you created
- In the "Main file path" field, type: `app.py`
- Click "Deploy"

### 5. Wait a minute or two
A permanent link will be ready, something like:
`https://machine-health-monitoring-xxxx.streamlit.app`

This is your link to put in your portfolio/CV — it runs 24/7 without ever having to open Colab again.

## Important note about prediction accuracy
The models were also trained on "rolling mean/std" and "rate of change" features for each sensor (which require a history of readings, not just one reading). This dashboard only takes a single reading and defaults the rolling/rate features to 0 — that's fine for demos and testing, but for higher accuracy you'd need to modify the code to accept a series of readings (history) instead of a single reading.
