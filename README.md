# CreditWise AI — End Term Project

## Selected use case
**Loan eligibility / credit-risk pre-screener (App + AI Copilot)**

This use case was selected from the supplied project menu because it combines a clear finance decision, structured data, interpretable machine learning, explainability, what-if analysis and a generative-AI layer.

## What the app does
1. Takes a structured synthetic applicant profile.
2. Runs an interpretable Logistic Regression model trained on reproducible synthetic data.
3. Produces an estimated approval probability and three-way screening bucket.
4. Applies a separate rule layer for transparent guardrails such as high DTI, low credit score, missed payments and low savings buffer.
5. Generates a plain-English explanation using Gemini when an API key is configured.
6. Falls back automatically to a local explanation if Gemini is unavailable.
7. Includes an executive overview, applicant screening workspace, interactive What-If Lab, portfolio analytics and an AI Copilot.
8. Provides model-performance metrics, risk-mix charts, a local contribution chart, data dictionary and production roadmap.
9. Allows the synthetic sample dataset to be downloaded from the Overview tab.

## Refreshed UI
The latest version is designed as a polished fintech-style presentation app rather than a plain Streamlit form. Navigation is now a high-contrast, always-visible control bar instead of hard-to-see tabs, and the Overview page includes a live What-If simulator so the key interactive feature is visible immediately. It includes:
- Executive KPI cards and model-health gauge
- End-to-end workflow visualisation
- Risk framework cards for low / medium / high screening outcomes
- Applicant decision-support dashboard
- Probability gauge and feature-contribution chart
- What-If sensitivity analysis with baseline vs. scenario comparison
- Portfolio risk-mix and affordability analytics
- Chat-style AI Copilot
- Data dictionary and production roadmap

## Run locally
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## Optional Gemini setup
Create a Gemini API key using Google AI Studio and configure it as either:

- environment variable: `GEMINI_API_KEY=...`
- `.streamlit/secrets.toml` using `.streamlit/secrets.toml.example`

Optional model override:
```powershell
$env:GEMINI_MODEL="gemini-3.8-flash"
```

The project remains fully demonstrable without Gemini because the local scoring, rule layer, local explanation and Copilot fallback are built in.

## Deploy for a shareable link
1. Create a GitHub repository and upload the contents of this folder.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Add `GEMINI_API_KEY` under deployment Secrets if you want Gemini enabled online.
4. Use the generated `https://...streamlit.app` link in the end-term submission.

Do not commit a real API key to GitHub.

## Suggested demo flow
1. Open **Overview**, use the visible **Live Demo Control Room** sliders, and explain the business problem and hybrid architecture.
2. Go to **Applicant Screening** → choose **Demo — Strong profile** → click **Run AI pre-screen**.
3. Explain the probability, risk bucket, guardrails and local contribution chart.
4. Open **What-If Lab** and reduce the credit score / increase DTI to demonstrate sensitivity.
5. Open **Portfolio Intelligence** and show risk mix, affordability pressure and ROC-AUC.
6. Open **AI Copilot** and ask: `Why does DTI matter in the screening score?`
7. Use **Project Guide** for limitations, data dictionary and production roadmap.

## Validation
The bundled tests should return:
```text
2 passed
```

The current synthetic-data holdout validation is approximately:
- Accuracy: **72.2%**
- ROC-AUC: **0.781**

These metrics are for the classroom prototype only and are not evidence of real-world lending performance.

## Important academic limitation
All applicant data are synthetic. The model and thresholds must not be presented as a production lending or financial-advice system. A production implementation would require verified data, documented policy thresholds, fairness testing, adverse-action reason codes, privacy/security controls, audit trails, model monitoring, drift detection and human oversight.

## Viva-ready one-liner
**CreditWise AI is a hybrid decision-support app: a transparent ML model estimates credit-screening risk, rule checks provide deterministic guardrails, and Gemini converts the result into a human-readable explanation without being allowed to make the final lending decision.**

## Streamlit Community Cloud entrypoint
Use `app.py` as the Community Cloud entrypoint. Do not use `create_report.py`; the report is already provided in `docs/` and is not part of the web application runtime.

Recommended GitHub repository root:
- `app.py`
- `requirements.txt`
- `data/`
- `.streamlit/`
- `tests/`

For Community Cloud, set the Main file / entrypoint path to `app.py`.
Add `GEMINI_API_KEY` in the app's Secrets settings. Never commit the real API key.
