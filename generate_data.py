from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(20261004)
n = 2400

age = rng.integers(21, 66, n)
annual_income = np.clip(rng.lognormal(mean=np.log(900000), sigma=0.42, size=n), 250000, 6000000)
employment_years = np.clip((age - 20) * rng.uniform(0.25, 0.75, n), 0, 35)
loan_amount = np.clip(rng.lognormal(mean=np.log(2200000), sigma=0.65, size=n), 100000, 15000000)
loan_term_months = rng.choice([24, 36, 48, 60, 72, 84], n, p=[0.07, 0.17, 0.18, 0.32, 0.18, 0.08])
credit_score = np.clip(rng.normal(700, 75, n), 300, 850).round().astype(int)
debt_to_income = np.clip(rng.normal(30, 12, n), 2, 75)
existing_loans = np.clip(rng.poisson(1.3, n), 0, 8)
missed_payments_12m = np.clip(rng.poisson(0.65, n), 0, 8)
savings_balance = np.clip(annual_income * rng.uniform(0.03, 0.65, n), 10000, 5000000)
employment_type = rng.choice(["Salaried", "Self-employed", "Contract"], n, p=[0.64, 0.26, 0.10])
housing_status = rng.choice(["Owned", "Rented", "Other"], n, p=[0.45, 0.48, 0.07])
dependents = np.clip(rng.poisson(1.4, n), 0, 6)

# Transparent synthetic outcome mechanism designed only for classroom demonstration.
score = (
    -0.2
    + 0.006 * (credit_score - 680)
    - 0.045 * (debt_to_income - 30)
    - 0.42 * missed_payments_12m
    + 0.12 * employment_years
    - 0.00000008 * loan_amount
    + 0.00000016 * annual_income
    + 0.00000015 * savings_balance
    - 0.16 * existing_loans
    - 0.04 * dependents
    + 0.18 * (employment_type == "Salaried")
    + 0.12 * (housing_status == "Owned")
)
prob = 1 / (1 + np.exp(-score))
approved = rng.binomial(1, prob)

out = pd.DataFrame({
    "age": age,
    "annual_income": annual_income.round(0).astype(int),
    "employment_years": employment_years.round(1),
    "loan_amount": loan_amount.round(0).astype(int),
    "loan_term_months": loan_term_months,
    "credit_score": credit_score,
    "debt_to_income": debt_to_income.round(1),
    "existing_loans": existing_loans,
    "missed_payments_12m": missed_payments_12m,
    "savings_balance": savings_balance.round(0).astype(int),
    "employment_type": employment_type,
    "housing_status": housing_status,
    "dependents": dependents,
    "approved": approved,
})

path = Path(__file__).resolve().parent / "sample_credit_applications.csv"
out.to_csv(path, index=False)
print(f"wrote {path} with {len(out)} rows")
