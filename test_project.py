import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "data/sample_credit_applications.csv"
F = ["age","annual_income","employment_years","loan_amount","loan_term_months","credit_score","debt_to_income","existing_loans","missed_payments_12m","savings_balance","employment_type","housing_status","dependents"]
N = ["age","annual_income","employment_years","loan_amount","loan_term_months","credit_score","debt_to_income","existing_loans","missed_payments_12m","savings_balance","dependents"]
C = ["employment_type","housing_status"]

def test_dataset():
    df = pd.read_csv(DATA)
    assert len(df) == 2400
    assert set(F + ["approved"]).issubset(df.columns)
    assert df["approved"].isin([0, 1]).all()

def test_model_quality_and_demos():
    df = pd.read_csv(DATA)
    X_train, X_test, y_train, y_test = train_test_split(df[F], df["approved"], test_size=0.25, random_state=42, stratify=df["approved"])
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), N),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), C),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1500, random_state=42))])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    assert 0.65 <= accuracy_score(y_test, pred) <= 0.85
    assert 0.70 <= roc_auc_score(y_test, prob) <= 0.90
    strong = {
        "age":34,"annual_income":1500000,"employment_years":8,"loan_amount":2500000,"loan_term_months":60,
        "credit_score":790,"debt_to_income":22.0,"existing_loans":1,"missed_payments_12m":0,"savings_balance":700000,
        "employment_type":"Salaried","housing_status":"Owned","dependents":1}
    high = {
        "age":28,"annual_income":550000,"employment_years":2,"loan_amount":3500000,"loan_term_months":48,
        "credit_score":545,"debt_to_income":58.0,"existing_loans":4,"missed_payments_12m":4,"savings_balance":50000,
        "employment_type":"Self-employed","housing_status":"Rented","dependents":3}
    p1 = model.predict_proba(pd.DataFrame([strong])[F])[0,1]
    p2 = model.predict_proba(pd.DataFrame([high])[F])[0,1]
    assert p1 > 0.70
    assert p2 < 0.20
