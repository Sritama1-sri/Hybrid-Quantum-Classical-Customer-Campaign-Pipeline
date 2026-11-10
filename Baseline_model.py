
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("processed_data.csv")

features = [
    "age", "previous_purchases", "website_visits",
    "email_clicks", "spend_score"
]

X = df[features]
y = df["converted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

lr.fit(X_train, y_train)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr.predict(X_test)))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf.predict(X_test)))

lr.fit(X, y)

df["conversion_probability"] = lr.predict_proba(X)[:, 1]

df.to_csv("customer_predictions.csv", index=False)

print(df[["customer_id", "conversion_probability"]])
