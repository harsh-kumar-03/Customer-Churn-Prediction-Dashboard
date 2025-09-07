
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_excel("Telco-Customer-Churn.xlsx")  

# data cleaning
# convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# fill missing values
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# feature Encoding
# convert categorical variables to dummies
df_encoded = pd.get_dummies(df, drop_first=True)

#separate features and target
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#train randomforestmodel
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

#evaluation
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("AUC Score:", roc_auc_score(y_test, y_prob_test))

#prediction for all 100%
y_pred_all = model.predict(X)   # predictions for all 7043
y_prob_all = model.predict_proba(X)[:, 1]

# Save results for Power BI
results = pd.DataFrame({
    "CustomerID": df["customerID"],
    "ChurnProbability": y_prob_all,
    "PredictedChurn": y_pred_all
})

results.to_csv("churn_predictions.csv", index=False)
print("✅ Predictions saved to churn_predictions.csv (all 7043 customers)")

#optional
top_risk = results.sort_values(by='ChurnProbability', ascending=False).head(10)
print("\nTop 10 High-Risk Customers:\n", top_risk)
