
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

# Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Churn probability

#evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))

#save for powerbi
results = pd.DataFrame({
    "CustomerID": df.loc[y_test.index, "customerID"],
    "ChurnProbability": y_prob,
    "PredictedChurn": y_pred
})

results.to_csv("churn_predictions.csv", index=False)
print("Predictions saved to churn_predictions.csv")


# Optional: Show top 10 high-risk customers
top_risk = results.sort_values(by='ChurnProbability', ascending=False).head(10)
print("\nTop 10 High-Risk Customers:\n", top_risk)
