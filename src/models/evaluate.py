import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

test_data_path = "../../data/processed/test.csv"
df_test = pd.read_csv(test_data_path)

model_path = "../../model_store/trained_model.pkl"
model = joblib.load(model_path)

X_test = df_test.drop(columns=['is_healthy'])
y_test = df_test['is_healthy']

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

report = classification_report(y_test, y_pred)
print("Model Evaluation Report:\n", report)

report_path = "../../reports/model_performance.md"
with open(report_path, "w") as f:
    f.write("# Model Evaluation Report\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write("\n\n")
    f.write(report)

