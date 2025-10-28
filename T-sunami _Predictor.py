# disaster_prediction_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load the dataset
data = pd.read_csv("earthquake_data_tsunami.csv")

# Display dataset info
print("Dataset Shape:", data.shape)
print(data.head())

#Identify features and target
# Replace 'tsunami' with the actual target column name if different
target_column = 'tsunami'  
X = data.drop(columns=[target_column])
y = data[target_column]

#Handle missing values
X = X.fillna(X.mean())

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Predict disaster occurrence for a new input
# Example: use average feature values for demonstration
sample = X.mean().values.reshape(1, -1)
prediction = model.predict(sample)
print("\n Predicted Disaster Occurrence:", prediction[0])

#Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disaster', 'Disaster'], yticklabels=['No Disaster', 'Disaster'])
plt.title("Confusion Matrix - Disaster Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#Feature Importance Visualization
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
plt.title("💡 Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()