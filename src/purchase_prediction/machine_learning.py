import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned data
cleaned_file_name = "cleaned_purchase_data.csv"
data = pd.read_csv(cleaned_file_name)

# Split the data into features and target
X = data[['age', 'location', 'page_name', 'time_on_page']]
y = data['purchased_boolean']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict the target variable
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = (y_pred == y_test).mean()
print(f"Model Accuracy: {accuracy}")

# Save the trained model
model_file = "trained_model.joblib"
joblib.dump(model, model_file)
