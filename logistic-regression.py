#Program for Logistic Regression:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load your dataset
df = pd.read_csv('/content/sleepdataset.csv')  # Adjusted path for Colab

# Step 2: Preprocess the data
# Selecting features and target variable
features = df.columns[df.columns != 'classification']
target = 'classification'

X = df[features]
y = df[target]

# Step 3: Split the dataset into training and testing sets
# Adjust the test_size as needed (0.2 or 0.3)
test_size = 0.2  # Change to 0.3 if you want a larger test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Step 4: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Step 5: Make predictions on the testing set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

classification_report_output = classification_report(y_test, y_pred)
print('Classification Report:')
print(classification_report_output)

confusion_matrix_output = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion_matrix_output)

# Step 7: Save the trained model (optional)
joblib.dump(model, 'logistic_regression_model.pkl')
