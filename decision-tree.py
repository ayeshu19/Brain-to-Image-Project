#Program of Decision tree :
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load your dataset
df = pd.read_csv('/content/sleepdataset.csv')  # Adjust the path as needed

# Separate features and target variable
features = df.columns[df.columns != 'classification']
target = 'classification'

X = df[features]
y = df[target]

# Combine features and target into a single DataFrame
data = pd.concat([X, y], axis=1)

# Split the data into training and testing sets
test_size = 0.2  # Change to 0.3 if you want a larger test set
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# Separate features and target for train and test data
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Initialize and train the Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

classification_report_output = classification_report(y_test, y_pred)
print('Classification Report:')
print(classification_report_output)

confusion_matrix_output = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion_matrix_output)

# Save the trained model (optional)
joblib.dump(model, 'decision_tree_model.pkl')
