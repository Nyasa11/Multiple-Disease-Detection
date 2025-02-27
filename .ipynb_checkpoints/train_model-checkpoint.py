import panda as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('diabetes-dataset.csv')

# Preprocess the data (example for Pima Indians Diabetes Dataset)
X = data.drop(columns=['Outcome'])  # Features
y = data['Outcome']                 # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
