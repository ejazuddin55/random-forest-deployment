import joblib

# Load model
model = joblib.load("models/model.pkl")

# View model details
print(model)
print(model.get_params())
import joblib
import matplotlib.pyplot as plt

# Load model from pickle file
model = joblib.load("models/model.pkl")

# Access feature importances
importances = model.feature_importances_

# Plot
plt.bar(range(len(importances)), importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()
