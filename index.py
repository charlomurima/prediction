import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the road accident dataset
df = pd.read_csv("road_accident_dataset.csv")

# Split the dataset into training and test sets, stratifying by accident severity
X_train, X_test, y_train, y_test = train_test_split(
    df, df["accident_severity"], test_size=0.25, random_state=42, stratify=df["accident_severity"]
)

# Extract the independent and dependent variables
X_train = X_train[["driver_age", "driver_gender", "vehicle_type", "road_type", "weather_conditions", "time_of_day", "location_of_accident"]]
y_train = y_train
X_test = X_test[["driver_age", "driver_gender", "vehicle_type", "road_type", "weather_conditions", "time_of_day", "location_of_accident"]]
y_test = y_test

# Create the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the fitted model
joblib.dump(model, "linear_regression_model.pkl")

# Load the saved model
model = joblib.load("linear_regression_model.pkl")

# Predict the accident severity for a hypothetical set of independent variables
driver_age = 25
driver_gender = "Male"
vehicle_type = "Car"
road_type = "Highway"
weather_conditions = "Sunny"
time_of_day = "Daytime"
location_of_accident = "Urban"

# Create a NumPy array of the independent variables
X_test_new = np.array([[driver_age, driver_gender, vehicle_type, road_type, weather_conditions, time_of_day, location_of_accident]])

# Make a prediction of accident severity
accident_severity_prediction = model.predict(X_test_new)

# Print the predicted accident severity
print(accident_severity_prediction)
