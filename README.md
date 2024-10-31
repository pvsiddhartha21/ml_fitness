# ml_fitness
here's a simple fitness plan recommender based on multiple regression which recommends workout duration and type of strength training including its reps based on the users height , weight and fitness goals
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('fitness_data.csv')

# Basic preprocessing
X = df[['weight', 'height', 'age', 'fitness_goal']]  # Features
y = df['workout_time']  # Target variable

# Convert categorical data (like fitness_goal) into numerical values
X = pd.get_dummies(X, drop_first=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial transformation (degree 2 for demonstration)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit a polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict on the test set
y_pred = model.predict(X_test_poly)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualizing the actual vs predicted workout time based on weight
plt.figure(figsize=(10, 6))
plt.scatter(X_test['weight'], y_test, color='blue', label='Actual data')
plt.scatter(X_test['weight'], y_pred, color='red', label='Predictions')
plt.plot(X_test['weight'], y_pred, color='green', label='Best Fit Line', linewidth=2)
plt.xlabel('Weight (kg)')
plt.ylabel('Workout Time (minutes)')
plt.title('Actual vs Predicted Workout Time (based on Weight)')
plt.legend()
plt.grid()
plt.show()

# Fitness goals and corresponding workouts
fitness_goals = {
    "muscle_gain": [
        "Weightlifting (3-5 sets of 8-12 reps)",
        "Bodyweight exercises (push-ups, squats)",
        "Compound movements (deadlifts, bench press)"
    ],
    "weight_loss": [
        "HIIT workouts (30 mins)",
        "Cardio (running, cycling, swimming)",
        "Circuit training (full-body workout)"
    ],
    "endurance": [
        "Long-distance running or cycling",
        "Swimming (freestyle or laps)",
        "Rowing for cardio"
    ],
    "flexibility": [
        "Yoga sessions (Hatha or Vinyasa)",
        "Pilates classes",
        "Static stretching routines"
    ],
    "overall_fitness": [
        "Mixed workout routines (strength + cardio)",
        "Group fitness classes (Zumba, spinning)",
        "Outdoor activities (hiking, sports)"
    ]
}

# Function to recommend workouts based on user input
def recommend_workout(weight, height, age, fitness_goal):
    # Prepare the input data
    input_data = pd.DataFrame({
        'weight': [weight],
        'height': [height],
        'age': [age],
        'fitness_goal': [fitness_goal]  # Assume this is pre-processed or encoded
    })

    # Ensure input_data has the same dummy columns as training data
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Align columns with the training set
    missing_cols = set(X_train.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0  # Add missing columns with default value 0

    # Ensure the columns are in the same order
    input_data = input_data[X_train.columns]

    # Transform the input data using the same polynomial transformer
    input_poly = poly.transform(input_data)

    # Predict the workout time
    predicted_workout_time = model.predict(input_poly)[0]

    # Recommend workout, diet, and advice based on fitness goal
    workout_options = fitness_goals.get(fitness_goal, ["No specific recommendations available."])

    if predicted_workout_time < 30:
        advice = "Focus on light, enjoyable activities."
    elif predicted_workout_time < 60:
        advice = "Aim for a balanced routine that includes moderate effort."
    else:
        advice = "Incorporate intense training for best results."

    return {
        'predicted_workout_time': predicted_workout_time,
        'workout_options': workout_options,
        'advice': advice
    }

# Get user inputs
weight = float(input("Enter your weight (kg): "))
height = float(input("Enter your height (cm): "))
age = int(input("Enter your age: "))

# Display fitness goal options
print("\nAvailable fitness goals:")
for goal in fitness_goals.keys():
    print(f"- {goal}")

fitness_goal = input("Enter your fitness goal from the options above: ")

# Example usage
recommendation = recommend_workout(weight, height, age, fitness_goal)

# Print recommendations with new lines
print("\nRecommendations:")
print(f"Predicted Workout Time: {recommendation['predicted_workout_time']:.2f} minutes")
print("Recommended Workouts:")
for workout in recommendation['workout_options']:
    print(f"- {workout}")
print(f"Advice: {recommendation['advice']}")

