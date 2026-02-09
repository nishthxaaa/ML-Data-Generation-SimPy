import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# STEP 1 & 2: Define the Simulation (Bank Queue System)
# ---------------------------------------------------------
def bank_simulation(env, num_tellers, arrival_rate, service_time_avg):
    """
    Simulates a bank where customers arrive and wait for tellers.
    Returns the average waiting time for that day.
    """
    teller = simpy.Resource(env, capacity=num_tellers)
    wait_times = []

    def customer(env, name, teller):
        arrival_time = env.now
        with teller.request() as request:
            yield request
            wait = env.now - arrival_time
            wait_times.append(wait)
            # Service time is exponential
            yield env.timeout(random.expovariate(1.0 / service_time_avg))

    def setup(env, num_tellers, arrival_rate, service_time_avg):
        i = 0
        while True:
            yield env.timeout(random.expovariate(arrival_rate))
            i += 1
            env.process(customer(env, f'Customer {i}', teller))

    env.process(setup(env, num_tellers, arrival_rate, service_time_avg))
    env.run(until=480) # Simulate an 8-hour work day (480 minutes)
    
    if len(wait_times) > 0:
        return sum(wait_times) / len(wait_times)
    else:
        return 0

# ---------------------------------------------------------
# STEP 3, 4 & 5: Generate 1000 Simulations
# ---------------------------------------------------------
print("Generating 1000 simulations... This may take a moment.")

data = []
# Define parameter bounds
# Tellers: 1 to 5
# Arrival Rate: 0.1 to 1.0 customers per minute
# Service Time: 3 to 15 minutes average
for _ in range(1000):
    num_tellers = random.randint(1, 5)
    arrival_rate = random.uniform(0.1, 1.0)
    service_time = random.uniform(3.0, 15.0)
    
    env = simpy.Environment()
    avg_wait = bank_simulation(env, num_tellers, arrival_rate, service_time)
    
    data.append([num_tellers, arrival_rate, service_time, avg_wait])

# Save to DataFrame
df = pd.DataFrame(data, columns=['Num_Tellers', 'Arrival_Rate', 'Service_Time_Avg', 'Avg_Wait_Time'])
df.to_csv('generated_dataset.csv', index=False)
print("Dataset generated and saved to 'generated_dataset.csv'.")

# ---------------------------------------------------------
# STEP 6: Compare ML Models
# ---------------------------------------------------------
X = df[['Num_Tellers', 'Arrival_Rate', 'Service_Time_Avg']]
y = df['Avg_Wait_Time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Regressor": SVR()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append([name, mse, r2])

# Create Comparison Table
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2_Score'])
results_df = results_df.sort_values(by='R2_Score', ascending=False)

print("\nModel Comparison Results:")
print(results_df)
results_df.to_csv('model_results.csv', index=False)

# Generate Graph
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['R2_Score'], color='teal')
plt.xlabel('R2 Score (Accuracy)')
plt.title('Comparison of ML Models on Simulated Data')
plt.xlim(0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("Graph saved to 'model_comparison.png'.")