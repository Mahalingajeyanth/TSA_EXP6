# Ex.No: 6               HOLT WINTERS METHOD



### AIM:
To apply the Holt-Winters method for time series forecasting in order to analyze and predict future values by considering level, trend, and seasonality components of the data.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv('/ai_financial_market_daily_realistic_synthetic.csv', parse_dates=['Date'], index_col='Date')

# Preview the data
data.head()
# Resample the data by month (sum the data for the month)
data_monthly = data.resample('MS').sum()  # MS stands for month start frequency

# Preview the resampled data
data_monthly.head()

# Plot the resampled data
data_monthly.plot(figsize=(10, 6))
plt.title('Monthly Resampled Data')
plt.show()
# Step 1: Scale the 'Stock_Impact' column
scaler = MinMaxScaler()

# Ensure we are scaling the 'Stock_Impact' column (you can adjust this for other columns if needed)
scaled_stock_impact = scaler.fit_transform(data['Stock_Impact_%'].values.reshape(-1, 1)).flatten()

# Step 2: Create a time series for the scaled 'Stock_Impact'
scaled_data = pd.Series(scaled_stock_impact, index=data.index)

# Step 3: Plot the scaled data
plt.figure(figsize=(10, 6))
scaled_data.plot()
plt.title('Scaled Stock Impact')
plt.ylabel('Scaled Value')
plt.xlabel('Date')
plt.show()

# Step 4: Decompose the data to check for seasonality
decomposition = seasonal_decompose(scaled_data, model="additive", period=12)  # Adjust 'period' if necessary

# Step 5: Plot the decomposition components (Trend, Seasonality, Residual)
decomposition.plot()
plt.show()
# Split data into train and test (80% train, 20% test)
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Plot to visualize the split
plt.figure(figsize=(10, 6))
train_data.plot(label='Train Data')
test_data.plot(label='Test Data')
plt.legend()
plt.title('Train-Test Split')
plt.show()
# Build the Holt-Winters model (additive trend and additive seasonality)
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast for the test period
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot the predictions and actual data
plt.figure(figsize=(10, 6))
train_data.plot(label='Train Data')
test_data.plot(label='Test Data')
test_predictions_add.plot(label='Predictions', linestyle='--')
plt.legend()
plt.title('Holt-Winters Model Predictions')
plt.show()

# Evaluate the model's performance using RMSE
import numpy as np
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'RMSE: {rmse}')
# Train the final model on the entire dataset
final_model = ExponentialSmoothing(data_monthly['Stock_Impact_%'], trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast for the next 12 months (or more)
final_predictions = final_model.forecast(steps=12)

# Plot the results
plt.figure(figsize=(10, 6))
data_monthly['Stock_Impact_%'].plot(label='Actual Data')
final_predictions.plot(label='Future Predictions', linestyle='--')
plt.legend()
plt.title('Final Model Predictions')
plt.show()
```
### OUTPUT:

Monthly Resampled Data:

<img width="1024" height="595" alt="Screenshot 2025-10-07 142057" src="https://github.com/user-attachments/assets/4af0574c-bd94-4a22-b9fb-30e416a30072" />

Scaled Stock Impact:

<img width="1070" height="537" alt="Screenshot 2025-10-07 142112" src="https://github.com/user-attachments/assets/fa92b20d-e11e-424d-bf62-6e89d865370d" />

<img width="993" height="513" alt="Screenshot 2025-10-07 142122" src="https://github.com/user-attachments/assets/47a26cc8-22de-4264-ac23-982eba84ba77" />

Train-Test Split:

<img width="1105" height="546" alt="image" src="https://github.com/user-attachments/assets/89a8ec18-27eb-4b98-b3ea-f1734b8beab2" />


FINAL_PREDICTION

<img width="1019" height="586" alt="image" src="https://github.com/user-attachments/assets/907bc12d-c50a-49af-8582-76ce441a0996" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
