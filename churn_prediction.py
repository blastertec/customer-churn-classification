
import pandas as pd
import numpy as np
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import shap
import joblib
import json


# Load the dataset and keep original df for export with predictions
original_data = pd.read_csv("churn_data.csv")
df = original_data.copy()

####################################### DATA PREPROCESSING

# Fill missing plan_type based on most common plan_type per customer_id, 
# although on practice I would define ranges of transaction_amount and fill missing plan_type based on that
most_common_plan = df.groupby('customer_id')['plan_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
df['plan_type'] = df.apply(lambda row: most_common_plan[row['customer_id']] if pd.isna(row['plan_type']) else row['plan_type'], axis=1)

# Fill missing transaction_amount based on average per plan_type and customer_id, or just to ffill from previous row of the same plan_type
df['transaction_amount'] = df['transaction_amount'].fillna(df.groupby(['plan_type', 'customer_id'])['transaction_amount'].transform('mean'))

# If there are still any missing values (e.g., if all grouped values were NaN), fill them with overall mean
df['transaction_amount'] = df['transaction_amount'].fillna(df['transaction_amount'].mean())


####################################### FEATURE ENGINEERING

# Perform one-hot encoding on plan_type
df = pd.get_dummies(df, columns=['plan_type'], prefix='plan')

# Generate issue_date_difference column by calculating difference in months between date and issuing_date
df['date'] = pd.to_datetime(df['date'])
df['issuing_date'] = pd.to_datetime(df['issuing_date'])
df['issue_date_difference'] = (df['date'].dt.year - df['issuing_date'].dt.year) * 12 + (df['date'].dt.month - df['issuing_date'].dt.month)

# Generate months_since_sub column representing months since first datapoint for each client
df['first_date'] = df.groupby('customer_id')['date'].transform('min')
df['months_since_sub'] = (df['date'].dt.year - df['first_date'].dt.year) * 12 + (df['date'].dt.month - df['first_date'].dt.month)
df = df.drop(columns=['first_date'])

# Add month and quarter extracted from date
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter


# Transaction rolling average (expanding window per customer)
df = df.sort_values(['customer_id', 'date'])
df['trans_amount_expanding_avg'] = df.groupby('customer_id')['transaction_amount'].expanding().mean().reset_index(level=0, drop=True)

# Transaction rolling sum (expanding window per customer)
df['trans_amount_expanding_sum'] = df.groupby('customer_id')['transaction_amount'].expanding().sum().reset_index(level=0, drop=True)

# Average transaction amount per date and plan_type
plan_cols = [col for col in df.columns if col.startswith('plan_')]
df['plan_type'] = df[plan_cols].idxmax(axis=1)
date_plan_avg = df.groupby(['date', 'plan_type'])['transaction_amount'].transform('mean')
df['date_plan_avg'] = date_plan_avg

# Ratios
# transaction_amount to expanding average
df['trans_to_expanding_avg_ratio'] = df['transaction_amount'] / df['trans_amount_expanding_avg']
# transaction_amount to date-plan average
df['trans_to_date_plan_avg_ratio'] = df['transaction_amount'] / df['date_plan_avg']
# transaction_amount to expanding sum
df['trans_to_expanding_sum_ratio'] = df['transaction_amount'] / df['trans_amount_expanding_sum']

df = df.drop(columns=['plan_type'])  # drop temporary column


# Add relevant external economic indicators directly for 2023 dates (manually copied values from a source)
economic_indicators = pd.DataFrame({
    'month': range(1, 13),
    'euro600_price': [428, 453, 457, 458, 466, 455, 461, 467, 458, 445, 436, 466],
    'cpi_euro_area': [9.2, 8.6, 8.5, 6.9, 7.0, 6.1, 5.5, 5.3, 5.2, 4.3, 2.9, 2.4],
    'unemployment_euro_area': [6.6, 6.7, 6.6, 6.5, 6.5, 6.5, 6.4, 6.4, 6.4, 6.5, 6.5, 6.4]
})

# Merge these indicators with the main dataframe for 2023 data only
df['year'] = df['date'].dt.year
df = df.merge(economic_indicators, on='month', how='left')
df.loc[df['year'] != 2023, ['euro600_price', 'cpi_euro_area', 'unemployment_euro_area']] = None

# Drop the 'year' column after merge as it is not giving the model any useful information
df = df.drop(columns=['year'])

# Create economic indicators ratio features
df['transaction_euro600_ratio'] = df['transaction_amount'] / df['euro600_price']
df['transaction_cpi_ratio'] = df['transaction_amount'] / df['cpi_euro_area']
df['transaction_unemployment_ratio'] = df['transaction_amount'] / df['unemployment_euro_area']

# Add previous churn feature (lagged churn)
df = df.sort_values(['customer_id', 'date'])
df['previous_churn'] = df.groupby('customer_id')['churn'].shift(1).fillna(0)


####################################### EVAL MODEL TRAINING

# Shift existing churn target to prevent data leakage
df['target_churn'] = df.groupby('customer_id')['churn'].shift(-2)
df = df.dropna(subset=['target_churn'])

# Define features and target
features = [col for col in df.columns if col not in ['customer_id', 'date', 'issuing_date', 'churn', 'target_churn']]

# Validation split using the last two datapoints per customer (November and December 2023)
val_df = df.groupby('customer_id').tail(2)
train_df = df.drop(val_df.index)

X_train, y_train = train_df[features], train_df['target_churn']
X_val, y_val = val_df[features], val_df['target_churn']

# Model training
val_model = LogisticRegression(max_iter=1000)
val_model.fit(X_train, y_train)

# Validation predictions
y_pred = val_model.predict(X_val)

####################################### MODEL EVALUATION

# Save metrics to a JSON file
metrics = {
    'Precision': precision_score(y_val, y_pred),
    'Recall': recall_score(y_val, y_pred),
    'Accuracy': accuracy_score(y_val, y_pred),
    'F1': f1_score(y_val, y_pred)
}

print(metrics)

# Save metrics to a JSON file
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)


####################################### MODEL EXPORT

# Train a new model on full dataset before export
model = LogisticRegression(max_iter=1000)
model.fit(df[features], df['target_churn'])

# Save the model
joblib.dump(model, 'churn_forecast_model.pkl')


####################################### OUT OF SAMPLE PREDICTIONS AND EXPORT OF CSV

# Forecast for Jan-Feb 2024 and export results with original data 
# First, fix the dates for predictions (shift forward 2 months)
val_df = val_df.copy()
val_df['prediction_date'] = val_df['date'] + pd.DateOffset(months=2)
val_df['predicted_churn'] = y_pred

# Merge corrected predictions back into original_data
original_data['date'] = pd.to_datetime(original_data['date'])
val_predictions = val_df[['customer_id', 'prediction_date', 'predicted_churn']].rename(columns={'prediction_date': 'date'})
original_data = original_data.merge(val_predictions, on=['customer_id', 'date'], how='left')


# Prepare dataframe for forecasting (Jan & Feb 2024)
future_dates = pd.date_range('2024-01-01', '2024-02-01', freq='MS')
customer_ids = original_data['customer_id'].unique()
forecast_rows = pd.MultiIndex.from_product([customer_ids, future_dates], names=['customer_id', 'date']).to_frame(index=False)

# Merge additional columns as NaN
forecast_df = forecast_rows.merge(original_data[['customer_id']].drop_duplicates(), on='customer_id', how='left')
for col in original_data.columns:
    if col not in ['customer_id', 'date', 'predicted_churn']:
        forecast_df[col] = np.nan

# Perform feature engineering for forecast rows
forecast_features_df = pd.concat([df, forecast_df], ignore_index=True, sort=False)

# Fill missing engineered features with sensible defaults or NaNs as necessary
forecast_features_df = forecast_features_df.sort_values(['customer_id', 'date'])
forecast_features_df[features] = forecast_features_df[features].fillna(method='ffill').fillna(method='bfill').fillna(0)

# Select forecast rows
forecast_only_df = forecast_features_df[forecast_features_df['date'].isin(future_dates)]

# Make forecasts for Jan 2024 and Feb 2024
forecast_only_df['predicted_churn'] = model.predict(forecast_only_df[features])

# Append these forecast rows to original data
forecast_export = forecast_only_df[['customer_id', 'date', 'predicted_churn']]
final_export_df = pd.concat([original_data, forecast_export], ignore_index=True, sort=False)

# Export original dataframe with predictions for Validation period and Jan-Feb 2024
final_export_df.sort_values(['customer_id', 'date']).to_csv('churn_data_preds.csv', index=False)


####################################### SHAP model explanation

X_shap = df[features].fillna(0).replace([np.inf, -np.inf], 0).astype(float).values
explainer = shap.LinearExplainer(model, X_shap)
shap_values = explainer.shap_values(X_shap)

# Plot SHAP summary
shap.summary_plot(shap_values, features=X_shap, feature_names=features)
