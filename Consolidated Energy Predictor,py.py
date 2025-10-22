import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest

# --- Configuration & Constants ---
# If your data is available, replace this with your file path:
DATA_PATH = 'building_energy_data.csv'  
TARGET_COLUMN = 'kwh_consumption'
DATE_COLUMN = 'timestamp'
TEST_SIZE_FRACTION = 0.10 # Use the last 10% of the data for testing (time-series split)
ANOMALY_CONTAMINATION = 0.01 # Expect 1% of data points to be anomalies

# ==============================================================================
# PART 1: DATA PREPARATION (Consolidated from data_prep.py)
# ==============================================================================

def generate_mock_data() -> pd.DataFrame:
    """Generates synthetic time-series data for prototyping purposes."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    # Using 'h' instead of 'H' to avoid FutureWarning
    date_range = pd.date_range(start=start_date, end=end_date, freq='h') 
    
    # Simulate consumption: base load + seasonality + noise
    consumption = (
        150 +                                     # Base load
        50 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 365)) + # Yearly seasonality
        20 * np.sin(2 * np.pi * np.arange(len(date_range)) / 24) +        # Daily seasonality
        np.random.normal(0, 5, len(date_range))   # Random noise
    )
    consumption[consumption < 0] = 0 # Ensure no negative consumption
    
    df = pd.DataFrame({
        TARGET_COLUMN: consumption
    }, index=date_range)
    df.index.name = DATE_COLUMN
    
    # Add external factors (conceptual feature engineering)
    df['temperature_c'] = 20 + 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 365)) + np.random.normal(0, 3, len(date_range))
    df['is_weekday'] = (df.index.dayofweek < 5).astype(int)
    df['hour_of_day'] = df.index.hour
    
    # Inject a few strong outliers (wastage patterns) for the anomaly detector to find
    outlier_indices = np.random.choice(len(df), size=8, replace=False)
    df.loc[df.index[outlier_indices], TARGET_COLUMN] += 150 
    
    return df

def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Attempts to load data from the specified path, using mock data as fallback.
    """
    try:
        # NOTE: For this consolidated script, we assume the user is running it without the separate files.
        # We will attempt to use the specific path identified in previous user uploads for the real file.
        # If your file is not there, it will fall back to mock data.
        user_specific_path = r'C:\Users\Tilak\Downloads\EnergyConsumptionAnalyzer\EnergyConsumptionAnalyzer\data\data_energy.csv'

        if pd.io.common.file_exists(user_specific_path):
            print(f"Loading data from specific user path: {user_specific_path}")
            df = pd.read_csv(user_specific_path)
        else:
            print(f"ERROR: Data file not found at {user_specific_path} or {file_path}. Generating mock data.")
            return generate_mock_data()
            
        # CRITICAL FIX: Strip whitespace from all column names (removes hidden spaces)
        df.columns = df.columns.str.strip() 

        # Ensure 'timestamp' is datetime and numerical columns are numeric based on deposit_reporting.py
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
            df.set_index('timestamp', inplace=True)
            df.index.name = DATE_COLUMN
        
        # Based on your previous file's structure, rename columns if needed
        # We assume 'power_kW' in the real file corresponds to our TARGET_COLUMN (kwh_consumption)
        if 'power_kW' in df.columns:
            df.rename(columns={'power_kW': TARGET_COLUMN}, inplace=True)
        if 'temperature_C' in df.columns and 'temperature_c' not in df.columns:
             df.rename(columns={'temperature_C': 'temperature_c'}, inplace=True)
        
        # Ensure target column exists after all renaming
        if TARGET_COLUMN not in df.columns:
             raise KeyError(f"Target column '{TARGET_COLUMN}' (or 'power_kW') not found after loading.")

        print(f"Data loaded successfully. Initial shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"ERROR: Data file not found. Generating mock data.")
        return generate_mock_data()
    except Exception as e:
        print(f"An error occurred during data loading: {e}. Generating mock data.")
        return generate_mock_data()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values, outliers, and ensures data integrity."""
    print(f"Starting data cleaning. Shape: {df.shape}")
    
    # 1. Handle Missing Values (Imputation - simple example)
    df.fillna(df.mean(numeric_only=True), inplace=True)
        
    # 2. Outlier Removal (Simple Z-Score on Target)
    if TARGET_COLUMN in df.columns:
        z_scores = (df[TARGET_COLUMN] - df[TARGET_COLUMN].mean()) / df[TARGET_COLUMN].std()
        df.loc[np.abs(z_scores) > 3, TARGET_COLUMN] = df[TARGET_COLUMN].median()

    print(f"Data cleaning complete. New shape: {df.shape}")
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-series features essential for robust energy prediction.
    """
    # Ensure necessary columns for feature engineering are present
    if TARGET_COLUMN not in df.columns or 'temperature_c' not in df.columns:
        print("Required columns for feature engineering missing. Using only time features.")
    
    # Time-Based Features 
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_holiday'] = ((df.index.month == 1) & (df.index.day == 1)).astype(int) # Simple holiday proxy
    
    # Lagged Features (Previous period's consumption is the strongest predictor)
    df['lag_1h'] = df[TARGET_COLUMN].shift(1)
    df['lag_24h'] = df[TARGET_COLUMN].shift(24)
    df['lag_168h'] = df[TARGET_COLUMN].shift(168)
    
    # Rolling/Window Features (Smoothing out noise and capturing trend)
    df['rolling_mean_4h'] = df[TARGET_COLUMN].shift(1).rolling(window=4).mean()
    df['rolling_std_24h'] = df[TARGET_COLUMN].shift(1).rolling(window=24).std()

    # Drop rows with NaN values created by lagging/rolling operations
    df.dropna(inplace=True)
    
    return df

def prepare_data(file_path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Runs the entire data preparation pipeline."""
    # 1. Load Data
    df = load_data(file_path)
    if df.empty or TARGET_COLUMN not in df.columns:
        return pd.DataFrame(), pd.Series()

    # 2. Clean Data
    df = clean_data(df)

    # 3. Feature Engineer
    df = feature_engineer(df)
    
    # Separate features (X) and target (y)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    
    print("\nData preparation complete!")
    return X, y

# ==============================================================================
# PART 2: MODEL TRAINING (Consolidated from model_training.py)
# ==============================================================================

def split_data_time_series(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE_FRACTION) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits time-series data ensuring the test set is chronologically after the train set.
    """
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"\n--- Data Split Summary ---")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)} ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"Testing samples: {len(X_test)} ({X_test.index.min().date()} to {X_test.index.max().date()})")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Initializes and trains the XGBoost Regressor model."""
    print("\nStarting XGBoost Model Training...")
    
    # Initialize the model with reasonable default parameters
    model = XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("XGBoost training complete.")
    return model

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray):
    """Calculates and prints key regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kWh")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kWh")
    print(f"R-squared (R²): {r2:.4f}")
    
    if r2 > 0.8:
        print("Model performance is strong! High accuracy for forecasting.")
    else:
        print("Model performance may need tuning or more sophisticated features.")

# ==============================================================================
# PART 3: ANOMALY DETECTION & RECOMMENDATIONS (Consolidated from anomaly_detector.py)
# ==============================================================================

def detect_anomalies(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Identifies energy wastage (anomalies) based on prediction error (residual).
    """
    print("\nStarting Anomaly Detection (Wastage Pattern Identification)...")
    
    # 1. Predict and calculate the residual error
    y_pred = model.predict(X_test)
    residuals = pd.Series(np.abs(y_test.values - y_pred), index=X_test.index)
    
    # 2. Train an Isolation Forest model on the residuals
    # High residuals (large errors) indicate unexpected consumption (wastage)
    iso_forest = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model and predict anomalies (-1 is an anomaly, 1 is normal)
    residuals_df = residuals.to_frame(name='residual_error')
    residuals_df['anomaly'] = iso_forest.fit_predict(residuals_df[['residual_error']])
    
    # Filter for anomaly points and combine with original data
    anomalies = residuals_df[residuals_df['anomaly'] == -1].copy()
    
    # Combine anomaly data with original features (temp, hour, etc.) and true consumption
    anomalies = anomalies.merge(y_test.to_frame(name=TARGET_COLUMN), left_index=True, right_index=True)
    anomalies = anomalies.merge(X_test, left_index=True, right_index=True)
    
    # Calculate the magnitude of the unexpected usage (wastage)
    anomalies['wastage_magnitude'] = anomalies['residual_error']
    
    # Sort by the largest residual error (biggest wastage)
    anomalies.sort_values(by='wastage_magnitude', ascending=False, inplace=True)
    
    print(f"Anomaly Detection Complete. Found {len(anomalies)} potential wastage events.")
    return anomalies

def generate_recommendations(anomalies: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generates AI-based, actionable recommendations based on the detected wastage patterns.
    """
    print("\nGenerating AI-Based Optimization Recommendations...")
    
    # Analyze the top 5 most severe anomalies
    top_anomalies = anomalies.head(5)
    
    # Find the most frequent pattern in the top anomalies
    most_common_hour = top_anomalies['hour_of_day'].mode().iloc[0] if not top_anomalies.empty else 10
    most_common_weekday = top_anomalies['is_weekday'].mode().iloc[0] if not top_anomalies.empty else 1
    
    recommendations = []
    
    # Recommendation 1: Time-based scheduling
    time_type = "a weekday" if most_common_weekday == 1 else "a weekend day"
    recommendations.append({
        'type': 'Scheduling Optimization',
        'details': f"High wastage is clustered around **Hour {most_common_hour}:00** on {time_type}. Review operations, large equipment startup, or lighting schedules around this specific time slot."
    })
    
    # Recommendation 2: Temperature correlation (If temperature data is available)
    if 'temperature_c' in top_anomalies.columns and not top_anomalies.empty:
        avg_temp_at_anomaly = top_anomalies['temperature_c'].mean()
        
        if avg_temp_at_anomaly > 25:
            temp_rec = f"Wastage occurs during **high temperatures ({avg_temp_at_anomaly:.1f}°C)**. Check HVAC setpoints for over-cooling, inspect cooling tower efficiency, and schedule large equipment use for cooler parts of the day."
        elif avg_temp_at_anomaly < 10:
            temp_rec = f"Wastage occurs during **low temperatures ({avg_temp_at_anomaly:.1f}°C)**. Verify that heating is not running excessively overnight or that insulation is adequate to prevent heat loss."
        else:
            temp_rec = "Temperature is not a primary driver of the top wastage events. Focus on behavioral or equipment issues."
            
        recommendations.append({'type': 'HVAC/Thermal Review', 'details': temp_rec})
    
    # Recommendation 3: Continuous Monitoring
    recommendations.append({
        'type': 'Continuous Improvement',
        'details': f"Implement real-time sub-metering on the equipment contributing to the largest wastage event ({top_anomalies.index[0].strftime('%Y-%m-%d %H:%M')}) to isolate the specific source of the unexpected consumption."
    })

    return recommendations

# ==============================================================================
# PART 4: MAIN EXECUTION FLOW
# ==============================================================================

def main():
    """
    Orchestrates the entire Energy Predictor pipeline.
    """
    print("=========================================================")
    print(" Smart Energy Efficiency Predictor (AI-Enabled System) ")
    print("=========================================================")
    
    # 1. Prepare Data
    X, y = prepare_data()
    
    if X.empty:
        print("\nFATAL ERROR: Data preparation failed or returned an empty dataset. Cannot proceed with training.")
        return
        
    # 2. Split Data (Time-Series Split)
    X_train, X_test, y_train, y_test = split_data_time_series(X, y)
    
    # 3. Train Forecasting Model
    model = train_model(X_train, y_train)
    
    # 4. Evaluate Model Performance
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
    
    # 5. Detect Anomalies (Wastage Patterns)
    anomalies_df = detect_anomalies(model, X_test, y_test)
    
    # 6. Generate Recommendations
    recommendations = generate_recommendations(anomalies_df)

    # 7. Final Output Summary
    print("\n\n====================== FINAL REPORT =======================")
    print(f"Total Analyzed Test Period: {X_test.index.min().date()} to {X_test.index.max().date()}")
    print(f"Total Anomalies Found: {len(anomalies_df)}")
    print("-----------------------------------------------------------")

    if not anomalies_df.empty:
        print("\nTOP 5 IDENTIFIED WASTAGE EVENTS (Highest Residual Error):")
        
        # Display key features of the top anomalies
        for i, (index, row) in enumerate(anomalies_df.head(5).iterrows()):
            time = index.strftime('%Y-%m-%d %H:%M')
            weekday_str = 'Weekday' if row['is_weekday'] == 1 else 'Weekend'
            temp = f"{row['temperature_c']:.1f}°C"
            actual = f"{row[TARGET_COLUMN]:.2f} kWh"
            expected = f"{(row[TARGET_COLUMN] - row['residual_error']):.2f} kWh"
            wastage = f"{row['wastage_magnitude']:.2f} kWh"

            print(f"  {i+1}. Time: {time} ({weekday_str}, Hour {row['hour_of_day']}:00)")
            print(f"     Actual: {actual} | Expected: {expected} | Wastage: {wastage} | Temp: {temp}")
            
        print("\n\nAI OPTIMIZATION RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"\n[{rec['type']}]")
            print(f"  > {rec['details']}")
            
    else:
        print("🎉 No significant anomalies detected in the test period. Efficiency appears optimal!")
        
    print("=========================================================")


if __name__ == '__main__':
    main()

