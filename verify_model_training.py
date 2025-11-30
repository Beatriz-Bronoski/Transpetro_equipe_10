import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Mock Data Loading (Simplified from previous steps)
# We assume the merge logic works (verified previously). 
# Here we create a mock dataset that mimics the structure AFTER merge.

print("Creating Mock Data...")
n_samples = 100
data = {
    'speed': np.random.uniform(10, 15, n_samples),
    'CONSUMED_QUANTITY': np.random.uniform(20, 50, n_samples),
    'distance': np.random.uniform(200, 300, n_samples),
    'duration': np.random.uniform(20, 25, n_samples),
    'aftDraft': np.random.uniform(10, 15, n_samples),
    'fwdDraft': np.random.uniform(10, 15, n_samples),
    'displacement': np.random.uniform(100000, 150000, n_samples),
    'beaufortScale': np.random.randint(0, 8, n_samples),
    'Days_Since_Inspection': np.random.randint(0, 1000, n_samples),
    'shipName': np.random.choice(['ShipA', 'ShipB', 'ShipC'], n_samples),
    'Fouling_Score_General': np.random.uniform(0, 100, n_samples)
}
model_df = pd.DataFrame(data)

# Add AIS features to HALF the data (simulating sparsity)
model_df['AIS_Speed_Mean'] = np.nan
model_df.loc[:n_samples//2, 'AIS_Speed_Mean'] = np.random.uniform(10, 15, n_samples//2 + 1)
model_df['AIS_Speed_Std'] = np.nan
model_df.loc[:n_samples//2, 'AIS_Speed_Std'] = np.random.uniform(0, 2, n_samples//2 + 1)
model_df['AIS_Data_Count'] = np.nan
model_df.loc[:n_samples//2, 'AIS_Data_Count'] = np.random.randint(100, 1000, n_samples//2 + 1)

print("Mock Data Created. Shape:", model_df.shape)

# --- REPLICATING MODELING LOGIC ---

# 1. Feature Engineering
if 'AIS_Speed_Mean' in model_df.columns:
    model_df['Has_AIS_Data'] = model_df['AIS_Speed_Mean'].notnull().astype(int)
else:
    model_df['Has_AIS_Data'] = 0

features = [
    'speed', 'CONSUMED_QUANTITY', 'distance', 'duration',
    'aftDraft', 'fwdDraft', 'displacement',
    'beaufortScale',
    'Days_Since_Inspection',
    'Has_AIS_Data'
]
ais_features = ['AIS_Speed_Mean', 'AIS_Speed_Std', 'AIS_Data_Count']
for f in ais_features:
    if f in model_df.columns:
        features.append(f)

target = 'Fouling_Score_General'
model_df = model_df.dropna(subset=[target])

# Encoding
model_df_encoded = pd.get_dummies(model_df[features + ['shipName']], columns=['shipName'], drop_first=True)
X = model_df_encoded
y = model_df[target]

print(f"Features: {list(X.columns)}")

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train
print("Training XGBoost...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=10, # Low for speed
    max_depth=3,
    missing=float('nan'),
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

if rmse > 0:
    print("SUCCESS: Model trained and evaluated.")
else:
    print("FAILURE: RMSE is 0 (suspicious) or error occurred.")
