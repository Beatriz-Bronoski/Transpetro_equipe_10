import json
import os

notebook_path = r"c:/Users/Aluno/Downloads/Hackaton Transpetro/Transpetro_equipe_10/EDA.ipynb"

model_cells_source = [
    [
        "# 6. Modeling: Fouling Detection with XGBoost\n",
        "import xgboost as xgb\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Prepare Data for Modeling\n",
        "if 'merged_data_ais' in locals():\n",
        "    model_df = merged_data_ais.copy()\n",
        "else:\n",
        "    model_df = merged_data.copy() # Fallback if AIS merge failed/skipped\n",
        "\n",
        "# 1. Feature Engineering: Has_AIS_Data Flag\n",
        "if 'AIS_Speed_Mean' in model_df.columns:\n",
        "    model_df['Has_AIS_Data'] = model_df['AIS_Speed_Mean'].notnull().astype(int)\n",
        "else:\n",
        "    model_df['Has_AIS_Data'] = 0\n",
        "\n",
        "# Select Features\n",
        "# We include both high-precision (AIS) and low-precision (Noon Report) features\n",
        "features = [\n",
        "    'speed', 'CONSUMED_QUANTITY', 'distance', 'duration', # Noon Report Data\n",
        "    'aftDraft', 'fwdDraft', 'displacement', # Ship State\n",
        "    'beaufortScale', # Weather (if available)\n",
        "    'Days_Since_Inspection', # Time Factor\n",
        "    'Has_AIS_Data' # Structural Missingness Flag\n",
        "]\n",
        "\n",
        "# Add AIS features if they exist\n",
        "ais_features = ['AIS_Speed_Mean', 'AIS_Speed_Std', 'AIS_Data_Count']\n",
        "for f in ais_features:\n",
        "    if f in model_df.columns:\n",
        "        features.append(f)\n",
        "\n",
        "target = 'Fouling_Score_General'\n",
        "\n",
        "# Filter for rows with valid Target\n",
        "model_df = model_df.dropna(subset=[target])\n",
        "\n",
        "# Handle categorical features if any (e.g., shipName) - XGBoost can handle them but needs encoding\n",
        "# For simplicity, let's stick to numeric first. We can add OneHotEncoding for 'shipName' later if needed.\n",
        "# Actually, 'shipName' is crucial for baseline performance. Let's use Pandas dummies.\n",
        "model_df_encoded = pd.get_dummies(model_df[features + ['shipName']], columns=['shipName'], drop_first=True)\n",
        "X = model_df_encoded\n",
        "y = model_df[target]\n",
        "\n",
        "print(f\"Training Data Shape: {X.shape}\")\n",
        "print(f\"Features used: {list(X.columns)}\")\n"
    ],
    [
        "# 2. Train/Test Split\n",
        "# We use a random split here to ensure both 'Old' (No AIS) and 'New' (AIS) data are in both sets for validation.\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "print(f\"Train set size: {len(X_train)}\")\n",
        "print(f\"Test set size: {len(X_test)}\")\n"
    ],
    [
        "# 3. XGBoost Model Configuration\n",
        "# XGBoost handles missing values (NaNs in AIS columns) natively.\n",
        "model = xgb.XGBRegressor(\n",
        "    objective='reg:squarederror',\n",
        "    n_estimators=100,\n",
        "    learning_rate=0.1,\n",
        "    max_depth=5,\n",
        "    missing=float('nan'), # Explicitly tell XGBoost to treat NaNs as missing\n",
        "    random_state=42\n",
        ")\n",
        "\n",
        "print(\"Training XGBoost Model...\")\n",
        "model.fit(X_train, y_train)\n",
        "# Feature Importance\n",
        "plt.figure(figsize=(10, 8))\n",
        "xgb.plot_importance(model, max_num_features=20)\n",
        "plt.title(\"Feature Importance\")\n",
        "plt.show()\n"
    ]
]

def add_model_cells():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    new_cells = []
    for source in model_cells_source:
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        })
    
    nb_data['cells'].extend(new_cells)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)
    
    print("Added Modeling cells to notebook.")

if __name__ == "__main__":
    add_model_cells()
