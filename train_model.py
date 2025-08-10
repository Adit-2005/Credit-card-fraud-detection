import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import os

def safe_print(message):
    """Prints messages safely on all terminals"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'replace').decode())

# 1. Load data with explicit error handling
try:
    df = pd.read_csv("data/fraudTrain.csv")
    safe_print("[SUCCESS] Data loaded successfully. Shape: {}".format(df.shape))
except Exception as e:
    safe_print("[ERROR] Failed to load data: {}".format(str(e)))
    exit()

# 2. Model pipeline
model = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])
        ])),
    ('classifier', RandomForestClassifier(
        n_estimators=150,
        class_weight='balanced_subsample',
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

# 3. Train model
try:
    safe_print("[STATUS] Training model...")
    model.fit(
        df[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'category']],
        df['is_fraud']
    )
    safe_print("[SUCCESS] Model trained successfully!")
except Exception as e:
    safe_print("[ERROR] Training failed: {}".format(str(e)))
    exit()

# 4. Save files with verification
try:
    dump(model, 'pretrained_model.joblib')
    df.to_pickle('training_data.pkl')
    safe_print("[STATUS] Saved model files:")
    safe_print(" - {}".format(os.path.abspath('pretrained_model.joblib')))
    safe_print(" - {}".format(os.path.abspath('training_data.pkl')))
    
    # Verify files exist
    assert os.path.exists('pretrained_model.joblib'), "Model file not created!"
    assert os.path.exists('training_data.pkl'), "Data file not created!"
    safe_print("[SUCCESS] Verification passed - files exist!")
except Exception as e:
    safe_print("[ERROR] Failed to save files: {}".format(str(e)))