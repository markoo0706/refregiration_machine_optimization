
import os
import sys
import pandas as pd
import joblib
from datetime import datetime

# --- Path Setup ---
# Add project root to sys.path to allow for module imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
web_path = os.path.join(project_root, 'web')

paths_to_add = [project_root, src_path, web_path]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# --- Imports ---
try:
    from src.data_processing import DataPreprocessor
    from src.feature_engineering import FeatureProcessor
    from src.prediction_models import ModelManager
    print("Successfully imported project modules.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from the project root directory.")
    sys.exit(1)

# --- Configuration ---
MODELS_DIR = os.path.join(project_root, 'models')
POWER_MODEL_TARGET = 'cooling_system_total_power_kw'
COP_MODEL_TARGET = 'cooling_system_cop'

POWER_MODEL_FILENAME = 'default_cooling_system_total_power_kw.pkl'
COP_MODEL_FILENAME = 'default_cooling_system_cop.pkl'

# --- Main Execution ---
def main():
    """
    Main function to run the default model training pipeline.
    """
    print("--- Starting Default Model Training ---")

    # --- 1. Data Preprocessing ---
    print("\n[Phase 1/4] Running data preprocessing...")
    preprocessor = DataPreprocessor()
    try:
        processed_df = preprocessor.run_preprocessing_pipeline()
        print("Data preprocessing completed successfully.")
    except Exception as e:
        print(f"Data preprocessing failed: {e}")
        return

    # --- 2. Feature Engineering ---
    print("\n[Phase 2/4] Running feature engineering...")
    feature_processor = FeatureProcessor(verbose=False)
    try:
        features_df = feature_processor.process(processed_df)
        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        return

    # --- 3. Train Power Model ---
    print(f"\n[Phase 3/4] Training Power Model ({POWER_MODEL_TARGET})...")
    try:
        power_X, power_y = feature_processor.get_training_data(features_df, target_columns=[POWER_MODEL_TARGET])
        
        # Ensure target is not in features
        if POWER_MODEL_TARGET in power_X.columns:
            power_X = power_X.drop(columns=[POWER_MODEL_TARGET])

        power_model_manager = ModelManager()
        power_metrics = power_model_manager.train_model('xgboost', power_X, power_y)
        
        power_model = power_model_manager.get_model('xgboost')
        
        # Save the model
        os.makedirs(MODELS_DIR, exist_ok=True)
        power_model_path = os.path.join(MODELS_DIR, POWER_MODEL_FILENAME)
        joblib.dump(power_model, power_model_path)
        
        print(f"Power model trained. R²: {power_metrics.get('r2', 'N/A'):.4f}")
        print(f"Power model saved to: {power_model_path}")

    except Exception as e:
        print(f"Power model training failed: {e}")
        return

    # --- 4. Train COP Model ---
    print(f"\n[Phase 4/4] Training COP Model ({COP_MODEL_TARGET})...")
    try:
        cop_X, cop_y = feature_processor.get_training_data(features_df, target_columns=[COP_MODEL_TARGET])

        # Ensure target is not in features
        if COP_MODEL_TARGET in cop_X.columns:
            cop_X = cop_X.drop(columns=[COP_MODEL_TARGET])

        cop_model_manager = ModelManager()
        cop_metrics = cop_model_manager.train_model('xgboost', cop_X, cop_y)

        cop_model = cop_model_manager.get_model('xgboost')

        # Save the model
        cop_model_path = os.path.join(MODELS_DIR, COP_MODEL_FILENAME)
        joblib.dump(cop_model, cop_model_path)

        print(f"COP model trained. R²: {cop_metrics.get('r2', 'N/A'):.4f}")
        print(f"COP model saved to: {cop_model_path}")

    except Exception as e:
        print(f"COP model training failed: {e}")
        return
        
    print("\n--- Default Model Training Completed Successfully ---")

if __name__ == "__main__":
    main()
