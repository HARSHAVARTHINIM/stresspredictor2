# train_autotrain.py
from autotrain import AutoTrain

# ↓↓↓ PASTE YOUR COPIED TOKEN INSIDE THESE QUOTES ↓↓↓
API_KEY = "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"  
# Example: API_KEY = "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"

USERNAME = "Naveetha"   # ← Your HF username, e.g., "johnsmith"

# Example: Tabular classification training
project = AutoTrain(
    api_key=API_KEY,
    project_name="student-stress-predictor",
    username=USERNAME,
    task="tabular-classification",
    train_data_path="StressLevelDataset.csv",  # Update with your actual CSV path
    target_column="stress_level",             # Update with your actual target column
    model="xgboost",                         # Example model, can be changed
    max_trials=10,                            # Number of model trials
    metric="accuracy",                       # Evaluation metric
)

# Start training
project.train()

# Save best model
project.save("best_student_stress_model")