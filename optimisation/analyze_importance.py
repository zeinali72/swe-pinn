import optuna
from optuna.importance import FanovaImportanceEvaluator
import matplotlib.pyplot as plt

# 1. Load your study
storage_url = "sqlite:///optimisation/database/hpo-sensitivity-fourier-nobuilding.db" # Update path
study_name = "hpo-sensitivity-fourier-nobuilding" # Update name

study = optuna.load_study(study_name=study_name, storage=storage_url)

# 2. Define the Evaluator (fANOVA)
# This calculates the marginal contribution of each parameter to the variance in NSE
evaluator = FanovaImportanceEvaluator(n_trees=64, seed=42)

# 3. Compute Importances
importances = optuna.importance.get_param_importances(study, evaluator=evaluator)

print("--- Parameter Importance (fANOVA) ---")
for param, importance in importances.items():
    print(f"{param:<25}: {importance:.4f}")

# 4. (Optional) Visualization
# This uses the same evaluator defined above
fig = optuna.visualization.plot_param_importances(study, evaluator=evaluator)
fig.write_image("fANOVA_sensitivity.png")