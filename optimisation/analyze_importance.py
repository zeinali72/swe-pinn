import os
import glob
import optuna
from optuna.importance import FanovaImportanceEvaluator
import sys

def main():
    # --- Configuration ---
    db_dir = os.path.join("optimisation", "database")
    analysis_dir = os.path.join("optimisation", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"🔍 Scanning for databases in: {db_dir}")
    print(f"📂 Output directory set to: {analysis_dir}\n")
    
    db_files = glob.glob(os.path.join(db_dir, "*.db"))
    
    if not db_files:
        print("❌ No database files found!")
        return

    print(f"✅ Found {len(db_files)} databases. Starting analysis...\n")

    for db_path in db_files:
        filename = os.path.basename(db_path)
        storage_url = f"sqlite:///{db_path}"

        print(f"{'='*60}")
        print(f"📊 Processing File: {filename}")

        try:
            # --- FIX: Retrieve the actual study name from the DB ---
            summaries = optuna.get_all_study_summaries(storage=storage_url)
            if not summaries:
                print(f"⚠️  Skipping {filename}: Database is empty.")
                continue
            
            # We assume one study per file (standard for this project)
            target_study = summaries[0]
            study_name = target_study.study_name
            print(f"   ↳ Found Internal Study Name: {study_name}")

            # 1. Load the study using the CORRECT name
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            
            # Check for completed trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) < 2:
                print(f"⚠️  Skipping {study_name}: Not enough completed trials ({len(completed_trials)}).")
                continue

            # 2. Run fANOVA Analysis
            evaluator = FanovaImportanceEvaluator(n_trees=64, seed=42)
            importances = optuna.importance.get_param_importances(study, evaluator=evaluator)

            # 3. Print Results
            print(f"--- fANOVA Parameter Importance ---")
            for param, importance in importances.items():
                print(f"{param:<30}: {importance:.4f}")

            # 4. Generate Plot
            fig = optuna.visualization.plot_param_importances(study, evaluator=evaluator)
            
            # Use the filename for the output image to ensure uniqueness
            # (e.g. database_hpo-sensitivity.db -> database_hpo-sensitivity_importance.html)
            file_slug = os.path.splitext(filename)[0]
            output_filename = f"{file_slug}_importance.html"
            output_path = os.path.join(analysis_dir, output_filename)
            
            fig.write_html(output_path)
            print(f"\n✅ Plot saved to: {output_path}\n")

        except Exception as e:
            print(f"❌ Error analyzing {filename}: {e}\n")

    print(f"{'='*60}")
    print(f"🎉 Analysis Complete! Check the '{analysis_dir}' folder.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    main()