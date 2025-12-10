import os
import glob
import optuna
from optuna.importance import FanovaImportanceEvaluator
import sys

def main():
    # --- Configuration ---
    # 1. Input: Path to where your .db files are stored
    db_dir = os.path.join("optimisation", "database")
    
    # 2. Output: Path to save the analysis plots and text logs
    analysis_dir = os.path.join("optimisation", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"🔍 Scanning for databases in: {db_dir}")
    print(f"📂 Output directory set to: {analysis_dir}\n")
    
    db_files = glob.glob(os.path.join(db_dir, "*.db"))
    
    if not db_files:
        print("❌ No database files found!")
        return

    print(f"✅ Found {len(db_files)} databases. Starting analysis...\n")

    # --- Iterate through each database ---
    for db_path in db_files:
        filename = os.path.basename(db_path)
        storage_url = f"sqlite:///{db_path}"
        file_slug = os.path.splitext(filename)[0]

        print(f"{'='*60}")
        print(f"📊 Processing File: {filename}")

        try:
            # --- Retrieve the actual study name from the DB ---
            summaries = optuna.get_all_study_summaries(storage=storage_url)
            if not summaries:
                print(f"⚠️  Skipping {filename}: Database is empty.")
                continue
            
            # Assume one study per file
            target_study = summaries[0]
            study_name = target_study.study_name
            print(f"   ↳ Found Internal Study Name: {study_name}")

            # 1. Load the study
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            
            # Check for completed trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) < 2:
                print(f"⚠️  Skipping {study_name}: Not enough completed trials ({len(completed_trials)}).")
                continue

            # 2. Run fANOVA Analysis
            # We use a fixed seed for reproducibility
            evaluator = FanovaImportanceEvaluator(n_trees=64, seed=42)
            importances = optuna.importance.get_param_importances(study, evaluator=evaluator)

            # 3. Generate Text Log
            log_lines = []
            log_lines.append(f"--- Parameter Importance (fANOVA) for {study_name} ---")
            log_lines.append(f"Total Completed Trials: {len(completed_trials)}")
            log_lines.append("-" * 40)
            
            for param, importance in importances.items():
                line = f"{param:<30}: {importance:.4f}"
                log_lines.append(line)
            
            log_content = "\n".join(log_lines)

            # Print to Console (so you can copy-paste to me)
            print("\n" + log_content + "\n")

            # Save to .txt file
            txt_output_path = os.path.join(analysis_dir, f"{file_slug}_importance.txt")
            with open(txt_output_path, "w") as f:
                f.write(log_content)
            print(f"📝 Text log saved to: {txt_output_path}")

            # 4. Generate and Save Plot (HTML)
            fig = optuna.visualization.plot_param_importances(study, evaluator=evaluator)
            html_output_path = os.path.join(analysis_dir, f"{file_slug}_importance.html")
            fig.write_html(html_output_path)
            print(f"📈 Plot saved to:     {html_output_path}\n")

        except Exception as e:
            print(f"❌ Error analyzing {filename}: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"{'='*60}")
    print(f"🎉 Analysis Complete! Check the '{analysis_dir}' folder.")

if __name__ == "__main__":
    # Ensure we can import project modules if running from root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    main()