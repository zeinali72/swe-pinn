import argparse
import os
import glob
import optuna
from optuna.importance import FanovaImportanceEvaluator
import sys

from optimisation.utils import setup_study_storage


def main(storage_url=None):
    # --- Configuration ---
    analysis_dir = os.path.join("optimisation", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    if storage_url is not None:
        # Remote or explicit storage: list all studies from that URL
        print(f"Connecting to storage: {storage_url}")
        print(f"Output directory set to: {analysis_dir}\n")
        summaries = optuna.get_all_study_summaries(storage=storage_url)
        if not summaries:
            print("No studies found in storage!")
            return
        db_entries = [(storage_url, s.study_name, s.study_name) for s in summaries]
    else:
        # Default: scan local SQLite files
        db_dir = os.path.join("optimisation", "database")
        print(f"Scanning for databases in: {db_dir}")
        print(f"Output directory set to: {analysis_dir}\n")

        db_files = glob.glob(os.path.join(db_dir, "*.db"))
        if not db_files:
            print("No database files found!")
            return

        print(f"Found {len(db_files)} databases. Starting analysis...\n")
        db_entries = []
        for db_path in db_files:
            url = f"sqlite:///{db_path}"
            file_slug = os.path.splitext(os.path.basename(db_path))[0]
            try:
                summaries = optuna.get_all_study_summaries(storage=url)
                for s in summaries:
                    db_entries.append((url, s.study_name, file_slug))
            except Exception as e:
                print(f"Error reading {db_path}: {e}")

    # --- Iterate through each study ---
    for entry_storage, study_name, file_slug in db_entries:
        print(f"{'='*60}")
        print(f"Processing study: {study_name}")

        try:
            study = optuna.load_study(study_name=study_name, storage=entry_storage)

            # Check for completed trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) < 2:
                print(f"  Skipping {study_name}: Not enough completed trials ({len(completed_trials)}).")
                continue

            # Run fANOVA Analysis (fixed seed for reproducibility)
            evaluator = FanovaImportanceEvaluator(n_trees=64, seed=42)
            importances = optuna.importance.get_param_importances(study, evaluator=evaluator)

            # Generate Text Log
            log_lines = []
            log_lines.append(f"--- Parameter Importance (fANOVA) for {study_name} ---")
            log_lines.append(f"Total Completed Trials: {len(completed_trials)}")
            log_lines.append("-" * 40)

            for param, importance in importances.items():
                line = f"{param:<30}: {importance:.4f}"
                log_lines.append(line)

            log_content = "\n".join(log_lines)
            print("\n" + log_content + "\n")

            # Save to .txt file
            txt_output_path = os.path.join(analysis_dir, f"{file_slug}_importance.txt")
            with open(txt_output_path, "w") as f:
                f.write(log_content)
            print(f"Text log saved to: {txt_output_path}")

            # Generate and Save Plot (HTML)
            fig = optuna.visualization.plot_param_importances(study, evaluator=evaluator)
            html_output_path = os.path.join(analysis_dir, f"{file_slug}_importance.html")
            fig.write_html(html_output_path)
            print(f"Plot saved to:     {html_output_path}\n")

        except Exception as e:
            print(f"Error analyzing {study_name}: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"{'='*60}")
    print(f"Analysis Complete! Check the '{analysis_dir}' folder.")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Run fANOVA parameter importance analysis on Optuna studies.")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL. Falls back to OPTUNA_STORAGE env var, "
                             "then scans local SQLite files.")
    args = parser.parse_args()

    url = None
    if args.storage is not None:
        url = setup_study_storage(args.storage, project_root)
    elif os.environ.get("OPTUNA_STORAGE"):
        url = setup_study_storage(None, project_root)

    main(url)
