import argparse
import optuna
import os
import yaml
from pathlib import Path

from optimisation.utils import setup_study_storage

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_SCRIPT_DIR.parent)


def extract_best_trials(storage_url=None):
    output_base_dir = _SCRIPT_DIR / "sensitivity_analysis_output" / "best_parameters"

    if storage_url is None:
        # Default: scan local SQLite files in database/exploration/
        db_dir = _SCRIPT_DIR / "database" / "exploration"
        if not db_dir.exists():
            print(f"Source directory {db_dir} does not exist.")
            return
        storage_urls = [f"sqlite:///{db_file}" for db_file in db_dir.glob("*.db")]
        if not storage_urls:
            print(f"No .db files found in {db_dir}.")
            return
    else:
        storage_urls = [storage_url]

    # Ensure output directory exists
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for url in storage_urls:
        db_label = url.split("///")[-1] if "///" in url else url
        print(f"Processing {db_label}...")

        try:
            # Get all study summaries in the database
            summaries = optuna.get_all_study_summaries(storage=url)

            for summary in summaries:
                study = optuna.load_study(study_name=summary.study_name, storage=url)

                try:
                    best_trial = study.best_trial
                except ValueError:
                    print(f"  No completed trials in study: {summary.study_name}")
                    continue

                # Save directly in the best_parameters folder without subdirectories
                output_file = output_base_dir / f"{summary.study_name}_best_params.txt"

                with open(output_file, "w") as f:
                    f.write(f"Study: {summary.study_name}\n")
                    f.write(f"Database: {db_label}\n")
                    f.write(f"Best Trial Number: {best_trial.number}\n")
                    f.write(f"Best Value (NSE): {best_trial.value}\n")
                    f.write("-" * 40 + "\n")
                    f.write("Parameters:\n")
                    f.write(yaml.dump(best_trial.params, default_flow_style=False))

                    # If full_config was stored in user_attrs
                    if "full_config" in best_trial.user_attrs:
                        f.write("-" * 40 + "\n")
                        f.write("Full Configuration (User Attrs):\n")
                        f.write(yaml.dump(best_trial.user_attrs["full_config"], default_flow_style=False))

                print(f"  Saved best parameters to {output_file}")

        except Exception as e:
            print(f"  Error processing {db_label}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract best trial parameters from Optuna studies.")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL. Falls back to OPTUNA_STORAGE env var, "
                             "then scans local SQLite files.")
    args = parser.parse_args()

    storage = None
    if args.storage is not None:
        storage = setup_study_storage(args.storage, _PROJECT_ROOT)
    elif os.environ.get("OPTUNA_STORAGE"):
        storage = setup_study_storage(None, _PROJECT_ROOT)

    extract_best_trials(storage)
