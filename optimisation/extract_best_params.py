import optuna
import os
import yaml
from pathlib import Path

def extract_best_trials():
    db_dir = Path("/workspaces/swe-pinn/optimisation/database/exploration")
    output_base_dir = Path("/workspaces/swe-pinn/optimisation/sensivity_analysis_output/best_parameters")
    
    if not db_dir.exists():
        print(f"Source directory {db_dir} does not exist.")
        return

    # Ensure output directory exists
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all .db files
    for db_file in db_dir.glob("*.db"):
        print(f"Processing {db_file.name}...")
        storage_url = f"sqlite:///{db_file}"
        
        try:
            # Get all study summaries in the database
            summaries = optuna.get_all_study_summaries(storage=storage_url)
            
            for summary in summaries:
                study = optuna.load_study(study_name=summary.study_name, storage=storage_url)
                
                try:
                    best_trial = study.best_trial
                except ValueError:
                    print(f"  No completed trials in study: {summary.study_name}")
                    continue

                # Save directly in the best_parameters folder without subdirectories
                output_file = output_base_dir / f"{summary.study_name}_best_params.txt"
                
                with open(output_file, "w") as f:
                    f.write(f"Study: {summary.study_name}\n")
                    f.write(f"Database: {db_file.name}\n")
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
            print(f"  Error processing {db_file.name}: {e}")

if __name__ == "__main__":
    extract_best_trials()