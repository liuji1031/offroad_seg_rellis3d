import optuna
from pathlib import Path
from typing import Optional
from offroad_det_seg_rellis.train.train import train, CONFIG_DIR, load_config


base_train_config = load_config(CONFIG_DIR / "hp_tune_exp_config.yaml")


def generate_hpt_split(input_file: Optional[Path] = None, output_file: Optional[Path] = None, step: int = 3):
    """Generate a subset of the training split list by taking every nth row.
    
    Args:
        input_file: Path to input split list file. Defaults to split_list_train.lst in CONFIG_DIR.
        output_file: Path to output split list file. Defaults to split_list_train_hpt.lst in CONFIG_DIR.
        step: Take every nth row (default: 3, meaning take 1 row every 3 rows).
    
    Returns:
        Path to the created output file.
    """
    if input_file is None:
        input_file = CONFIG_DIR / "split_list_train.lst"
    if output_file is None:
        output_file = CONFIG_DIR / "split_list_train_hpt.lst"
    
    # Read all lines from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Take every nth line (starting from index 0)
    subset_lines = [lines[i] for i in range(0, len(lines), step)]
    
    # Write to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(subset_lines)
    
    print(f"Generated subset: {len(subset_lines)} samples (from {len(lines)} total)")
    print(f"Output file: {output_file}")
    
    return output_file

def objective_train_config(trial: optuna.Trial):
    """Perform hyperparameter tuning for the training configuration."""
    # the list of hyperparameters to tune
    base_train_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-4, step=1e-5)
    base_train_config['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    focal_alpha = trial.suggest_float('focal_alpha', 0.0, 1.0, step=0.2)
    focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0, log=True)
    dice_smooth = trial.suggest_float('dice_smooth', 1e-5, 1.0, log=True)
    dice_weight = trial.suggest_float('dice_weight', 0.0, 2.0, log=True)

    base_train_config['loss']['focal_alpha'] = focal_alpha
    base_train_config['loss']['focal_gamma'] = focal_gamma
    base_train_config['loss']['dice_smooth'] = dice_smooth
    base_train_config['loss']['dice_weight'] = dice_weight

    model_config = load_config(CONFIG_DIR / base_train_config['model_config'])

    result = train(base_train_config, model_config)
    
    # Store checkpoint path and other metadata as trial user attributes
    trial.set_user_attr('best_checkpoint_path', result['best_checkpoint_path'])
    trial.set_user_attr('run_timestamp', result['run_timestamp'])
    
    return result['best_val_iou']

def train_config_study(
    n_trials: int = 100,
    study_name: str = "train_config_optimization",
    storage_path: Optional[Path] = None,
):
    """Run hyperparameter tuning study with support for resuming from previous runs.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study (used to identify and resume studies)
        storage_path: Path to SQLite database file for persisting study state.
                     If None, uses default location: CONFIG_DIR / "optuna_studies.db"
                     Set to ":memory:" for in-memory study (cannot be resumed)
    """
    # Set up storage
    if storage_path is None:
        storage_path = CONFIG_DIR / "optuna_studies.db"
        storage_url = f"sqlite:///{storage_path}"
    elif storage_path == ":memory:":
        storage_url = None
        print("Warning: Using in-memory storage. Study cannot be resumed after termination.")
    else:
        storage_url = f"sqlite:///{storage_path}"
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=storage_url,
        load_if_exists=True,
    )
    
    # Print study status
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed > 0:
        print(f"Resuming study '{study_name}': {n_completed} trials already completed")
        if storage_url:
            print(f"Storage: {storage_path}")
    else:
        print(f"Starting new study '{study_name}'")
        if storage_url:
            print(f"Storage: {storage_path}")
    
    # Run optimization
    study.optimize(objective_train_config, n_trials=n_trials)
    
    # report study results
    print("\n" + "="*80)
    print("Study results:")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.4f}")
    print("Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"\t{key}: {value}")
    
    # print checkpoint information
    print("Best checkpoint:")
    print(f"\tPath: {study.best_trial.user_attrs.get('best_checkpoint_path', 'N/A')}")
    print(f"\tTimestamp: {study.best_trial.user_attrs.get('run_timestamp', 'N/A')}")
    print("="*80)
    
    return study


