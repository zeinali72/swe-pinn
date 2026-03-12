"""Shared training utilities: setup, optimizer, data loading, and epoch loop."""
from src.training.setup import (
	setup_experiment,
	init_model_from_config,
	create_output_dirs,
	extract_loss_weights,
)
from src.training.optimizer import create_optimizer
from src.training.data_loading import resolve_data_mode, load_training_data, load_validation_from_file
from src.training.loop import run_training_loop, post_training_save
