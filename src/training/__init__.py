"""Shared training utilities: setup, optimizer, data loading, and epoch loop."""
from src.training.setup import (
	setup_experiment,
	init_model_from_config,
	create_output_dirs,
	extract_loss_weights,
	get_active_loss_weights,
	get_experiment_name,
	resolve_experiment_paths,
	get_data_filename,
	resolve_configured_asset_path,
	get_sampling_count_from_config,
	get_boundary_segment_count,
	calculate_num_batches,
	apply_irregular_domain_bounds,
	apply_output_scales,
)
from src.training.optimizer import create_optimizer
from src.training.data_loading import resolve_data_mode, load_training_data, load_validation_from_file
from src.training.loop import run_training_loop, post_training_save
from src.training.epoch import (
	make_scan_body,
	sample_and_batch,
	empty_batch,
	maybe_batch_data,
)
