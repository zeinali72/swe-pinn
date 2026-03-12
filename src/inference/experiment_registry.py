"""Per-experiment metadata for the inference pipeline.

The runner consults this registry to decide which assets to load and which
optional metric groups to evaluate.
"""

EXPERIMENT_REGISTRY = {
    "experiment_1": {
        "domain_type": "rectangular",
        "has_building": False,
        "has_bathymetry": False,
        "reference_type": "analytical",
        "flood_extent": False,
        "extra_checks": ["hv_zero", "wet_dry_front"],
        "default_val_file": None,
    },
    "experiment_2": {
        "domain_type": "rectangular",
        "has_building": True,
        "has_bathymetry": False,
        "reference_type": "simulation",
        "flood_extent": False,
        "extra_checks": ["building_split"],
        "default_val_file": "validation_sample.npy",
    },
    "experiment_3": {
        "domain_type": "rectangular",
        "has_building": False,
        "has_bathymetry": True,
        "reference_type": "simulation",
        "flood_extent": False,
        "extra_checks": [],
        "default_val_file": "validation_gauges.npy",
    },
    "experiment_4": {
        "domain_type": "rectangular",
        "has_building": False,
        "has_bathymetry": True,
        "reference_type": "simulation",
        "flood_extent": False,
        "extra_checks": [],
        "default_val_file": "validation_sample.npy",
    },
    "experiment_5": {
        "domain_type": "rectangular",
        "has_building": False,
        "has_bathymetry": True,
        "reference_type": "simulation",
        "flood_extent": False,
        "extra_checks": ["overtopping"],
        "default_val_file": "validation_sample.npy",
    },
    "experiment_6": {
        "domain_type": "rectangular",
        "has_building": False,
        "has_bathymetry": True,
        "reference_type": "simulation",
        "flood_extent": False,
        "extra_checks": [],
        "default_val_file": "validation_sample.npy",
    },
    "experiment_7": {
        "domain_type": "irregular",
        "has_building": False,
        "has_bathymetry": True,
        "reference_type": "simulation",
        "flood_extent": True,
        "extra_checks": [],
        "default_val_file": "validation_sample.npy",
    },
    "experiment_8": {
        "domain_type": "irregular",
        "has_building": True,
        "has_bathymetry": True,
        "reference_type": "simulation",
        "flood_extent": True,
        "extra_checks": ["per_building_slip", "street_corridor"],
        "default_val_file": "validation_sample.npy",
    },
}


def get_experiment_meta(experiment_name: str) -> dict:
    """Return registry entry for *experiment_name*, with sensible defaults."""
    return EXPERIMENT_REGISTRY.get(experiment_name, {
        "domain_type": "rectangular",
        "has_building": False,
        "has_bathymetry": False,
        "reference_type": "simulation",
        "flood_extent": False,
        "extra_checks": [],
        "default_val_file": "validation_sample.npy",
    })
