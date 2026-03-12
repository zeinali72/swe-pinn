"""Trial naming utilities."""
import datetime


def generate_trial_name(config_filename):
    """Generate a unique trial name using the current date and config filename."""
    now = datetime.datetime.now()
    return f"{now.strftime('%Y-%m-%d_%H-%M')}_{config_filename}"
