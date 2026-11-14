"""End-to-end pipeline entrypoint for wafer processing histories and stress prediction.

Usage:
    .\\.venv\\Scripts\\python.exe src/main.py [--config <path>] [--headless]
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from src.config.pseudo_data_schema import load_pseudo_data_config
from src.utility.pseudo_data import generate_pseudo_data, save_pseudo_data

LOGGER = logging.getLogger(__name__)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Generate wafer processing histories and stress measurements."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/pseudo_data_config.yaml",
        help="Path to pseudo-data generator YAML configuration file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode to suppress interactive windows.",
    )
    return parser.parse_args(argv)


def _log_summary_statistics(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Log dataset summary metrics including wafer counts and stress distribution."""
    group_count = int(dataframe["group"].nunique())
    wafer_count = int(dataframe["wafer_id"].nunique())
    total_row_count = len(dataframe)

    stress_mask = dataframe["parameter"] == "stress"
    stress_numeric = dataframe.loc[stress_mask, "value"].astype(float)
    stress_min = float(stress_numeric.min())
    stress_max = float(stress_numeric.max())
    stress_mean = float(stress_numeric.mean())
    stress_std = float(stress_numeric.std())

    LOGGER.info("Successfully generated pseudo-data saved to: %s", output_path)
    LOGGER.info("Summary: %d groups, %d wafers, %d long-table rows", group_count, wafer_count, total_row_count)
    LOGGER.info(
        "Stress statistics (N=%d): min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        len(stress_numeric),
        stress_min,
        stress_max,
        stress_mean,
        stress_std,
    )


def main(argv: list[str] | None = None) -> Path:
    """Execute pseudo-data generation pipeline entrypoint.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line arguments, defaults to sys.argv[1:].

    Returns
    -------
    Path
        Path to the generated and persisted pseudo-data CSV file.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parsed_args = _parse_arguments(argv)

    config_path = Path(parsed_args.config).resolve()
    LOGGER.info("Loading configuration from: %s (headless=%s)", config_path, parsed_args.headless)
    configuration = load_pseudo_data_config(config_path)

    LOGGER.info("Starting pseudo-data generation (random_seed=%d)...", configuration.random_seed)
    generated_dataframe = generate_pseudo_data(configuration)

    output_path = Path(configuration.output_path).resolve()
    saved_path = save_pseudo_data(generated_dataframe, output_path)
    _log_summary_statistics(generated_dataframe, saved_path)

    return saved_path


if __name__ == "__main__":
    main(sys.argv[1:])
