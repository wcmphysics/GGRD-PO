"""End-to-end pipeline entrypoint for wafer processing histories and stress prediction.

Usage:
    .\\.venv\\Scripts\\python.exe src/main.py --config src/config/config.yaml --headless
"""

import argparse
from collections.abc import Callable
import csv
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np

from src.analysis.ablation import write_transition_ablation
from src.analysis.exploration import write_exploration
from src.config.pipeline_schema import (
    OptimizeConfig,
    PipelineConfig,
    PredictConfig,
    SyntheticDomainConfig,
    TrainConfig,
    load_config,
    load_entry_config,
)
from src.config.pseudo_data_schema import PseudoDataConfig, load_pseudo_data_config
from src.dataset.contracts import ProcessSequence, WaferSample
from src.dataset.history import read_history
from src.dataset.requests import read_requests
from src.feature.wide import FeatureSchema, build_training_batch
from src.inference.bundle import DeploymentBundle, load_bundle, save_bundle
from src.inference.predict import predict_sequences
from src.optimize.domain import synthetic_domain
from src.optimize.search import SearchContext, optimize_sequences
from src.trainer.reporting import (
    log_evaluation, training_manifest, write_evaluation, write_trace_reports, write_wide_table,
)
from src.trainer.splits import preflight_training, split_records
from src.trainer.stacking import evaluate_stack, fit_deployment
from src.utility.pseudo_data import generate_pseudo_data, save_pseudo_data
from src.utility.run_output import file_fingerprint, run_timestamp, stage_output, timestamped_output, write_json
from src.visual.convergence import plot_convergence
from src.visual.pipeline_plots import plot_optimization, plot_training

LOGGER = logging.getLogger(__name__)


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Accept the workflow YAML and the rendering option."""
    parser = argparse.ArgumentParser(description="Run the wafer pipeline stages enabled in the configuration.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/config.yaml"),
        help="Workflow YAML (default: src/config/config.yaml). A generator-only YAML also works.",
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


def _generate(configuration: PseudoDataConfig, root: Path) -> Path:
    """Generate a CSV using the selected pseudo-data settings."""
    LOGGER.info("Starting pseudo-data generation (random_seed=%d)...", configuration.random_seed)
    generated_dataframe = generate_pseudo_data(configuration)
    output_path = (root / configuration.output_path).resolve()
    saved_path = save_pseudo_data(generated_dataframe, output_path)
    _log_summary_statistics(generated_dataframe, saved_path)

    return saved_path


def _generate_stage(config_path: Path, root: Path, _headless: bool, _timestamp: str, _bundle: Path | None) -> Path:
    """Load the enabled generation stage and write its CSV."""
    return _generate(load_pseudo_data_config(config_path), root)


def _record_count(path: Path) -> int:
    """Count CSV records without counting embedded line endings as records."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        return sum(bool(row) for row in csv.reader(source)) - 1


def _group_references(samples: tuple[WaferSample, ...]) -> dict[str, ProcessSequence]:
    """Keep one already validated, unimputed history per group."""
    groups = {sample.group: sample.sequence for sample in samples}
    return dict(sorted(groups.items()))


def _log_training_mode(config: TrainConfig, partitions: int) -> None:
    """Report the active predictors and the evaluation actually scheduled."""
    LOGGER.info("Training mode: %s; active models: %s", config.mode,
                ", ".join(base.name for base in config.active_models))
    if not config.evaluate:
        LOGGER.info("Fit only: no independent held-out score will be computed.")
        return
    LOGGER.info("Grouped evaluation: %d repeats x %d folds, %d distinct partitions",
                config.cv.outer_repeats, config.cv.outer_folds, partitions)
    if config.mode == "stacked":
        LOGGER.info("Meta-training OOF: %d inner folds", config.cv.inner_folds)


def _train(config_path: Path, root: Path, headless: bool, timestamp: str, _bundle: Path | None) -> Path:
    """Evaluate grouped stacks, fit deployment, and publish their artifacts together."""
    config_hash = file_fingerprint(config_path)
    config = load_config(config_path, TrainConfig)
    if file_fingerprint(config_path) != config_hash:
        raise ValueError(f"Configuration {config_path} changed while reading it. Expected stable settings.")
    destination = timestamped_output(root / config.output.directory, timestamp)
    source = root / config.data.csv_path
    with stage_output(destination) as output:
        # Canonical data and all nested group plans exist before any estimator sees a row.
        hashes = {"source_csv": file_fingerprint(source), "train_config": config_hash}
        samples = read_history(source)
        schema = FeatureSchema(config.features.N_step)
        batch = build_training_batch(samples, schema)
        plan = preflight_training(batch, config)
        raw_rows = _record_count(source)
        write_exploration(samples, raw_rows, output / "exploration")
        write_wide_table(batch, output / "wide_table.csv")
        if file_fingerprint(source) != hashes["source_csv"]:
            raise ValueError(f"Wafer history {source} changed while reading it. Expected a stable training input.")

        LOGGER.info("Training input %s: %d long rows, %d wafers, %d groups, %d wide features",
                    source, raw_rows, len(batch.X), len(set(batch.groups)), len(schema.columns))
        _log_training_mode(config, plan.distinct_outer_partitions)
        # Outer evaluation and ablation reuse one plan. Final training uses its separate plan.
        evaluations = []
        if config.evaluate:
            full = evaluate_stack(batch, schema, config, plan)
            evaluations.append(full)
            log_evaluation(write_evaluation(full, output, config.cv.mape_min_abs_y))
            if config.analysis.transition_ablation:
                ablated = evaluate_stack(batch, schema.without_transitions(), config, plan,
                                         variant="transition_ablation")
                write_transition_ablation(full.predictions, ablated.predictions, output / "analysis",
                                          config.primary_candidate)
                evaluations.append(ablated)
        final = fit_deployment(batch, schema, config, plan)
        manifest = training_manifest(batch, schema, config, plan, hashes)
        write_trace_reports(output, split_records(batch, plan), evaluations, final, manifest)
        bundle = DeploymentBundle(
            stack=final.stack, schema=schema, references=_group_references(samples),
            config=config.model_dump(mode="json"), source_fingerprint=hashes["source_csv"],
            target_unit=config.data.target_unit, parameter_units=config.data.parameter_units.model_dump(),
        )
        save_bundle(bundle, output / "model")
        if config.mode == "stacked":
            LOGGER.info("Final meta weights: %s", ", ".join(f"{name}={weight:.4f}" for name, weight in
                        zip(final.stack.base_names, final.stack.weights, strict=True)))
        else:
            LOGGER.info("Final predictor: %s, pass-through weight 1", config.active_model)
        plot_training(output, headless, config.primary_candidate if config.evaluate else None)
        if config.analysis.convergence:
            plot_convergence(output, headless)
    return destination


def _predict(config_path: Path, root: Path, _headless: bool, timestamp: str, trained_bundle: Path | None) -> Path:
    """Read ordered, unlabeled requests and write their stress predictions."""
    config = load_config(config_path, PredictConfig)
    destination = timestamped_output(root / config.output.directory, timestamp)
    bundle_dir = _resolve_bundle(root, config.bundle_directory, trained_bundle)
    with stage_output(destination) as output:
        bundle = load_bundle(bundle_dir)
        requests = read_requests(root / config.request_path)
        result = predict_sequences(bundle, tuple(request.sequence for request in requests))
        pd.DataFrame({
            "request_id": [request.request_id for request in requests],
            "predicted_stress": result.values,
            "target_unit": bundle.target_unit,
        }).to_csv(output / "predictions.csv", index=False)
    LOGGER.info("Predicted %d requests with %s (%s), range %.4g to %.4g", len(requests), bundle_dir,
                bundle.target_unit, result.values.min(), result.values.max())
    return destination


def _resolve_bundle(root: Path, explicit: Path | None, trained_bundle: Path | None) -> Path:
    """Use the configured bundle, or this workflow's training result when requested."""
    if explicit is not None:
        return root / explicit
    if trained_bundle is None:
        raise ValueError("bundle_directory=null; expected training in this workflow or a saved bundle directory.")
    return trained_bundle


def _optimize(config_path: Path, root: Path, headless: bool, timestamp: str, trained_bundle: Path | None) -> Path:
    """Search legal sequences using one loaded predictor and publish a trial ledger."""
    config = load_config(config_path, OptimizeConfig)
    bundle_dir = _resolve_bundle(root, config.bundle_directory, trained_bundle)
    destination = timestamped_output(root / config.output.directory, timestamp)
    with stage_output(destination) as output:
        # Adapt the recipe catalog and validate units before constructing a search context.
        bundle_file = bundle_dir / "bundle.joblib"
        bundle_hash = file_fingerprint(bundle_file)
        bundle = load_bundle(bundle_dir)
        if file_fingerprint(bundle_file) != bundle_hash:
            raise ValueError(f"Model at {bundle_file} changed while loading. Expected a stable deployment bundle.")
        generator = None
        domain = config.domain
        if isinstance(domain, SyntheticDomainConfig):
            generator = load_pseudo_data_config(root / domain.generator_config_path)
            domain = synthetic_domain(generator, domain.parameter_units)
        if domain.parameter_units.model_dump() != bundle.parameter_units:
            raise ValueError(
                f"Search domain units {domain.parameter_units}; expected bundle units {bundle.parameter_units}"
            )
        context = SearchContext(bundle.schema.N_step, bundle.target_unit, bundle.stack.base_names,
                                bundle_hash, bundle.references)
        result = optimize_sequences(domain, config, _bind_predictor(bundle), context)

        # Persist scored trials first so reporting reflects the exact search order and ranking.
        with (output / "optimization_trials.jsonl").open("w", encoding="utf-8") as stream:
            for trial in result.trials:
                stream.write(json.dumps(trial, allow_nan=False) + "\n")
        write_json(output / "optimization.json", result.summary)
        _write_rank_table(result.summary, output)
        if generator is not None:
            from src.analysis.synthetic_audit import audit_latent_results

            write_json(output / "synthetic_audit.json", audit_latent_results(result.summary, generator))
        plot_optimization(output, headless)
    best = result.summary["ranked_results"][0]
    LOGGER.info("%s search with %s: %d evaluated (%s), best stress %.4g vs target %.4g, objective %.4g",
                config.search.mode, bundle_dir, result.summary["scored"], result.summary["stop_reason"],
                best["predicted_stress"], best["target_stress"], best["objective"])
    return destination


def _bind_predictor(bundle: DeploymentBundle) -> Callable[[tuple[ProcessSequence, ...]], tuple[np.ndarray, np.ndarray]]:
    """Adapt shared bundle inference to the optimizer's batched predictor interface."""
    def predict(sequences: tuple[ProcessSequence, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Use one shared feature transform and expose the ordered base predictions."""
        result = predict_sequences(bundle, sequences)
        return result.values, result.base.to_numpy()

    return predict


def _write_rank_table(summary: dict, output: Path) -> None:
    """Provide a small tabular summary linked to complete trial records."""
    columns = ["rank", "trial_index", "predicted_stress", "target_stress", "objective", "base_range"]
    ranked = pd.DataFrame(summary["ranked_results"])
    ranked[columns].to_csv(output / "ranked_candidates.csv", index=False)


def _run_stages(config: PipelineConfig, headless: bool) -> Path:
    """Run enabled stages in dependency order and return the last result."""
    stamp = run_timestamp()
    trained_bundle = None
    stages = (
        ("generate", config.generate, _generate_stage),
        ("train", config.train, _train),
        ("predict", config.predict, _predict),
        ("optimize", config.optimize, _optimize),
    )
    outputs: list[Path] = []
    for name, settings, run in stages:
        if not settings.enabled:
            continue
        assert settings.config is not None  # Validated by StageConfig.
        try:
            path = _REPO_ROOT / settings.config
            output = run(path, _REPO_ROOT, headless, stamp, trained_bundle)
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.error("%s failed: %s", name, error)
            raise
        LOGGER.info("Published %s outputs to %s", name, output)
        if name == "train":
            trained_bundle = output / "model"
        outputs.append(output)
    return outputs[-1]


def main(argv: list[str] | None = None) -> Path:
    """Load the workflow configuration and run its enabled stages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    options = _parse_arguments(argv)
    config_path = (_REPO_ROOT / options.config).resolve()
    LOGGER.info("Loading configuration from %s (headless=%s)", config_path, options.headless)
    config = load_entry_config(config_path)
    if isinstance(config, PseudoDataConfig):
        return _generate(config, Path.cwd())
    return _run_stages(config, options.headless)


if __name__ == "__main__":
    main(sys.argv[1:])
