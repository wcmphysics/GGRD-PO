"""Compose the training, prediction, and optimization CLI stages."""

import argparse
import csv
from collections.abc import Callable
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.analysis.ablation import write_transition_ablation
from src.analysis.exploration import write_exploration
from src.config.pipeline_schema import OptimizeConfig, PredictConfig, SyntheticDomainConfig, TrainConfig, load_config
from src.config.pseudo_data_schema import load_pseudo_data_config
from src.dataset.contracts import ProcessSequence, WaferSample
from src.dataset.history import read_history
from src.dataset.requests import read_requests
from src.feature.wide import FeatureSchema, build_training_batch
from src.inference.bundle import DeploymentBundle, load_bundle, save_bundle
from src.inference.predict import predict_sequences
from src.optimize.domain import synthetic_domain
from src.optimize.search import SearchContext, optimize_sequences
from src.trainer.reporting import training_manifest, write_evaluation, write_trace_reports
from src.trainer.splits import preflight_training, split_records
from src.trainer.stacking import evaluate_stack, fit_deployment
from src.utility.run_output import file_fingerprint, stage_output, write_json
from src.visual.pipeline_plots import plot_optimization, plot_training

LOGGER = logging.getLogger(__name__)


def _record_count(path: Path) -> int:
    """Count CSV records without counting embedded line endings as records."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        return sum(bool(row) for row in csv.reader(source)) - 1


def _group_references(samples: tuple[WaferSample, ...]) -> dict[str, ProcessSequence]:
    """Keep one already validated, unimputed history per group."""
    groups = {sample.group: sample.sequence for sample in samples}
    return dict(sorted(groups.items()))


def _train(config_path: Path, root: Path, headless: bool) -> Path:
    """Evaluate grouped stacks, fit deployment, and publish their artifacts together."""
    config_hash = file_fingerprint(config_path)
    config = load_config(config_path, TrainConfig)
    if file_fingerprint(config_path) != config_hash:
        raise ValueError(f"Configuration {config_path} changed while reading it. Expected stable settings.")
    destination = root / config.output.directory
    source = root / config.data.csv_path
    with stage_output(destination) as output:
        # Canonical data and all nested group plans exist before any estimator sees a row.
        hashes = {"source_csv": file_fingerprint(source), "train_config": config_hash}
        samples = read_history(source)
        schema = FeatureSchema(config.features.N_step)
        batch = build_training_batch(samples, schema)
        plan = preflight_training(batch, config)
        write_exploration(samples, _record_count(source), output / "exploration")
        if file_fingerprint(source) != hashes["source_csv"]:
            raise ValueError(f"Wafer history {source} changed while reading it. Expected a stable training input.")

        # Outer evaluation and ablation reuse one plan. Final training uses its separate plan.
        full = evaluate_stack(batch, schema, config, plan)
        evaluations = [full]
        write_evaluation(full, output, config.cv.mape_min_abs_y)
        if config.analysis.transition_ablation:
            ablated = evaluate_stack(batch, schema.without_transitions(), config, plan, variant="transition_ablation")
            write_transition_ablation(full.predictions, ablated.predictions, output / "analysis")
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
        plot_training(output, headless)
    return destination


def _predict(config_path: Path, root: Path, _headless: bool) -> Path:
    """Read ordered, unlabeled requests and write their stress predictions."""
    config = load_config(config_path, PredictConfig)
    destination = root / config.output.directory
    with stage_output(destination) as output:
        bundle = load_bundle(root / config.bundle_directory)
        requests = read_requests(root / config.request_path)
        result = predict_sequences(bundle, tuple(request.sequence for request in requests))
        pd.DataFrame({
            "request_id": [request.request_id for request in requests],
            "predicted_stress": result.values,
            "target_unit": bundle.target_unit,
        }).to_csv(output / "predictions.csv", index=False)
    return destination


def _optimize(config_path: Path, root: Path, headless: bool) -> Path:
    """Search legal sequences using one loaded predictor and publish a trial ledger."""
    config = load_config(config_path, OptimizeConfig)
    bundle_dir = root / config.bundle_directory
    destination = root / config.output.directory
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


def run_command(arguments: list[str], root: Path) -> Path:
    """Parse a stage-specific config flag and run the selected stage."""
    command, *options = arguments
    parser = argparse.ArgumentParser(prog=f"src/main.py {command}")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parsed = parser.parse_args(options)
    stages = {"train": _train, "predict": _predict, "optimize": _optimize}
    try:
        saved = stages[command](root / parsed.config, root, parsed.headless)
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("%s failed: %s", command, error)
        raise
    LOGGER.info("Published %s outputs to %s", command, saved)
    return saved
