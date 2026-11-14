"""Persist and validate trusted deployment bundles with compatibility metadata."""

from dataclasses import dataclass
from hashlib import file_digest
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from typing import BinaryIO

import joblib
import numpy as np
from sklearn.utils.validation import check_is_fitted

from src.dataset.contracts import ProcessSequence
from src.feature.wide import FeatureSchema
from src.model.stack import FittedStack


BUNDLE_FORMAT = 1
FEATURE_SCHEMA_VERSION = 1
_BOOSTER_PACKAGES = {"lightgbm", "catboost", "xgboost"}


@dataclass(frozen=True)
class DeploymentBundle:
    """Trusted model payload used for unlabeled prediction and support diagnostics."""

    stack: FittedStack
    schema: FeatureSchema
    references: dict[str, ProcessSequence]
    config: dict
    source_fingerprint: str
    target_unit: str
    parameter_units: dict[str, str]


def _runtime(extra: set[str]) -> dict:
    """Capture package versions needed to deserialize the trained object graph."""
    packages = {"numpy", "pandas", "scipy", "scikit-learn", "joblib", "pydantic"} | extra
    return {"python": platform.python_version(),
            "packages": {package: version(package) for package in sorted(packages)}}


def _sha256(stream: BinaryIO) -> str:
    """Fingerprint an open payload and rewind for deserialization."""
    digest = file_digest(stream, "sha256").hexdigest()
    stream.seek(0)
    return digest


def _validate_stack(bundle: DeploymentBundle) -> None:
    """Verify compatibility and invariants of a trusted fitted stack."""
    stack = bundle.stack
    if not isinstance(stack, FittedStack) or stack.schema != bundle.schema:
        raise ValueError("Invalid model payload stack schema; expected the deployment feature schema.")
    count = len(stack.base_names)
    if (not count or len(set(stack.base_names)) != count
            or len(stack.pipelines) != count or len(stack.parameters) != count):
        raise ValueError(f"Invalid model base order {stack.base_names}; expected distinct, aligned fitted bases.")
    weights, tolerance = stack.weights, stack.constraint_tolerance
    if (weights.shape != (count,) or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isfinite(tolerance) or tolerance <= 0 or abs(weights.sum() - 1) > tolerance):
        raise ValueError(f"Invalid weights {weights}; expected {count} finite nonnegative weights summing to one.")
    for name, pipeline in zip(stack.base_names, stack.pipelines, strict=True):
        check_is_fitted(pipeline)
        if tuple(pipeline.feature_names_in_) != bundle.schema.columns:
            raise ValueError(f"Fitted base {name} schema differs; expected {bundle.schema.columns}.")


def _validate_bundle(bundle: DeploymentBundle) -> None:
    """Check bundle units and support fields before saving or use."""
    if not isinstance(bundle, DeploymentBundle) or not isinstance(bundle.schema, FeatureSchema):
        raise ValueError(f"Invalid deployment payload {type(bundle).__name__}; expected a versioned deployment bundle.")
    if not bundle.target_unit or set(bundle.parameter_units) != {"time", "temperature", "thickness"}:
        raise ValueError(f"Invalid units {bundle.parameter_units}; expected stress and three parameter units.")
    if any(not unit for unit in bundle.parameter_units.values()) or not bundle.source_fingerprint:
        raise ValueError("Invalid deployment provenance or units; expected a data fingerprint and nonempty unit names.")
    if not bundle.references or any(not group or not isinstance(seq, ProcessSequence)
                                    or len(seq.steps) > bundle.schema.N_step
                                    for group, seq in bundle.references.items()):
        raise ValueError("Invalid support references; expected canonical group histories within model capacity.")
    _validate_stack(bundle)


def _model_metadata(bundle: DeploymentBundle, digest: str) -> dict:
    """Build a sidecar that can be safely inspected before loading any model code."""
    modules = {type(pipeline.named_steps["estimator"]).__module__.split(".")[0] for pipeline in bundle.stack.pipelines}
    runtime = _runtime(modules & _BOOSTER_PACKAGES)
    return {"bundle_format": BUNDLE_FORMAT, "feature_schema_version": FEATURE_SCHEMA_VERSION, "runtime": runtime,
            "payload_sha256": digest, "source_fingerprint": bundle.source_fingerprint,
            "features": [list(pair) for pair in zip(bundle.schema.columns, bundle.schema.dtypes)],
            "base_order": list(bundle.stack.base_names), "parameters": bundle.stack.parameters,
            "meta_weights": bundle.stack.weights.tolist(), "target_unit": bundle.target_unit,
            "parameter_units": bundle.parameter_units}


def save_bundle(bundle: DeploymentBundle, destination: Path) -> Path:
    """Save the payload and its fingerprinted metadata in the stage directory."""
    _validate_bundle(bundle)
    destination.mkdir(exist_ok=False)
    path = destination / "bundle.joblib"
    joblib.dump(bundle, path)
    with path.open("rb") as stream:
        metadata = _model_metadata(bundle, _sha256(stream))
    (destination / "metadata.json").write_text(json.dumps(metadata, allow_nan=False, indent=2), encoding="utf-8")
    return path


def _load_metadata(source: Path, digest: str) -> dict:
    """Verify format, runtime compatibility, and hash before deserializing a trusted bundle."""
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Bundle metadata at {source} is {type(metadata).__name__}; expected an object.")
    recorded = metadata.get("runtime")
    packages = recorded.get("packages") if isinstance(recorded, dict) else None
    try:
        runtime = _runtime(set(packages) & _BOOSTER_PACKAGES if isinstance(packages, dict) else set())
    except PackageNotFoundError as error:
        raise ValueError(f"Bundle at {source} requires the installed dependency {error.name!r}.") from error
    checks = {"bundle_format": BUNDLE_FORMAT, "feature_schema_version": FEATURE_SCHEMA_VERSION,
              "runtime": runtime, "payload_sha256": digest}
    for field, expected in checks.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Bundle metadata {field}={metadata.get(field)!r}; expected {expected!r} at {source}.")
    return metadata


def load_bundle(source: Path) -> DeploymentBundle:
    """Load a trusted bundle after confirming compatibility and integrity."""
    path = source / "bundle.joblib"
    with path.open("rb") as stream:
        digest = _sha256(stream)
        metadata = _load_metadata(source, digest)
        bundle = joblib.load(stream)
    _validate_bundle(bundle)
    expected_metadata = _model_metadata(bundle, digest)
    if metadata != json.loads(json.dumps(expected_metadata, allow_nan=False)):
        raise ValueError(f"Bundle data at {source} disagree; expected the metadata and fitted payload to match.")
    return bundle
