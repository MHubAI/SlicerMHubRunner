import json
import os
import re
import tempfile
from datetime import datetime
from typing import Any


MANIFEST_FILENAME = "mhubrunner-run.json"
SCHEMA_VERSION = 1


class RunManifestError(ValueError):
    pass


def manifest_path(output_directory: str) -> str:
    return os.path.join(output_directory, MANIFEST_FILENAME)


def new_run_manifest(
    *,
    run_id: str,
    model_name: str,
    model_label: str,
    model_categories: list[str],
    image_name: str,
    image_digest: str,
    input_data: dict[str, Any],
    slicer_session_id: str,
    model_output_directory: str = "outputs",
    created_at: str | None = None,
) -> dict[str, Any]:
    timestamp = created_at or datetime.now().astimezone().isoformat()
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "runId": run_id,
        "createdAt": timestamp,
        "updatedAt": timestamp,
        "slicerSessionId": slicer_session_id,
        "modelOutputDirectory": model_output_directory,
        "model": {
            "name": model_name,
            "label": model_label,
            "categories": list(model_categories),
            "image": image_name,
            "imageDigest": image_digest,
        },
        "input": dict(input_data),
        "status": {
            "state": "running",
            "returnCode": None,
            "timedOut": False,
            "killed": False,
        },
        "outputs": [],
    }
    validate_run_manifest(manifest)
    return manifest


def validate_run_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise RunManifestError("Run manifest must be a JSON object.")
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise RunManifestError(
            f"Unsupported run manifest schema version {manifest.get('schemaVersion')!r}."
        )

    for field in ("runId", "createdAt", "updatedAt", "slicerSessionId"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise RunManifestError(f"Run manifest field {field!r} must be a non-empty string.")

    model_output_directory = manifest.get("modelOutputDirectory")
    if (
        not isinstance(model_output_directory, str)
        or not model_output_directory
        or os.path.isabs(model_output_directory)
        or model_output_directory == ".."
        or model_output_directory.startswith(".." + os.sep)
    ):
        raise RunManifestError("Run manifest modelOutputDirectory must remain inside the run directory.")

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise RunManifestError("Run manifest model must be an object.")
    for field in ("name", "label", "image"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise RunManifestError(f"Run manifest model field {field!r} is invalid.")
    image_digest = model.get("imageDigest")
    if image_digest is not None and not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", image_digest):
        raise RunManifestError("Run manifest model imageDigest must be a SHA-256 digest.")
    if not isinstance(model.get("categories"), list) or not all(
        isinstance(value, str) for value in model["categories"]
    ):
        raise RunManifestError("Run manifest model categories must be an array of strings.")

    input_data = manifest.get("input")
    if not isinstance(input_data, dict):
        raise RunManifestError("Run manifest input must be an object.")
    if not isinstance(input_data.get("nodeId"), str):
        raise RunManifestError("Run manifest input nodeId must be a string.")
    if not isinstance(input_data.get("wasDicom"), bool):
        raise RunManifestError("Run manifest input wasDicom must be a boolean.")
    for field in ("dicomSeriesInstanceUID", "dicomInstanceUIDHash"):
        if input_data.get(field) is not None and not isinstance(input_data[field], str):
            raise RunManifestError(f"Run manifest input field {field!r} is invalid.")
    geometry = input_data.get("geometry")
    if not isinstance(geometry, dict):
        raise RunManifestError("Run manifest input geometry must be an object.")
    dimensions = geometry.get("dimensions")
    matrix = geometry.get("ijkToRAS")
    if not isinstance(dimensions, list) or len(dimensions) != 3 or not all(
        isinstance(value, int) and value >= 0 for value in dimensions
    ):
        raise RunManifestError("Run manifest dimensions must contain three non-negative integers.")
    if not isinstance(matrix, list) or len(matrix) != 4 or not all(
        isinstance(row, list)
        and len(row) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row)
        for row in matrix
    ):
        raise RunManifestError("Run manifest ijkToRAS must be a numeric 4x4 matrix.")

    status = manifest.get("status")
    if not isinstance(status, dict):
        raise RunManifestError("Run manifest status must be an object.")
    if status.get("state") not in {"running", "succeeded", "failed", "timed_out", "cancelled"}:
        raise RunManifestError("Run manifest status state is invalid.")
    if status.get("returnCode") is not None and not isinstance(status["returnCode"], int):
        raise RunManifestError("Run manifest returnCode must be an integer or null.")
    if not isinstance(status.get("timedOut"), bool) or not isinstance(status.get("killed"), bool):
        raise RunManifestError("Run manifest timeout and killed fields must be booleans.")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(path, str) for path in outputs):
        raise RunManifestError("Run manifest outputs must be an array of relative paths.")
    if any(os.path.isabs(path) or path == ".." or path.startswith(".." + os.sep) for path in outputs):
        raise RunManifestError("Run manifest output paths must remain inside the run directory.")
    return manifest


def load_run_manifest(output_directory: str) -> dict[str, Any]:
    path = manifest_path(output_directory)
    try:
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunManifestError(f"Cannot read run manifest {path}: {exc}") from exc
    return validate_run_manifest(manifest)


def write_run_manifest(output_directory: str, manifest: dict[str, Any]) -> str:
    os.makedirs(output_directory, exist_ok=True)
    manifest["updatedAt"] = datetime.now().astimezone().isoformat()
    validate_run_manifest(manifest)
    destination = manifest_path(output_directory)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_directory,
            prefix=".mhubrunner-run-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(manifest, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
    return destination


def finalize_run_manifest(
    output_directory: str,
    *,
    return_code: int,
    timed_out: bool,
    killed: bool,
    output_paths: list[str],
) -> dict[str, Any]:
    manifest = load_run_manifest(output_directory)
    if killed:
        state = "cancelled"
    elif timed_out:
        state = "timed_out"
    elif return_code == 0:
        state = "succeeded"
    else:
        state = "failed"
    manifest["status"] = {
        "state": state,
        "returnCode": int(return_code),
        "timedOut": bool(timed_out),
        "killed": bool(killed),
    }
    manifest["outputs"] = sorted(output_paths)
    write_run_manifest(output_directory, manifest)
    return manifest
