"""Resolve structured MHub.ai GPU requirements with temporary local fallbacks."""

from enum import Enum


class GPURequirement(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_SUPPORTED = "not_supported"
    UNVERIFIED = "unverified"


# Keep verified fallbacks local until the MHub.ai API publishes gpu_requirement.
_GPU_REQUIREMENT_OVERRIDES = {
    "totalsegmentator": GPURequirement.OPTIONAL,
    "gc_grt123_lung_cancer": GPURequirement.OPTIONAL,
    "mrsegmentator": GPURequirement.REQUIRED,
}


# Accept the API's planned values while remaining tolerant of common separators.
_API_GPU_REQUIREMENTS = {
    "required": GPURequirement.REQUIRED,
    "optional": GPURequirement.OPTIONAL,
    "not_required": GPURequirement.OPTIONAL,
    "not_supported": GPURequirement.NOT_SUPPORTED,
    "unsupported": GPURequirement.NOT_SUPPORTED,
    "unverified": GPURequirement.UNVERIFIED,
    "unknown": GPURequirement.UNVERIFIED,
}


def gpu_requirement_from_model_data(model_data: dict) -> GPURequirement:
    """Prefer structured API metadata and fall back to verified model overrides."""

    api_value = model_data.get("gpu_requirement")
    if isinstance(api_value, str):
        normalized = api_value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _API_GPU_REQUIREMENTS:
            return _API_GPU_REQUIREMENTS[normalized]

    model_name = str(model_data.get("name", "")).strip().lower()
    return _GPU_REQUIREMENT_OVERRIDES.get(model_name, GPURequirement.UNVERIFIED)


def gpu_requirement_display(requirement: GPURequirement) -> tuple[str, str]:
    """Return compact table text and a detailed explanation for a requirement."""

    display = {
        GPURequirement.REQUIRED: ("Yes", "A GPU is required to run this model."),
        GPURequirement.OPTIONAL: ("Optional", "This model can run with or without a GPU."),
        GPURequirement.NOT_SUPPORTED: ("No", "This model does not support GPU acceleration."),
        GPURequirement.UNVERIFIED: ("?", "GPU requirements have not been verified for this model."),
    }
    return display[requirement]
