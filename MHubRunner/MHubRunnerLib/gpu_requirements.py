"""Resolve structured MHub.ai GPU requirements with temporary local fallbacks."""

from enum import Enum


class GPURequirement(Enum):
    REQUIRED = "required"  # Fails without a GPU.
    RECOMMENDED = "recommended"  # CPU may be very slow or is not verified.
    OPTIONAL = "optional"  # GPU improves performance, but CPU is supported.
    NOT_SUPPORTED = "not_supported"  # GPU is not supported; CPU only.
    UNVERIFIED = "unverified"  # GPU requirements have not been verified.


# Keep verified fallbacks local until the MHub.ai API publishes gpu_requirement.
_GPU_REQUIREMENT_OVERRIDES = {
    # Block models known to fail without GPU execution.
    "casust": GPURequirement.REQUIRED,
    "mrsegmentator": GPURequirement.REQUIRED,
    "msk_smit_lung_gtv": GPURequirement.REQUIRED,

    # Allow models with verified CPU and GPU execution without a slow-CPU warning.
    "totalsegmentator": GPURequirement.OPTIONAL,
    "gc_grt123_lung_cancer": GPURequirement.OPTIONAL,

    # Warn before CPU execution for slow or not-yet-verified nnU-Net CPU paths.
    "bamf_nnunet_ct_kidney": GPURequirement.RECOMMENDED,
    "bamf_nnunet_ct_liver": GPURequirement.RECOMMENDED,
    "bamf_nnunet_mr_liver": GPURequirement.RECOMMENDED,
    "bamf_nnunet_mr_prostate": GPURequirement.RECOMMENDED,
    "gc_nnunet_pancreas": GPURequirement.RECOMMENDED,
    "nnunet_liver": GPURequirement.RECOMMENDED,
    "nnunet_pancreas": GPURequirement.RECOMMENDED,
    "nnunet_prostate_task24": GPURequirement.RECOMMENDED,
    "nnunet_prostate_zonal_task05": GPURequirement.RECOMMENDED,
    "nnunet_segthor": GPURequirement.RECOMMENDED,

    # Keep CPU-only models from advertising unsupported GPU execution.
    "pyradiomics": GPURequirement.NOT_SUPPORTED,
}


# Accept the API's planned values while remaining tolerant of common separators.
_API_GPU_REQUIREMENTS = {
    "required": GPURequirement.REQUIRED,
    "recommended": GPURequirement.RECOMMENDED,
    "preferred": GPURequirement.RECOMMENDED,
    "slow_cpu": GPURequirement.RECOMMENDED,
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
        GPURequirement.REQUIRED: ("Required", "A GPU is required to run this model."),
        GPURequirement.RECOMMENDED: (
            "Recommended",
            "CPU execution may be substantially slower or has not been verified for this model.",
        ),
        GPURequirement.OPTIONAL: ("Optional", "This model can run with or without a GPU."),
        GPURequirement.NOT_SUPPORTED: ("No", "This model does not support GPU acceleration."),
        GPURequirement.UNVERIFIED: ("Unknown", "GPU requirements have not been verified for this model."),
    }
    return display[requirement]
