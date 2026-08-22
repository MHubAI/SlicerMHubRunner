# Model handlers

MHubRunner selects handlers by the exact MHub model name. Models without
an explicit handler use `GenericModelHandler`, which preserves generic
DICOM SEG, JSON, and CSV handling.

To add a model-specific renderer:

1. Add a module containing a `ModelHandler` subclass.
2. Set `model_names` to exact names returned by the MHub API.
3. Override `resolve_inputs` when the generic single-volume input plan is not
   sufficient. Return node references and destination subdirectories; the core
   performs the actual export.
4. Parse output files in `build_output_plan` and return an `OutputPlan` containing
   tables, markups, segmentations, and warnings.
5. Register the class in `registry.py`.
6. Add the module to `MODULE_PYTHON_SCRIPTS` in `MHubRunner/CMakeLists.txt`.
7. Add parser tests and, for MRML output, a Slicer runtime test.

Handlers produce a plan but do not run Docker, write to the DICOM database, or
directly manipulate the MRML scene. Those operations remain owned by
`MHubRunnerLogic` so output-handling settings and error behavior stay
consistent across models.

Spatial outputs must state their source coordinate system. For physical LPS
positions, include the reported image dimensions, voxel spacing, origin, and
orientation. MHubRunner creates markups only after that geometry matches the
run's input volume. Unknown coordinate systems should produce tables and a
warning, not guessed annotations.

Output values must be transcribed as supplied. A handler must not invent
clinical categories, thresholds, anatomical labels, or recommendations.
