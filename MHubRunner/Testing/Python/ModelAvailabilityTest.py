import unittest
from types import SimpleNamespace

import qt

from MHubRunner import MHubRunnerLogic, MHubRunnerWidget, ModelStatus
from MHubRunnerLib.gpu_requirements import GPURequirement


class ModelAvailabilityTest(unittest.TestCase):
    def _model(self, status):
        # Provide only the model fields exercised by image-availability behavior.
        return SimpleNamespace(
            name="example_model",
            label="Example Model",
            status=status,
            inputs_compatibility=True,
            gpu_requirement=GPURequirement.OPTIONAL,
        )

    def test_run_button_requires_pulled_model(self):
        # Build a minimal widget around the shared Run-button eligibility method.
        widget = MHubRunnerWidget.__new__(MHubRunnerWidget)
        model = self._model(ModelStatus.PULLABLE)
        widget.ui = SimpleNamespace(
            applyButton=SimpleNamespace(enabled=True, toolTip="", text=""),
            cancelButton=SimpleNamespace(enabled=False),
        )
        widget._parameterNode = SimpleNamespace(inputVolume=object())
        widget.getModelFromTableSelection = lambda: model
        widget._modelGpuRequirementMet = lambda selected_model: True
        widget._setButtonTextWithIcon = lambda button, text: setattr(button, "text", text)
        widget.updateLicenseSummary = lambda selected_model: None
        widget._updateMainButtonIcons = lambda: None

        widget._checkCanApply()

        self.assertFalse(widget.ui.applyButton.enabled)
        self.assertEqual(widget.ui.applyButton.text, "Pull Model First")

        model.status = ModelStatus.PULLED
        widget._checkCanApply()
        self.assertTrue(widget.ui.applyButton.enabled)
        self.assertEqual(widget.ui.applyButton.text, "Run Example Model")

    def test_pull_action_updates_shared_model_status(self):
        # Capture the existing asynchronous Docker pull callbacks without starting Docker.
        model = self._model(ModelStatus.PULLABLE)
        callbacks = {}
        logic = SimpleNamespace(_model_cache=[model])

        def update_image(image_name, on_stop, on_progress):
            callbacks["image_name"] = image_name
            callbacks["on_stop"] = on_stop
            callbacks["on_progress"] = on_progress

        logic.update_image = update_image
        widget = MHubRunnerWidget.__new__(MHubRunnerWidget)
        widget.logic = logic
        widget._pendingModelSearchText = ""
        widget._renderFilteredModels = lambda models, text: None
        widget._checkCanApply = lambda: None
        widget._appendLogOutput = lambda output: None
        widget.updateBackendImagesList = lambda: callbacks.__setitem__("images_refreshed", True)

        pull_started = widget.onModelPull(model)

        self.assertTrue(pull_started)
        self.assertEqual(model.status, ModelStatus.PULLING)
        self.assertEqual(callbacks["image_name"], "mhubai/example_model:latest")

        callbacks["on_stop"](0, "pulled", False, False)
        self.assertEqual(model.status, ModelStatus.PULLED)
        self.assertTrue(callbacks["images_refreshed"])

    def test_single_input_compatibility_uses_workflow_not_model_format(self):
        # Accept SMIT-like metadata because the default MHub workflow still consumes DICOM.
        self.assertTrue(
            MHubRunnerLogic._modelInputsCompatible(
                {
                    "inputs": [{"format": "NIFTI"}],
                    "categories": ["Segmentation"],
                }
            )
        )

        # Continue rejecting models that need multiple separately selected inputs.
        self.assertFalse(
            MHubRunnerLogic._modelInputsCompatible(
                {
                    "inputs": [{"format": "DICOM"}, {"format": "DICOM"}],
                    "categories": ["Segmentation"],
                }
            )
        )

    def test_model_sorting_uses_display_and_semantic_values(self):
        # Cover alphabetical model sorting and the operational GPU requirement order.
        models = [
            SimpleNamespace(
                name="optional",
                label="Alpha",
                categories=["Segmentation"],
                modalities=["CT"],
                gpu_requirement=GPURequirement.OPTIONAL,
                commercial_use=True,
            ),
            SimpleNamespace(
                name="required",
                label="Zulu",
                categories=["Prediction"],
                modalities=["MR"],
                gpu_requirement=GPURequirement.REQUIRED,
                commercial_use=False,
            ),
            SimpleNamespace(
                name="recommended",
                label="Bravo",
                categories=["Segmentation"],
                modalities=["CT"],
                gpu_requirement=GPURequirement.RECOMMENDED,
                commercial_use=False,
            ),
        ]

        # Sort labels case-insensitively in the requested direction.
        by_label = MHubRunnerWidget._sortedModels(models, 0, qt.Qt.AscendingOrder)
        self.assertEqual([model.name for model in by_label], ["optional", "recommended", "required"])

        # Sort GPU requirements by operational severity rather than their visible text.
        by_gpu = MHubRunnerWidget._sortedModels(models, 3, qt.Qt.AscendingOrder)
        self.assertEqual([model.name for model in by_gpu], ["required", "recommended", "optional"])

    def test_model_activation_opens_input_section_for_runnable_model(self):
        # Exercise the behavior connected to a model-table double click.
        model = self._model(ModelStatus.PULLED)
        model.inputs_compatibility = True
        expanded_sections = []
        widget = SimpleNamespace(
            ui=SimpleNamespace(inputsCollapsibleButton=object()),
            getModelFromTableSelection=lambda row: model,
            _modelGpuRequirementMet=lambda selected_model: True,
            _promptToPullModel=lambda selected_model: None,
            _expandMainSection=expanded_sections.append,
            _checkCanApply=lambda: None,
        )

        MHubRunnerWidget.onModelActivateFromTable(widget, 0, 0)

        self.assertEqual(expanded_sections, [widget.ui.inputsCollapsibleButton])


if __name__ == "__main__":
    unittest.main()
