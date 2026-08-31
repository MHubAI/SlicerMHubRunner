import unittest

import qt

from MHubRunner import MHubRunnerWidget, Model, ModelStatus
from MHubRunnerLib.gpu_requirements import (
    GPURequirement,
    gpu_requirement_from_model_data,
)


class GPURequirementsTest(unittest.TestCase):
    def test_api_metadata_takes_precedence_over_local_registry(self):
        # Let the future API replace a temporary local decision without an extension update.
        requirement = gpu_requirement_from_model_data(
            {"name": "totalsegmentator", "gpu_requirement": "optional"}
        )

        self.assertEqual(requirement, GPURequirement.OPTIONAL)

    def test_verified_registry_defaults(self):
        # Cover the three initial verified fallbacks requested for this release.
        self.assertEqual(
            gpu_requirement_from_model_data({"name": "totalsegmentator"}),
            GPURequirement.OPTIONAL,
        )
        self.assertEqual(
            gpu_requirement_from_model_data({"name": "gc_grt123_lung_cancer"}),
            GPURequirement.OPTIONAL,
        )
        self.assertEqual(
            gpu_requirement_from_model_data({"name": "mrsegmentator"}),
            GPURequirement.REQUIRED,
        )

    def test_unknown_models_remain_unverified(self):
        # Never infer GPU requirements from unrelated model metadata.
        self.assertEqual(
            gpu_requirement_from_model_data({"name": "future_model"}),
            GPURequirement.UNVERIFIED,
        )

    def test_only_required_models_are_blocked_without_gpu(self):
        # Exercise the shared eligibility rule independently of the table renderer.
        widget = MHubRunnerWidget.__new__(MHubRunnerWidget)
        widget.ui = type("Ui", (), {})()
        widget.ui.chkGpuEnabled = type("CheckBox", (), {"checked": False})()
        widget.ui.lstHostGpu = qt.QListWidget()

        required_model = type("Model", (), {"gpu_requirement": GPURequirement.REQUIRED})()
        unknown_model = type("Model", (), {"gpu_requirement": GPURequirement.UNVERIFIED})()
        unsupported_model = type("Model", (), {"gpu_requirement": GPURequirement.NOT_SUPPORTED})()

        self.assertFalse(widget._modelGpuRequirementMet(required_model))
        self.assertTrue(widget._modelGpuRequirementMet(unknown_model))
        self.assertTrue(widget._modelGpuRequirementMet(unsupported_model))

        widget.ui.lstHostGpu.addItem("GPU 0")
        widget.ui.chkGpuEnabled.checked = True
        self.assertTrue(widget._modelGpuRequirementMet(required_model))

    def test_model_table_displays_gpu_requirement_column(self):
        # Render a real Qt table to catch column-index and compact-label regressions.
        widget = MHubRunnerWidget.__new__(MHubRunnerWidget)
        widget.ui = type("Ui", (), {})()
        widget.ui.tblModelList = qt.QTableWidget()
        widget.ui.chkGpuEnabled = type("CheckBox", (), {"checked": False})()
        widget.ui.lstHostGpu = qt.QListWidget()
        widget._themeIcon = lambda *args: qt.QIcon()
        widget.updateLicenseSummary = lambda *args: None
        model = Model(
            id="test",
            name="mrsegmentator",
            label="MRSegmentator",
            description="",
            modalities=["CT"],
            categories=["Segmentation"],
            roi=[],
            cite="",
            license_model="",
            license_weights="",
            commercial_use=False,
            inputs=["CT"],
            inputs_compatibility=True,
            gpu_requirement=GPURequirement.REQUIRED,
            status=ModelStatus.PULLABLE,
        )

        widget.renderModelTable([model])

        self.assertEqual(widget.ui.tblModelList.columnCount, 6)
        self.assertEqual(widget.ui.tblModelList.horizontalHeaderItem(3).text(), "GPU")
        self.assertEqual(widget.ui.tblModelList.item(0, 3).text(), "Yes")
        self.assertIn("requires GPU execution", widget.ui.tblModelList.item(0, 0).toolTip())

        widget.ui.tblModelList.deleteLater()
        widget.ui.lstHostGpu.deleteLater()


if __name__ == "__main__":
    unittest.main()
