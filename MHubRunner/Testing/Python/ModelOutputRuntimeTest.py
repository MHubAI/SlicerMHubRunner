import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import qt
import slicer
import vtk

from MHubRunner import MHubRunnerLogic, MHubRunnerWidget, Model, ModelStatus
from MHubRunnerModelHandlers import MarkupOutput, MarkupPoint
from MHubRunnerModelHandlers.run_manifest import load_run_manifest, write_run_manifest


class ModelOutputRuntimeTest(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear()

    @staticmethod
    def _logic_without_initialization():
        return MHubRunnerLogic.__new__(MHubRunnerLogic)

    def test_extension_build_footer(self):
        label = qt.QLabel()
        widget = SimpleNamespace(ui=SimpleNamespace(lblExtensionBuildInfo=label))
        build_info = SimpleNamespace(
            EXTENSION_VERSION="9.8.7-test",
            BUILD_REVISION="abcdef1234567890",
            BUILD_REVISION_SHORT="abcdef123456",
        )

        # Verify that the footer renders metadata supplied by the generated build module.
        with patch.dict(sys.modules, {"MHubRunnerLib.build_info": build_info}):
            MHubRunnerWidget._updateExtensionBuildInfo(widget)

        self.assertEqual(label.text, "MHubRunner 9.8.7-test \u00b7 build abcdef123456")
        self.assertIn("Extension version: 9.8.7-test", label.toolTip)
        self.assertIn("Git revision: abcdef1234567890", label.toolTip)

    def test_extension_build_footer_without_generated_metadata(self):
        label = qt.QLabel()
        widget = SimpleNamespace(ui=SimpleNamespace(lblExtensionBuildInfo=label))

        # Identify direct source loading without duplicating the CMake release version.
        with patch.dict(sys.modules, {"MHubRunnerLib.build_info": None}):
            MHubRunnerWidget._updateExtensionBuildInfo(widget)

        self.assertEqual(label.text, "MHubRunner development source \u00b7 build unknown")
        self.assertIn("Extension version: development source", label.toolTip)
        self.assertIn("Git revision: unknown", label.toolTip)

    def test_output_display_path_handles_symlinked_run_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            real_run = os.path.join(temporary_directory, "real-run")
            alias_run = os.path.join(temporary_directory, "alias-run")
            outputs_directory = os.path.join(real_run, "outputs")
            os.makedirs(outputs_directory)
            output_file = os.path.join(outputs_directory, "result.json")
            with open(output_file, "w", encoding="utf-8") as stream:
                stream.write("{}")
            try:
                os.symlink(real_run, alias_run)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Symbolic links are unavailable: {exc}")

            # Resolve both sides before relativizing, as required for /tmp on macOS.
            display_path = MHubRunnerWidget._relativeOutputDisplayPath(
                os.path.realpath(output_file), alias_run
            )

        self.assertEqual(display_path, os.path.join("outputs", "result.json"))

    def test_default_runs_directory_uses_persistent_application_data(self):
        runs_directory = MHubRunnerWidget._defaultRunsDirectory()

        # Keep run history under application data instead of the system temporary directory.
        self.assertTrue(os.path.isabs(runs_directory))
        self.assertEqual(
            os.path.normpath(runs_directory).split(os.sep)[-2:],
            ["MHubRunner", "runs"],
        )
        self.assertNotEqual(
            os.path.normpath(runs_directory),
            os.path.join(tempfile.gettempdir(), "mhub_slicer_extension", "runs"),
        )

    def test_main_workflow_sections_are_exclusive(self):
        # Load the real UI so this test covers its initial collapse and page visibility states.
        ui_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "Resources",
                "UI",
                "MHubRunner.ui",
            )
        )
        ui_widget = slicer.util.loadUI(ui_path)
        ui = slicer.util.childWidgetVariables(ui_widget)
        widget = MHubRunnerWidget.__new__(MHubRunnerWidget)
        widget.ui = ui
        widget._mainSectionSignalsWired = False
        widget._dockerSetupDismissed = False
        widget._updateDockerSetupLogo = lambda: None

        # Verify the compact initial workflow state and wire manual section changes.
        self.assertFalse(ui.outputsCollapsibleButton.collapsed)
        self.assertTrue(ui.inputsCollapsibleButton.collapsed)
        self.assertTrue(ui.outputCollapsibleButton.collapsed)
        self.assertEqual(
            ui.outputTableSelector.sizePolicy.horizontalPolicy(),
            qt.QSizePolicy.Ignored,
        )
        self.assertEqual(
            ui.cmbSelectRunOutput.sizePolicy.horizontalPolicy(),
            qt.QSizePolicy.Ignored,
        )
        self.assertEqual(ui.cmdRefreshRuns.text, "Refresh Runs")
        widget._setupMainSectionCollapse()

        # Verify programmatic navigation closes the previously open workflow section.
        widget._expandMainSection(ui.outputCollapsibleButton)
        self.assertTrue(ui.outputsCollapsibleButton.collapsed)
        self.assertTrue(ui.inputsCollapsibleButton.collapsed)
        self.assertFalse(ui.outputCollapsibleButton.collapsed)

        # Verify manual expansion uses the same exclusive-section behavior.
        ui.inputsCollapsibleButton.collapsed = False
        slicer.app.processEvents()
        self.assertFalse(ui.inputsCollapsibleButton.collapsed)
        self.assertTrue(ui.outputCollapsibleButton.collapsed)

        # Verify hidden workflow/setup pages do not coexist in the parent layout.
        widget.showDockerSetupScreen()
        self.assertTrue(ui.mainPanel.isHidden())
        self.assertFalse(ui.dockerSetupPanel.isHidden())
        widget.hideDockerSetupScreen()
        self.assertFalse(ui.mainPanel.isHidden())
        self.assertTrue(ui.dockerSetupPanel.isHidden())

        # Release native Qt/MRML widgets before Slicer's leak check runs at shutdown.
        ui_widget.hide()
        ui_widget.setMRMLScene(None)
        ui_widget.deleteLater()
        slicer.app.processEvents()

    def test_docker_image_digest_is_read_from_local_image(self):
        logic = self._logic_without_initialization()
        logic._executables = {"docker": "/mock/docker"}
        expected_digest = "sha256:" + "a" * 64

        # Treat the mock Docker path as executable before inspecting the image digest.
        with patch("shutil.which", return_value="/mock/docker"), patch("subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout="sha256:" + "A" * 64)
            digest = logic.getDockerImageDigest("mhubai/example:latest")

        self.assertEqual(digest, expected_digest)
        self.assertEqual(
            run.call_args.args[0],
            [
                "/mock/docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                "mhubai/example:latest",
            ],
        )

    def test_dicom_identity_accepts_series_uid_without_instance_uid_attribute(self):
        logic = self._logic_without_initialization()
        node = SimpleNamespace(GetAttribute=lambda name: None)

        # Preserve DICOM provenance when only the subject-hierarchy series UID survives.
        with patch.object(logic, "_seriesInstanceUIDForNode", return_value="1.2.3"):
            identity = logic.dicomInputIdentity(node)

        self.assertTrue(identity["wasDicom"])
        self.assertEqual(identity["dicomSeriesInstanceUID"], "1.2.3")
        self.assertIsNone(identity["dicomInstanceUIDHash"])

    def test_docker_detection_uses_path_on_linux(self):
        logic = self._logic_without_initialization()
        logic._executables = {}

        # Simulate a Linux installation where Docker is available through PATH.
        with patch("platform.system", return_value="Linux"), patch(
            "shutil.which", return_value="/usr/bin/docker"
        ) as which:
            executable = logic.getDockerExecutable()

        self.assertEqual(executable, "/usr/bin/docker")
        which.assert_called_once_with("docker")

    def test_docker_detection_uses_macos_desktop_fallback(self):
        logic = self._logic_without_initialization()
        logic._executables = {}
        docker_desktop_path = "/Applications/Docker.app/Contents/Resources/bin/docker"

        # Simulate Docker Desktop being absent from PATH but present in its application bundle.
        def resolve_macos_executable(candidate):
            return docker_desktop_path if candidate == docker_desktop_path else None

        with patch("platform.system", return_value="Darwin"), patch(
            "shutil.which", side_effect=resolve_macos_executable
        ):
            executable = logic.getDockerExecutable()

        self.assertEqual(executable, docker_desktop_path)

    def test_docker_detection_uses_windows_desktop_executable(self):
        logic = self._logic_without_initialization()
        logic._executables = {}
        docker_desktop_path = os.path.join(
            r"C:\Program Files", "Docker", "Docker", "resources", "bin", "docker.exe"
        )

        # Simulate Docker Desktop being absent from PATH but installed in Program Files.
        def resolve_windows_executable(candidate):
            return docker_desktop_path if candidate == docker_desktop_path else None

        with patch.dict(os.environ, {"ProgramFiles": r"C:\Program Files"}, clear=False), patch(
            "platform.system", return_value="Windows"
        ), patch("shutil.which", side_effect=resolve_windows_executable):
            executable = logic.getDockerExecutable()

        self.assertEqual(executable, docker_desktop_path)

    def test_docker_detection_replaces_invalid_cached_path(self):
        logic = self._logic_without_initialization()
        logic._executables = {"docker": "/obsolete/docker-directory"}

        # Reject the stale configured value and fall back to the executable on PATH.
        def resolve_executable(candidate):
            return "/usr/local/bin/docker" if candidate == "docker" else None

        with patch("platform.system", return_value="Darwin"), patch(
            "shutil.which", side_effect=resolve_executable
        ):
            executable = logic.getDockerExecutable()

        self.assertEqual(executable, "/usr/local/bin/docker")
        self.assertEqual(logic._executables["docker"], "/usr/local/bin/docker")

    def test_docker_information_checks_resolved_executable_version(self):
        logic = self._logic_without_initialization()
        logic._executables = {"docker": "/resolved/docker"}

        # Simulate a validated Docker CLI responding to the version check.
        with patch("shutil.which", return_value="/resolved/docker"), patch(
            "subprocess.run"
        ) as run:
            run.return_value = SimpleNamespace(stdout=b"Docker version 27.0.0\n")
            information = logic.getDockerInformation()

        self.assertTrue(information.available)
        self.assertEqual(information.version, "Docker version 27.0.0\n")
        self.assertEqual(run.call_args.args[0], ["/resolved/docker", "--version"])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_matching_lps_geometry_creates_ras_fiducial(self):
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Input")
        image = vtk.vtkImageData()
        image.SetDimensions(512, 512, 133)
        image.AllocateScalars(vtk.VTK_SHORT, 1)
        volume.SetAndObserveImageData(image)

        ijk_to_ras = vtk.vtkMatrix4x4()
        ijk_to_ras.Identity()
        values = (
            (-0.703125, 0.0, 0.0, 166.0),
            (0.0, -0.703125, 0.0, 171.699997),
            (0.0, 0.0, 2.5, -340.0),
        )
        for row, row_values in enumerate(values):
            for column, value in enumerate(row_values):
                ijk_to_ras.SetElement(row, column, value)
        volume.SetIJKToRASMatrix(ijk_to_ras)

        markup = MarkupOutput(
            name="GRT123 Findings",
            points=[
                MarkupPoint(
                    label="Finding 0",
                    position_lps=(56.0, 86.300003, -112.0),
                    description="Detection probability: 0.9",
                )
            ],
            image_geometry={
                "dimensions": [512, 512, 133],
                "voxelsize": [0.703125, 0.703125, 2.5],
                "origin": [-166.0, -171.699997, -340.0],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            source_file="findings.json",
        )
        model = Model(
            id="test",
            name="gc_grt123_lung_cancer",
            label="GRT123",
            description="",
            modalities=["CT"],
            categories=["Prediction"],
            roi=[],
            cite="",
            license_model="MIT",
            license_weights="MIT",
            commercial_use=True,
            inputs=["Chest CT"],
            inputs_compatibility=True,
            status=ModelStatus.PULLED,
        )

        logic = self._logic_without_initialization()
        markups_node = logic._createMarkupFromOutput(model, volume, markup)

        self.assertIsNotNone(markups_node)
        self.assertEqual(markups_node.GetNumberOfControlPoints(), 1)
        position = [0.0, 0.0, 0.0]
        markups_node.GetNthControlPointPosition(0, position)
        self.assertAlmostEqual(position[0], -56.0)
        self.assertAlmostEqual(position[1], -86.300003)
        self.assertAlmostEqual(position[2], -112.0)
        self.assertEqual(markups_node.GetNthControlPointLabel(0), "Finding 0")

        mismatch_geometry = dict(markup.image_geometry)
        mismatch_geometry["origin"] = [-100.0, -100.0, -100.0]
        mismatch_markup = MarkupOutput(
            name="Mismatched Findings",
            points=markup.points,
            image_geometry=mismatch_geometry,
            source_file="mismatched-findings.json",
        )
        self.assertIsNone(logic._createMarkupFromOutput(model, volume, mismatch_markup))

    def test_stored_run_resolves_same_session_input(self):
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Input")
        image = vtk.vtkImageData()
        image.SetDimensions(2, 2, 2)
        image.AllocateScalars(vtk.VTK_SHORT, 1)
        volume.SetAndObserveImageData(image)
        ijk_to_ras = vtk.vtkMatrix4x4()
        ijk_to_ras.Identity()
        ijk_to_ras.SetElement(0, 0, -1.0)
        ijk_to_ras.SetElement(1, 1, -1.0)
        volume.SetIJKToRASMatrix(ijk_to_ras)

        model = Model(
            id="test",
            name="gc_grt123_lung_cancer",
            label="GRT123",
            description="",
            modalities=["CT"],
            categories=["Prediction"],
            roi=[],
            cite="",
            license_model="MIT",
            license_weights="MIT",
            commercial_use=True,
            inputs=["Chest CT"],
            inputs_compatibility=True,
            status=ModelStatus.PULLED,
        )
        result_payload = {
            "lungcad": {
                "revision": "test",
                "name": "grt123",
                "datetimeofexecution": "08/22/2026 12:00:00",
                "coordinatesystem": "World",
                "computationtimeinseconds": 1.0,
            },
            "imageinfo": {
                "dimensions": [2, 2, 2],
                "voxelsize": [1.0, 1.0, 1.0],
                "origin": [0.0, 0.0, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "findings": [
                {
                    "id": 0,
                    "x": 1.0,
                    "y": 1.0,
                    "z": 1.0,
                    "probability": 0.5,
                    "cancerprobability": 0.25,
                },
                {
                    "id": 1,
                    "x": None,
                    "y": None,
                    "z": None,
                    "probability": 0.1,
                    "cancerprobability": 0.05,
                },
            ],
            "cancerinfo": {
                "casecancerprobability": 0.25,
                "referencenoduleids": [0],
            },
        }

        logic = self._logic_without_initialization()
        with tempfile.TemporaryDirectory() as output_directory:
            model_output_directory = os.path.join(output_directory, "outputs")
            os.makedirs(model_output_directory)
            result_path = os.path.join(
                model_output_directory, "gc_grt123_lung_cancer_findings.json"
            )
            with open(result_path, "w", encoding="utf-8") as stream:
                json.dump(result_payload, stream)
            logic.createRunManifest(
                run_id="26.08.22-12.00.00_gc_grt123_lung_cancer",
                model=model,
                input_node=volume,
                output_dir=output_directory,
                image_digest="sha256:" + "a" * 64,
            )
            logic.finalizeRunManifest(output_directory, 0, False, False)

            loaded = logic.loadStoredRun(output_directory)
            self.assertIs(loaded["inputNode"], volume)
            self.assertIsNone(loaded["annotationWarning"])
            self.assertEqual(len(loaded["plan"].tables), 2)
            self.assertEqual(len(loaded["plan"].markups), 1)

            tables = slicer.util.getNodesByClass("vtkMRMLTableNode")
            markups = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
            self.assertEqual(len(tables), 2)
            self.assertEqual(len(markups), 1)
            findings_table = next(
                node
                for node in tables
                if node.GetAttribute("MHubRunner.OutputIdentity").endswith(":findings")
            )
            findings_markup = markups[0]
            self.assertEqual(
                findings_table.GetNodeReferenceID("MHubRunner.LinkedMarkup"),
                findings_markup.GetID(),
            )
            self.assertEqual(
                findings_markup.GetNodeReferenceID("MHubRunner.LinkedTable"),
                findings_table.GetID(),
            )
            self.assertEqual(
                json.loads(findings_table.GetAttribute("MHubRunner.RowKeys")),
                ["finding:0", "finding:1"],
            )
            self.assertEqual(
                json.loads(findings_markup.GetAttribute("MHubRunner.ControlPointKeys")),
                ["finding:0"],
            )
            linked_markup, linked_point_index = (
                MHubRunnerWidget._linkedMarkupPointForTableRow(findings_table, 0)
            )
            self.assertIs(linked_markup, findings_markup)
            self.assertEqual(linked_point_index, 0)
            MHubRunnerWidget._selectMarkupControlPoint(
                linked_markup, linked_point_index
            )
            self.assertTrue(findings_markup.GetNthControlPointSelected(0))
            unpositioned_markup, unpositioned_point_index = (
                MHubRunnerWidget._linkedMarkupPointForTableRow(findings_table, 1)
            )
            self.assertIs(unpositioned_markup, findings_markup)
            self.assertIsNone(unpositioned_point_index)
            # Verify both stable-key lookup directions, including an unpositioned finding.
            linked_table, linked_row = MHubRunnerWidget._linkedTableRowForMarkupPoint(
                findings_markup, 0
            )
            self.assertIs(linked_table, findings_table)
            self.assertEqual(linked_row, 0)
            out_of_range_table, out_of_range_row = (
                MHubRunnerWidget._linkedTableRowForMarkupPoint(findings_markup, 1)
            )
            self.assertIs(out_of_range_table, findings_table)
            self.assertIsNone(out_of_range_row)

            # Exercise Slicer's landmark-click event through to actual Qt table-row selection.
            widget = MHubRunnerWidget.__new__(MHubRunnerWidget)
            widget._linkedMarkupDisplayObservers = {}
            widget._pendingLinkedTableSelection = None
            widget._syncingResultSelection = False
            table_view = slicer.qMRMLTableView()
            table_view.setMRMLScene(slicer.mrmlScene)
            table_view.setMRMLTableNode(findings_table)
            widget._resultTableView = table_view
            widget._resultTableSelectionModel = table_view.selectionModel()
            widget._connectResultTableView = lambda: None
            widget._refreshLinkedMarkupDisplayObservers()
            display_node = findings_markup.GetDisplayNode()
            display_node.SetActiveComponent(
                slicer.vtkMRMLMarkupsDisplayNode.ComponentControlPoint, 0
            )
            display_node.InvokeEvent(
                slicer.vtkMRMLMarkupsDisplayNode.JumpToPointEvent
            )
            self.assertEqual(
                widget._pendingLinkedTableSelection,
                (findings_table.GetID(), 0),
            )
            self.assertEqual(
                slicer.app.applicationLogic().GetSelectionNode().GetActiveTableID(),
                findings_table.GetID(),
            )
            widget._applyPendingLinkedTableSelection()
            self.assertEqual(
                {index.row() for index in table_view.selectedIndexes()},
                {0},
            )
            widget._disconnectLinkedMarkupDisplayObservers()

            subject_hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(
                slicer.mrmlScene
            )
            table_parent = subject_hierarchy.GetItemParent(
                subject_hierarchy.GetItemByDataNode(findings_table)
            )
            markup_parent = subject_hierarchy.GetItemParent(
                subject_hierarchy.GetItemByDataNode(findings_markup)
            )
            self.assertEqual(table_parent, markup_parent)
            self.assertEqual(
                subject_hierarchy.GetItemAttribute(table_parent, "MHubRunner.RunId"),
                "26.08.22-12.00.00_gc_grt123_lung_cancer",
            )

            loaded_again = logic.loadStoredRun(output_directory)
            self.assertIs(loaded_again["inputNode"], volume)
            self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLTableNode")), 2)
            self.assertEqual(
                len(slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")), 1
            )

            manifest = load_run_manifest(output_directory)
            manifest["slicerSessionId"] = "different-session"
            write_run_manifest(output_directory, manifest)
            loaded_without_input = logic.loadStoredRun(output_directory)
            self.assertIsNone(loaded_without_input["inputNode"])
            self.assertIn("non-DICOM input", loaded_without_input["annotationWarning"])

        self.assertEqual(
            logic._legacyModelNameFromRunDirectory(
                "/tmp/26.08.22-12.00.00_gc_grt123_lung_cancer"
            ),
            "gc_grt123_lung_cancer",
        )
        self.assertIsNone(logic._legacyModelNameFromRunDirectory("/tmp/unrecognized"))


if __name__ == "__main__":
    unittest.main()
