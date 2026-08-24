import json
import os
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
    def _logic_without_dependency_setup():
        return MHubRunnerLogic.__new__(MHubRunnerLogic)

    def test_extension_build_footer(self):
        label = qt.QLabel()
        widget = SimpleNamespace(ui=SimpleNamespace(lblExtensionBuildInfo=label))

        MHubRunnerWidget._updateExtensionBuildInfo(widget)

        self.assertTrue(label.text.startswith("MHubRunner 2.4.0-dev \u00b7 build "))
        self.assertIn("Extension version: 2.4.0-dev", label.toolTip)
        self.assertIn("Git revision:", label.toolTip)

    def test_docker_image_digest_is_read_from_local_image(self):
        logic = self._logic_without_dependency_setup()
        logic._executables = {"docker": "/mock/docker"}
        expected_digest = "sha256:" + "a" * 64

        with patch("subprocess.run") as run:
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

        logic = self._logic_without_dependency_setup()
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

        logic = self._logic_without_dependency_setup()
        with tempfile.TemporaryDirectory() as output_directory:
            result_path = os.path.join(
                output_directory, "gc_grt123_lung_cancer_findings.json"
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
