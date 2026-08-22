import unittest

import slicer
import vtk

from MHubRunner import MHubRunnerLogic, Model, ModelStatus
from MHubRunnerModelHandlers import MarkupOutput, MarkupPoint


class ModelOutputRuntimeTest(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear()

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

        logic = MHubRunnerLogic()
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


if __name__ == "__main__":
    unittest.main()
