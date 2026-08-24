import json
import os
import tempfile
import unittest

from MHubRunnerModelHandlers import (
    InputHandlerContext,
    ModelHandlerRegistry,
    OutputHandlerContext,
)


class ModelOutputHandlersTest(unittest.TestCase):
    def test_grt123_output_plan(self):
        payload = {
            "lungcad": {
                "revision": "9a4ca0415c7fc1d3023a16650bf1cdce86f8bb59",
                "name": "grt123",
                "datetimeofexecution": "08/21/2026 18:51:08",
                "coordinatesystem": "World",
                "computationtimeinseconds": 32.580826,
            },
            "imageinfo": {
                "dimensions": [512, 512, 133],
                "voxelsize": [0.703125, 0.703125, 2.5],
                "origin": [-166.0, -171.699997, -340.0],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "seriesuid": "dicom",
            },
            "findings": [
                {
                    "id": 0,
                    "x": 56.0,
                    "y": 86.300003,
                    "z": -112.0,
                    "probability": 0.9999949932098389,
                    "cancerprobability": 0.8792480826377869,
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
                "casecancerprobability": 0.8809547424316406,
                "referencenoduleids": [0],
            },
        }

        with tempfile.TemporaryDirectory() as output_directory:
            output_file = os.path.join(
                output_directory, "gc_grt123_lung_cancer_findings (2).json"
            )
            with open(output_file, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)

            context = OutputHandlerContext(
                model_name="gc_grt123_lung_cancer",
                model_label="CT Lung cancer risk prediction",
                model_categories=["Prediction"],
                output_directory=output_directory,
            )
            handler = ModelHandlerRegistry().handler_for(context.model_name)
            plan = handler.build_output_plan(context)

        self.assertEqual(type(handler).__name__, "Grt123LungCancerHandler")
        self.assertEqual(len(plan.tables), 2)
        self.assertEqual(plan.tables[0].rows[0], ["Case cancer probability", 0.8809547424316406])
        self.assertEqual(plan.tables[1].rows[0][0], 0)
        self.assertEqual(plan.tables[1].identity, "findings")
        self.assertEqual(plan.tables[1].link_group, "findings")
        self.assertEqual(plan.tables[1].row_keys, ["finding:0", "finding:1"])
        self.assertEqual(len(plan.markups), 1)
        self.assertEqual(plan.markups[0].identity, "findings")
        self.assertEqual(plan.markups[0].link_group, "findings")
        self.assertEqual(plan.markups[0].points[0].key, "finding:0")
        self.assertEqual(len(plan.markups[0].points), 1)
        self.assertEqual(plan.markups[0].points[0].position_lps, (56.0, 86.300003, -112.0))

    def test_unknown_model_uses_generic_handler(self):
        handler = ModelHandlerRegistry().handler_for("unregistered_exact_model")
        self.assertEqual(type(handler).__name__, "GenericModelHandler")

    def test_generic_input_plan_preserves_selected_node_and_modality(self):
        selected_node = object()
        handler = ModelHandlerRegistry().handler_for("unregistered_exact_model")
        plan = handler.resolve_inputs(
            InputHandlerContext(
                model_name="unregistered_exact_model",
                selected_nodes=[selected_node],
                selected_modality="CT",
            )
        )
        self.assertEqual(len(plan.items), 1)
        self.assertIs(plan.items[0].node, selected_node)
        self.assertEqual(plan.items[0].modality, "CT")


if __name__ == "__main__":
    unittest.main()
