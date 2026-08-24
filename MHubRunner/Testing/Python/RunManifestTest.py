import json
import os
import tempfile
import unittest

from MHubRunnerModelHandlers.run_manifest import (
    MANIFEST_FILENAME,
    RunManifestError,
    finalize_run_manifest,
    load_run_manifest,
    new_run_manifest,
    resolve_manifest_output_files,
    write_run_manifest,
)


class RunManifestTest(unittest.TestCase):
    def _manifest(self):
        return new_run_manifest(
            run_id="26.08.22-18.51.08_gc_grt123_lung_cancer",
            model_name="gc_grt123_lung_cancer",
            model_label="CT Lung cancer risk prediction",
            model_categories=["Prediction"],
            image_name="mhubai/gc_grt123_lung_cancer:latest",
            image_digest="sha256:" + "a" * 64,
            slicer_session_id="test-session",
            input_data={
                "nodeId": "vtkMRMLScalarVolumeNode1",
                "wasDicom": True,
                "dicomSeriesInstanceUID": "1.2.3",
                "dicomInstanceUIDHash": "abc123",
                "geometry": {
                    "dimensions": [512, 512, 133],
                    "ijkToRAS": [
                        [-0.703125, 0.0, 0.0, 166.0],
                        [0.0, -0.703125, 0.0, 171.699997],
                        [0.0, 0.0, 2.5, -340.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            },
            created_at="2026-08-22T18:51:08+02:00",
        )

    def test_round_trip_and_finalize(self):
        with tempfile.TemporaryDirectory() as output_directory:
            write_run_manifest(output_directory, self._manifest())
            loaded = load_run_manifest(output_directory)
            self.assertEqual(loaded["status"]["state"], "running")
            self.assertEqual(loaded["modelOutputDirectory"], "outputs")
            self.assertEqual(loaded["model"]["imageDigest"], "sha256:" + "a" * 64)
            self.assertNotIn("nodeName", loaded["input"])

            finalized = finalize_run_manifest(
                output_directory,
                return_code=0,
                timed_out=False,
                killed=False,
                output_paths=["outputs/series/findings.json"],
            )
            self.assertEqual(finalized["status"]["state"], "succeeded")
            self.assertEqual(finalized["outputs"], ["outputs/series/findings.json"])
            self.assertTrue(os.path.exists(os.path.join(output_directory, MANIFEST_FILENAME)))

    def test_invalid_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_directory:
            path = os.path.join(output_directory, MANIFEST_FILENAME)
            manifest = self._manifest()
            manifest["schemaVersion"] = 999
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaises(RunManifestError):
                load_run_manifest(output_directory)

    def test_parent_output_path_is_rejected(self):
        unsafe_paths = [
            "../outside.json",
            "outputs/../../outside.json",
            r"outputs\..\..\outside.json",
            "/outside.json",
            r"C:\outside.json",
            "other/output.json",
        ]

        # Reject traversal, absolute paths, and outputs outside the declared output directory.
        for unsafe_path in unsafe_paths:
            with self.subTest(path=unsafe_path):
                manifest = self._manifest()
                manifest["outputs"] = [unsafe_path]
                with tempfile.TemporaryDirectory() as output_directory:
                    with self.assertRaises(RunManifestError):
                        write_run_manifest(output_directory, manifest)

    def test_nested_parent_model_output_directory_is_rejected(self):
        manifest = self._manifest()
        manifest["modelOutputDirectory"] = "outputs/../../outside"
        with tempfile.TemporaryDirectory() as output_directory:
            with self.assertRaises(RunManifestError):
                write_run_manifest(output_directory, manifest)

    def test_manifest_outputs_resolve_only_declared_regular_files(self):
        with tempfile.TemporaryDirectory() as output_directory:
            outputs_directory = os.path.join(output_directory, "outputs")
            os.makedirs(outputs_directory)
            declared_path = os.path.join(outputs_directory, "declared.json")
            undeclared_path = os.path.join(outputs_directory, "undeclared.json")
            for path in (declared_path, undeclared_path):
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write("{}")

            # Return only the file explicitly named by the validated manifest.
            manifest = self._manifest()
            manifest["outputs"] = ["outputs/declared.json"]
            self.assertEqual(
                resolve_manifest_output_files(output_directory, manifest),
                [os.path.realpath(declared_path)],
            )

    def test_manifest_output_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as output_directory, tempfile.TemporaryDirectory() as outside:
            outputs_directory = os.path.join(output_directory, "outputs")
            os.makedirs(outputs_directory)
            outside_path = os.path.join(outside, "outside.json")
            with open(outside_path, "w", encoding="utf-8") as stream:
                stream.write("{}")
            linked_path = os.path.join(outputs_directory, "linked.json")
            try:
                os.symlink(outside_path, linked_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"Symbolic links are unavailable: {exc}")

            # Reject a declared output whose resolved target leaves modelOutputDirectory.
            manifest = self._manifest()
            manifest["outputs"] = ["outputs/linked.json"]
            with self.assertRaises(RunManifestError):
                resolve_manifest_output_files(output_directory, manifest)

    def test_invalid_image_digest_is_rejected(self):
        manifest = self._manifest()
        manifest["model"]["imageDigest"] = "latest"
        with tempfile.TemporaryDirectory() as output_directory:
            with self.assertRaises(RunManifestError):
                write_run_manifest(output_directory, manifest)


if __name__ == "__main__":
    unittest.main()
