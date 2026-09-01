import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from MHubRunner import MHubRunnerLogic, ProgressObserver


class _FakeProgressObserver:
    latest = None

    def __init__(self, cmd, frequency, timeout, data, env):
        self.cmd = cmd
        self.frequency = frequency
        self.timeout = timeout
        self.data = data
        self.env = env
        self.stop_callback = None
        self.progress_callback = None
        self.terminate_callback = None
        _FakeProgressObserver.latest = self

    def onStop(self, callback):
        self.stop_callback = callback

    def onProgress(self, callback):
        self.progress_callback = callback

    def onTerminate(self, callback):
        self.terminate_callback = callback


class ContainerLifecycleTest(unittest.TestCase):
    def test_model_run_is_named_and_has_no_default_timeout(self):
        # Capture the Docker command without starting a real container.
        logic = MHubRunnerLogic.__new__(MHubRunnerLogic)
        logic.getDockerExecutable = lambda: "/mock/docker"
        logic._build_subprocess_env = lambda executable: {"PATH": "/mock"}
        model = SimpleNamespace(name="example_model")

        with patch("MHubRunner.ProgressObserver", _FakeProgressObserver):
            observer = logic._run_mhub_docker(
                model=model,
                gpus=None,
                input_dir="/input",
                output_dir="/output",
                onProgress=lambda progress, output: None,
                onStop=lambda returncode, output, timedout, killed: None,
                run_id="26.08.31-12.00.00_example_model",
            )

        self.assertEqual(observer.timeout, 0)
        self.assertIn("--name", observer.cmd)
        self.assertIn(
            "mhubrunner-26.08.31-12.00.00_example_model",
            observer.cmd,
        )
        self.assertIn("org.mhubai.slicer-mhub-runner=true", observer.cmd)
        self.assertEqual(
            observer.data["container_name"],
            "mhubrunner-26.08.31-12.00.00_example_model",
        )

    def test_cancel_stops_the_named_container(self):
        # Exercise the resource callback registered by the Docker run workflow.
        logic = MHubRunnerLogic.__new__(MHubRunnerLogic)
        logic.getDockerExecutable = lambda: "/mock/docker"
        logic._build_subprocess_env = lambda executable: {"PATH": "/mock"}
        model = SimpleNamespace(name="example_model")

        with patch("MHubRunner.ProgressObserver", _FakeProgressObserver):
            observer = logic._run_mhub_docker(
                model=model,
                gpus=None,
                input_dir="/input",
                output_dir="/output",
                onProgress=lambda progress, output: None,
                onStop=lambda returncode, output, timedout, killed: None,
                run_id="test-run",
            )

        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
            observer.terminate_callback(False, True)

        self.assertEqual(
            run.call_args.args[0],
            ["/mock/docker", "stop", "--timeout", "10", "mhubrunner-test-run"],
        )

    def test_progress_observer_reports_completion_after_termination(self):
        # Build a controlled observer to verify container cleanup precedes final callbacks.
        events = []

        class FakeTimer:
            def stop(self):
                events.append("timer-stopped")

        class FakeProcess:
            def poll(self):
                return None

            def kill(self):
                events.append("client-killed")

            def wait(self, timeout):
                events.append("client-reaped")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as stream:
            stream.write("model output")
            output_path = stream.name

        observer = ProgressObserver.__new__(ProgressObserver)
        observer.cmd = ["docker", "run"]
        observer._disabled = False
        observer._finished = False
        observer._timer = FakeTimer()
        observer._proc = FakeProcess()
        observer._stdout_file_name = output_path
        observer._onTerminate = lambda timedout, killed: events.append("container-stopped")
        observer._onStop = lambda returncode, output, timedout, killed: events.append(
            "completion-reported"
        )
        ProgressObserver._tasks.append(observer)

        observer.kill()

        self.assertEqual(
            events,
            [
                "timer-stopped",
                "container-stopped",
                "client-killed",
                "client-reaped",
                "completion-reported",
            ],
        )
        self.assertNotIn(observer, ProgressObserver._tasks)
        self.assertFalse(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
