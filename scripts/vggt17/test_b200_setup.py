import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

FAKE_PYTHON = r"""
import json
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as handle:
    handle.write(json.dumps({"args": args, "ds_build_ops": os.environ.get("DS_BUILD_OPS")}) + "\n")
ready = Path(os.environ["UV_READY"])
if args == ["-m", "uv", "--version"]:
    sys.exit(0 if ready.exists() else 1)
if args[:3] == ["-m", "pip", "install"]:
    ready.touch()
elif args[:3] == ["-m", "uv", "venv"]:
    target = Path(args[-1]) / "bin/python"
    target.parent.mkdir(parents=True)
    shutil.copyfile(__file__, target)
    target.chmod(0o755)
elif args == ["-"]:
    print("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl")
"""


class B200SetupTests(unittest.TestCase):
    def run_setup(self, root, existing=False):
        executable = root / "bootstrap"
        executable.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
        executable.chmod(0o755)
        prefix = root / "env"
        ready = root / "uv-ready"
        if existing:
            ready.touch()
            python = prefix / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(executable.read_text())
            python.chmod(0o755)
        logfile = root / "commands.jsonl"
        env = dict(os.environ, BOOTSTRAP_PYTHON=str(executable), COMMAND_LOG=str(logfile), UV_READY=str(ready))
        script = Path(__file__).with_name("setup_b200.sh")
        subprocess.run(["bash", str(script), str(prefix)], env=env, check=True, capture_output=True, text=True)
        return [json.loads(line) for line in logfile.read_text().splitlines()], prefix, executable

    def test_pip_bootstraps_uv_and_uv_targets_the_b200_environment(self):
        with tempfile.TemporaryDirectory() as name:
            commands, prefix, executable = self.run_setup(Path(name))
            args = [command["args"] for command in commands]
            self.assertIn(["-m", "pip", "install", "--user", "--upgrade", "uv"], args)
            self.assertIn(["-m", "uv", "venv", "--python", str(executable), "--seed", str(prefix)], args)
            installs = [command for command in commands if command["args"][:4] == ["-m", "uv", "pip", "install"]]
            self.assertEqual(len(installs), 5)
            for command in installs:
                options = command["args"]
                self.assertEqual(options[options.index("--python") + 1], str(prefix / "bin/python"))
            cuda_installs = [command["args"] for command in installs if "--torch-backend" in command["args"]]
            self.assertEqual(len(cuda_installs), 2)
            for options in cuda_installs:
                self.assertEqual(options[options.index("--torch-backend") + 1], "cu128")
            requirements = next(command for command in installs if "-r" in command["args"])
            self.assertEqual(requirements["ds_build_ops"], "0")
            self.assertIn("--no-build-isolation", requirements["args"])

    def test_existing_uv_and_environment_are_reused(self):
        with tempfile.TemporaryDirectory() as name:
            commands, _, _ = self.run_setup(Path(name), existing=True)
            self.assertFalse(any(command["args"][:3] == ["-m", "pip", "install"] for command in commands))
            self.assertFalse(any(command["args"][:3] == ["-m", "uv", "venv"] for command in commands))


if __name__ == "__main__":
    unittest.main()
