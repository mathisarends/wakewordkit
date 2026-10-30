"""Build and test the wheel in an isolated, non-editable environment."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wakewordkit-wheel-") as directory:
        work = Path(directory)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(work / "dist")],
            cwd=ROOT,
            check=True,
        )
        environment = work / "env"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)], check=True
        )
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        wheel = next((work / "dist").glob("*.whl"))
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)], check=True
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                """
import importlib.metadata as metadata
from pathlib import Path
import numpy as np
import wakewordkit
from wakewordkit import WakeWord, WakeWordDetector
assert Path(wakewordkit.__file__).is_relative_to(Path(__import__('sys').prefix))
assert metadata.version('openwakeword') == '0.4.0'
assert metadata.version('wakewordkit') == wakewordkit.__version__
assert Path(wakewordkit.__file__).with_name('py.typed').is_file()
assert all(word.path.is_file() for word in WakeWord)
detector = WakeWordDetector()
assert len(detector.names) == 4
import asyncio
asyncio.run(detector.process(np.zeros(320, dtype=np.int16)))
asyncio.run(detector.finish())
print('Installed wheel: model assets and real inference OK')
""",
            ],
            cwd=work,
            check=True,
        )


if __name__ == "__main__":
    main()
