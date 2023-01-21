import sys

from easyprocess import EasyProcess
from pyvirtualdisplay import Display

with Display(visible=False, size=(100, 60), use_xauth=True) as display:
    with EasyProcess(
        [
            "python",
            "-m",
            "pytest",
            "--durations=5",
            "--durations-min=0.04",
            "--cov-branch",
            "--cov=hwinarion/",
            "--cov-report",
            "term-missing:skip-covered",
        ]
    ) as proc:
        result = proc.wait()
        print(result.stdout)
        sys.exit(result.return_code)
