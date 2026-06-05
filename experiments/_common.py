"""Shared utilities for experiment scripts.

Provides import shims so that scripts in experiments/ can `import` the
solvers under solver/ without an editable install, and standardized
output paths.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLVER_DIR = REPO_ROOT / "solver"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(SOLVER_DIR))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def out_path(name: str) -> Path:
    return RESULTS_DIR / name


def _cbc_version() -> str:
    """Query CBC's reported version. Returns 'unknown' on any failure."""
    try:
        import pulp
        solver = pulp.PULP_CBC_CMD(msg=False)
        cbc_path = solver.path
        if cbc_path is None:
            return "unknown (no CBC path)"
        result = subprocess.run(
            [cbc_path, "-version"], capture_output=True, text=True, timeout=5
        )
        first_line = (result.stdout or result.stderr).splitlines()[0].strip()
        return first_line or "unknown"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def _highs_version() -> str:
    """Query HiGHS's reported version via highspy. Returns 'unknown' on failure."""
    try:
        import highspy
        return f"HiGHS {highspy.__version__}" if hasattr(highspy, "__version__") \
            else "HiGHS (version not exposed)"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def _gurobi_version() -> str:
    """Query Gurobi's reported version via gurobipy. Returns 'unknown' on failure.

    Reports both the gurobipy wheel version and the linked Gurobi engine
    version; on a WLS license the engine version is what reviewers will
    cite (e.g. 'Gurobi 13.0.2').
    """
    try:
        import gurobipy as gp
        engine = ".".join(str(v) for v in gp.gurobi.version())
        return f"Gurobi {engine} (gurobipy {gp.__version__})"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(REPO_ROOT),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _ram_gb() -> float:
    """Best-effort total RAM in GB. Returns 0.0 if undetectable."""
    try:
        if sys.platform == "win32":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 1)
        # POSIX fallback via sysconf
        pages = __import__("os").sysconf("SC_PHYS_PAGES")
        page_size = __import__("os").sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3), 1)
    except Exception:
        return 0.0


def write_machine_json(script_name: str) -> Path:
    """Record reproducibility metadata for the current script run.

    Writes two files in results/:
      - machine.json: latest run wins (quick reference)
      - machine_<script_stem>.json: per-script archive (audit trail)

    Captures: Python, PuLP, CBC, OS, CPU, RAM, git commit, timestamp.
    If git_commit is 'unknown' or the working tree is dirty, the data
    cannot be reliably tied back to a specific code state — note this in
    the audit when interpreting results.
    """
    try:
        import pulp
        pulp_version = pulp.__version__
    except Exception:
        pulp_version = "unknown"

    meta = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "pulp_version": pulp_version,
        "highs_version": _highs_version(),
        "cbc_version": _cbc_version(),
        "gurobi_version": _gurobi_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": _ram_gb(),
    }
    latest = out_path("machine.json")
    with open(latest, "w") as f:
        json.dump(meta, f, indent=2)
    stem = Path(script_name).stem
    archive = out_path(f"machine_{stem}.json")
    with open(archive, "w") as f:
        json.dump(meta, f, indent=2)
    return latest
