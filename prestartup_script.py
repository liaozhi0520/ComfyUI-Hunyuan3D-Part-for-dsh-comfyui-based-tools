"""Pre-startup script for Hunyuan3D-Part."""

from pathlib import Path
from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy assets to input/3d/
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d")

# Copy example workflows to user directory
copy_files(SCRIPT_DIR / "workflows", COMFYUI_DIR / "user" / "default" / "workflows", "*.json")

# ---------------------------------------------------------------------------
# pixi self-heal (dsh fork addition)
#
# Some GPU cloud platforms sweep the persistent volume on boot, replacing
# files whose content hits their read-only object store with 644 symlinks on
# read-only zfs (chmod denied) - executables lose their exec bit. comfy-env
# 0.3.x hardcodes exec ~/.pixi/bin/pixi (registration metadata scan + isolated
# worker spawn), so a swept pixi means either "Registered 0 total nodes"
# (nodes all red) or PermissionError: [Errno 13] at runtime.
#
# Layout (installed by the dsh provision script): the real binary is kept as a
# gz data copy (data needs no exec bit, stays readable even when swept), and
# ~/.pixi/bin/pixi is a small unique shim that re-materializes the binary on
# every call. This prestartup hook rebuilds the shim itself if it was swept,
# warns loudly when the data copy is missing, and pre-warms materialization.
# Every failure is logged to ~/fix-pixi.log and never breaks ComfyUI startup.
# ---------------------------------------------------------------------------
def _heal_pixi():
    import os
    import shutil
    import stat
    import subprocess
    import sys
    from datetime import datetime

    home = os.path.expanduser("~")
    real = os.path.join(home, ".pixi-real")
    pixi = os.path.join(home, ".pixi", "bin", "pixi")
    shim_src = os.path.join(real, "shim.sh")
    gz = os.path.join(real, "pixi.gz")
    bin_ = os.path.join(real, "pixi.bin")
    log = os.path.join(home, "fix-pixi.log")

    def _log(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [pixi-guard] {msg}"
        print(line, file=sys.stderr, flush=True)
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _ok_exec(path):
        try:
            st = os.lstat(path)
        except OSError:
            return False
        if stat.S_ISLNK(st.st_mode):
            return False
        return bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

    if sys.platform == "win32":
        return
    if not os.path.isdir(os.path.dirname(pixi)):
        return  # pixi never installed for this user: nothing to guard
    if not os.path.isfile(gz):
        _log(f"BROKEN: missing data copy {gz} - re-run the dsh provision script to rebuild")
        return
    if not _ok_exec(pixi):
        if not os.path.isfile(shim_src):
            _log(f"HEAL FAILED: missing shim source {shim_src} - re-run the dsh provision script")
            return
        try:
            os.makedirs(os.path.dirname(pixi), exist_ok=True)
            if os.path.lexists(pixi):
                os.remove(pixi)
            shutil.copy2(shim_src, pixi)
            os.chmod(pixi, 0o755)
            _log(f"shim rebuilt ({pixi} was missing or swept)")
        except OSError as e:
            _log(f"HEAL FAILED: shim rebuild {type(e).__name__}: {e}")
            return
    if not _ok_exec(bin_):
        try:
            r = subprocess.run([pixi, "--version"], capture_output=True, timeout=180)
            if r.returncode == 0:
                _log("pixi materialized and ready (prestartup warm-up)")
            else:
                tail = r.stderr.decode("utf-8", "replace").strip()[:200]
                _log(f"HEAL FAILED: {pixi} --version rc={r.returncode} {tail}")
        except Exception as e:
            _log(f"HEAL FAILED: warm-up {type(e).__name__}: {e}")


try:
    _heal_pixi()
except Exception as e:  # never break ComfyUI startup
    try:
        import sys
        print(f"[pixi-guard] unexpected error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    except Exception:
        pass
