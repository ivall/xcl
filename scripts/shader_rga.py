"""Opt-in RGA DX12 offline compute diagnostics, separate from XCL validation."""
import csv
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def execute(command, directory, label, timeout):
    stdout, stderr = directory / f"{label}.stdout.txt", directory / f"{label}.stderr.txt"
    result = dict(command=command, exit_code=None, timed_out=False,
                  stdout=str(stdout), stderr=str(stderr))
    start = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        with subprocess.Popen(command, cwd=directory, stdout=out, stderr=err,
                              start_new_session=os.name != "nt") as process:
            try:
                result["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                if os.name == "nt":
                    subprocess.run([str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"),
                                    "/PID", str(process.pid), "/T", "/F"],
                                   stdout=err, stderr=err, timeout=15, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                if process.poll() is None:
                    process.kill()
                result["exit_code"] = process.wait()
    result["wall_seconds"] = time.monotonic() - start
    return result


def analyze(run, report, tool, target, timeout=120):
    if report["category"] != "analyzer_not_requested":
        return
    report["analyzer"] = dict(name="Radeon GPU Analyzer", mode="dx12-offline", target=target,
                              host_adapter=None, version=None, invocations=[], statistics=[])
    info = report["analyzer"]
    if report["manifest"].get("stage") not in ("CS", "kernel"):
        report["category"] = "unsupported_stage"
        return
    if not target or not re.fullmatch(r"gfx[0-9a-f]+", target):
        report.update(status="ERROR", category="invalid_target")
        return
    executable = shutil.which(tool) if tool else None
    if not executable:
        report["category"] = "tool_missing"
        return
    executable = str(Path(executable).resolve())
    # Batch files invoke a shell on Windows; only a native executable is supported.
    if os.name == "nt" and Path(executable).suffix.lower() != ".exe":
        report.update(status="ERROR", category="invalid_tool")
        return
    directory = run / "rga"
    directory.mkdir()
    try:
        for label, arguments in (("version", ["--version"]), ("help", ["-s", "dx12", "-h"])):
            invocation = execute([executable, *arguments], directory, label, timeout)
            info["invocations"].append(invocation)
            if invocation["timed_out"] or invocation["exit_code"] != 0:
                report.update(status="ERROR", category="tool_probe_failed")
                return
        info["version"] = (directory / "version.stdout.txt").read_text(encoding="utf-8", errors="replace").strip()
        help_text = (directory / "help.stdout.txt").read_text(encoding="utf-8", errors="replace")
        if not all(flag in help_text for flag in ("--offline", "--cs-blob")):
            report["category"] = "unsupported_tool_mode"
            return
        dxil = next(v["path"] for k, v in report["artifacts"].items() if k.endswith(".direct.dxil"))
        command = [executable, "-s", "dx12", "--offline", "-c", target, "--cs-blob", dxil,
                   "--isa", str(directory / "shader.isa"), "-a", str(directory / "stats.csv")]
        invocation = execute(command, directory, "analysis", timeout)
        info["invocations"].append(invocation)
        outputs = [p for p in directory.glob("*.isa") if p.stat().st_size]
        info["isa_files"] = [str(p) for p in outputs]
        for path in sorted(directory.glob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="", errors="replace") as stream:
                info["statistics"].append(dict(path=str(path), rows=list(csv.DictReader(stream))))
        if invocation["timed_out"]:
            report.update(status="ERROR", category="analyzer_timeout")
        elif invocation["exit_code"] != 0 or not outputs:
            # RGA versions can return zero even when pipeline creation fails.
            report.update(status="ERROR", category="analyzer_failed")
        else:
            report.update(status="OK", category="analysis_complete")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report.update(status="ERROR", category="analyzer_failed", detail=str(exc))
