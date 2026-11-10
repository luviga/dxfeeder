"""Package checks.

This module runs the test suite, validates the settings file, checks that every public function of the
package has a docstring, and scans the tracked files for drive letters and machine paths.
It prints one line per finding and returns exit code 1 if any check fails, else 0.
Run it as python -m dxfeeder.check or as the console command dxfeeder check.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
LABELS = {"SOURCE", "ASSUMPTION", "ESTIMATION"}
SKIPPED_FOLDERS = {".git", "data", "__pycache__", ".pytest_cache"}
# A drive letter path: one letter, a colon and a slash, not preceded by another letter or digit.
DRIVE_LETTER = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
# A user home folder or a per-user application data folder.
MACHINE_PATH = re.compile(r"[\\/](?:Users|home)[\\/][^\\/\s\"']+|App[D]ata", re.IGNORECASE)


def run_tests(root=ROOT):
    """Run the test suite in a separate process and return a list of findings, empty when every test passes."""
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=root)
    return [] if done.returncode == 0 else [f"tests: pytest exited with code {done.returncode}"]


def setting_findings(table, prefix=""):
    """Return the findings of one settings table, where a table with a value is a setting and any other table groups settings."""
    findings = []
    for name, entry in table.items():
        where = f"{prefix}{name}"
        if not isinstance(entry, dict):
            findings.append(f"settings.toml: {where} is a bare value, not a table with value, label and reason")
        elif "value" in entry:
            if entry.get("label") not in LABELS:
                findings.append(f"settings.toml: {where} has no label from {sorted(LABELS)}")
            if not str(entry.get("reason", "")).strip():
                findings.append(f"settings.toml: {where} has no reason")
            extra = set(entry) - {"value", "label", "reason"}
            if extra:
                findings.append(f"settings.toml: {where} has unknown keys {sorted(extra)}")
        else:
            findings.extend(setting_findings(entry, where + "."))
    return findings


def check_settings(root=ROOT):
    """Return the findings of the settings file: every value must carry a label and a reason."""
    path = Path(root) / "settings.toml"
    if not path.exists():
        return ["settings.toml: the file is missing"]
    try:
        with path.open("rb") as handle:
            return setting_findings(tomllib.load(handle))
    except tomllib.TOMLDecodeError as error:
        return [f"settings.toml: the file does not parse: {error}"]


def public_functions(tree):
    """Yield each public function of a module tree, with the methods of each public class, as name and node."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            yield node.name, node
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not item.name.startswith("_") or item.name == "__init__"):
                    yield f"{node.name}.{item.name}", item


def check_docstrings(package=PACKAGE):
    """Return one finding per public function of the package without a docstring."""
    findings = []
    for path in sorted(Path(package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, node in public_functions(tree):
            if not ast.get_docstring(node):
                findings.append(f"docstring: {path.relative_to(Path(package).parent).as_posix()}:{node.lineno} {name} has no docstring")
    return findings


def tracked_files(root=ROOT):
    """Return the tracked files of the repository, or every file outside the skipped folders when git is not available."""
    try:
        done = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
        return [Path(root) / name for name in done.stdout.decode("utf-8").split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        return [path for path in Path(root).rglob("*")
                if path.is_file() and not SKIPPED_FOLDERS & set(path.relative_to(root).parts)
                and not path.relative_to(root).parts[0].startswith("env")]


def check_paths(root=ROOT):
    """Return one finding per drive letter or machine path in the text of a tracked file. Binary files are skipped."""
    findings = []
    for path in tracked_files(root):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
            for pattern in (DRIVE_LETTER, MACHINE_PATH):
                for match in pattern.finditer(line):
                    findings.append(f"path: {path.relative_to(root).as_posix()}:{number} holds {match.group(0)!r}")
    return findings


def main():
    """Run every check, print each finding and a summary, and return the exit code."""
    findings = []
    for name, check in (("tests", run_tests), ("settings", check_settings), ("docstrings", check_docstrings),
                        ("paths", check_paths)):
        found = check()
        print(f"{name}: {'ok' if not found else f'{len(found)} finding(s)'}", flush=True)
        findings.extend(found)
    for finding in findings:
        print(finding)
    print("check passed" if not findings else f"check failed with {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
