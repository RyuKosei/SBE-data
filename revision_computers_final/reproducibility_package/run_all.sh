#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd "$(dirname "$0")" && pwd)"
revision_dir="$(cd "$package_dir/.." && pwd)"
repo_root="$(cd "$revision_dir/.." && pwd)"
cd "$repo_root"
if command -v python3 >/dev/null 2>&1; then
  python_bin=python3
elif command -v python >/dev/null 2>&1; then
  python_bin=python
else
  echo "Python 3.10 or later is required." >&2
  exit 1
fi
"$python_bin" -m pytest -q "$revision_dir/tests"
"$python_bin" "$package_dir/run_example.py"
