#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd "$(dirname "$0")" && pwd)"
revision_dir="$(cd "$package_dir/.." && pwd)"
python -m pytest -q "$revision_dir/tests"
python "$package_dir/run_example.py"
