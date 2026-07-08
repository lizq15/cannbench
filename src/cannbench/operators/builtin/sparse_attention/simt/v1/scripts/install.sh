#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -e "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
