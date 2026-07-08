#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

prepare_default_env

cd "${PROJECT_ROOT}"
"${PYTHON:-python3}" -m pip install -e . --no-build-isolation --no-deps "$@"
