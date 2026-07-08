#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

prepare_default_env() {
  export PIP_NO_BUILD_ISOLATION=1
}
