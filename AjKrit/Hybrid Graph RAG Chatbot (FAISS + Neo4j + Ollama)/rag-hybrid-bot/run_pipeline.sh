#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
.venv/bin/python build_index.py
.venv/bin/python evaluate.py
.venv/bin/python run_answers.py
.venv/bin/python export_graph.py
.venv/bin/python finalize.py
# Start app separately, then: .venv/bin/python capture.py
