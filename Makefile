# SPDX-License-Identifier: Apache-2.0
#
# On this machine `python3` is 3.9; this project needs 3.12.  Every target therefore uses
# the virtual environment's interpreter directly, and the bootstrap uses python3.12
# explicitly.  tests/test_repo_hygiene.py::test_no_bare_python3_invocation enforces it.

PY      := .venv/bin/python
RUFF    := .venv/bin/ruff
BOOT    := /usr/bin/python3.12

.PHONY: help venv smoke test fast lint data baseline conformal quantum mps explain measure \
        figures freeze claims tex pdf pdf-draft submission check reproduce clean

help:
	@echo "venv       create .venv with python3.12 and install pinned dependencies"
	@echo "smoke      S0-S8 environment assertions; any failure stops the build"
	@echo "test       full pytest suite"
	@echo "fast       pytest without the slow cases"
	@echo "lint       ruff"
	@echo "data       stage IEEE-CIS from the local zip; fetch ULB and Sparkov"
	@echo "baseline   E1 splits and integrity, E2 label-censoring audit, E3 classical baselines"
	@echo "conformal  E4 PSD2 envelope, E5 two-sided risk control, E6 coverage, E7 exchangeability"
	@echo "quantum    E8 a-priori screens, E9 kernel arm and controls, E11 simulator parity"
	@echo "mps        E10 tensor-network arm"
	@echo "explain    E12 SHAP and circuit sensitivity"
	@echo "measure    E13 latency, E14 issuer economics, E15 ULB stress case"
	@echo "figures    submission figures, generated from the tables"
	@echo "freeze     write the SHA-256 manifest"
	@echo "claims     recompute every number quoted in prose from its source table"
	@echo "tex        regenerate the LaTeX macros from docs/claims.yaml"
	@echo "pdf        build the submission PDFs; format constraints are build assertions"
	@echo "pdf-draft  same build with the assertions demoted to a report (fitting loop)"
	@echo "check      build the PDFs, then the claim, citation and manifest gates"
	@echo "submission stage the five files the portal accepts"
	@echo "reproduce  run every measurement target in order, then freeze"
	@echo "clean      remove build caches and LaTeX artefacts"

# torch is installed first from the cu130 index because the PyPI default build does not
# carry sm_120.  The GPU cross-check extra is deliberately NOT installed here: it pulls
# cuquantum-cu11, whose cuStateVec binary is NVIDIA proprietary.  Install it explicitly
# with `make venv-gpu` when running E11, and update NOTICE when you do.
venv:
	$(BOOT) -m venv .venv
	$(PY) -m pip install --upgrade pip -q
	$(PY) -m pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
	$(PY) -m pip install -e ".[dev]" -q
	$(PY) -c "import torch, xgboost; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| arch', torch.cuda.get_arch_list()[-3:] if torch.cuda.is_available() else 'n/a'); print('xgboost', xgboost.__version__)"

# qiskit-aer and qiskit-aer-gpu-cu11 are separate distributions that both install the same
# `qiskit_aer` package, so whichever pip unpacks last wins.  Resolved together, the CPU
# build can land second and Aer then reports devices ('CPU',) with no error anywhere -- the
# GPU cross-check would silently become a second CPU run.  The GPU wheel is therefore
# reinstalled last, and smoke S1 asserts the device is real by reading nvidia-smi.
venv-gpu:
	$(PY) -m pip install -e ".[dev,gpu-crosscheck]" -q
	$(PY) -m pip install --force-reinstall --no-deps -q qiskit-aer-gpu-cu11==0.17.2
	$(PY) -c "from qiskit_aer import AerSimulator; d = AerSimulator().available_devices(); print('Aer devices:', d); assert 'GPU' in d, 'GPU wheel did not win the install order'"
	@echo "Aer GPU installed.  cuquantum-cu11 came with it under the NVIDIA SLA; see NOTICE section 4."

smoke:
	$(PY) scripts/smoke.py

test:
	$(PY) -m pytest tests/ -q

fast:
	$(PY) -m pytest tests/ -q -m "not slow"

lint:
	$(RUFF) check src tests scripts

data:
	$(PY) scripts/fetch_data.py

baseline:
	$(PY) scripts/make_splits.py
	$(PY) scripts/audit_labels.py
	$(PY) scripts/run_baselines.py

conformal:
	$(PY) scripts/run_envelope.py
	$(PY) scripts/run_conformal.py
	$(PY) scripts/measure_shift.py

quantum:
	$(PY) scripts/screen_kernels.py
	$(PY) scripts/run_quantum.py
	$(PY) scripts/check_parity.py

mps:
	$(PY) scripts/run_mps.py

explain:
	$(PY) scripts/run_explain.py

measure:
	$(PY) scripts/measure_latency.py
	$(PY) scripts/measure_economics.py
	$(PY) scripts/run_ulb.py

figures:
	$(PY) scripts/make_figures.py

freeze:
	$(PY) scripts/freeze.py

claims:
	$(PY) scripts/check_claims.py

# `check` depends on `pdf` because it gates *against* the built PDFs: claims.yaml lists them
# as documents and freeze.py hashes them in the scientific class.  Without the dependency,
# `make reproduce && make check` validates documents built before the tables moved.
check: pdf claims
	$(PY) scripts/freeze.py --check

reproduce: baseline conformal quantum mps explain measure figures freeze
	@echo
	@echo "Reproduction complete.  Verify a later run against this one with: make check"

# Deterministic output so the built PDFs can enter the manifest.
TEXFLAGS       := -pdf -interaction=nonstopmode -halt-on-error -file-line-error
DETERMINISTIC  := SOURCE_DATE_EPOCH=1757894400 FORCE_SOURCE_DATE=1
BODY_PAGES     := 6   # the guidelines allow six; five left a page unused
APPENDIX_PAGES := 3
TEX_SOURCES    := $(wildcard submission/*.tex submission/content/*.tex)
PDF_GUARD      := $(addprefix --source ,$(TEX_SOURCES))

tex:
	$(PY) scripts/make_tex.py --out submission/generated

submission/proposal.pdf: $(TEX_SOURCES) submission/generated/claims.tex
	cd submission && $(DETERMINISTIC) latexmk $(TEXFLAGS) proposal.tex

submission/appendix.pdf: $(TEX_SOURCES) submission/generated/claims.tex
	cd submission && $(DETERMINISTIC) latexmk $(TEXFLAGS) appendix.tex

pdf: tex submission/proposal.pdf submission/appendix.pdf
	$(PY) scripts/check_pdf.py submission/proposal.pdf --max-pages $(BODY_PAGES) \
	    --paper a4 --min-font 10 $(PDF_GUARD)
	$(PY) scripts/check_pdf.py submission/appendix.pdf --max-pages $(APPENDIX_PAGES) \
	    --paper a4 --min-font 10 $(PDF_GUARD)

# The fitting loop: build and report, never fail.  An over-length PDF must still be
# produced, because deciding what to cut requires looking at it.
pdf-draft: tex
	cd submission && $(DETERMINISTIC) latexmk $(TEXFLAGS) proposal.tex
	-$(PY) scripts/check_pdf.py submission/proposal.pdf --max-pages $(BODY_PAGES) \
	    --paper a4 --min-font 10 --report-only $(PDF_GUARD)

submission: pdf
	$(PY) scripts/assemble_submission.py

clean:
	rm -rf submission/portal .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	cd submission && latexmk -C >/dev/null 2>&1 || true
