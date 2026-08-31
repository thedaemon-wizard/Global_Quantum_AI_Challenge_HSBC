# SPDX-License-Identifier: Apache-2.0
#
# On this machine `python3` is 3.9; this project needs 3.12.  Every target therefore uses
# the virtual environment's interpreter directly, and the bootstrap uses python3.12
# explicitly.  tests/test_repo_hygiene.py::test_no_bare_python3_invocation enforces it.

PY      := .venv/bin/python
RUFF    := .venv/bin/ruff
BOOT    := /usr/bin/python3.12

.PHONY: help venv venv-gpu smoke test fast lint baseline conformal quantum mps explain \
        latency predictions summaries seedsweep figures walkthrough freeze derived claims tex \
        pdf pdf-draft \
        submission check reproduce clean

help:
	@echo "venv       create .venv with python3.12 and install pinned dependencies"
	@echo "smoke      S0-S8 environment assertions; any failure stops the build"
	@echo "test       full pytest suite"
	@echo "fast       pytest without the slow cases"
	@echo "lint       ruff"
	@echo "seedsweep  E10 at full scale: 16 fits over 4 bond dimensions x 4 seeds (13 GPU-hours)"
	@echo "baseline   E1 splits and integrity, E2 label-censoring audit, E3 classical baselines"
	@echo "conformal  E4 PSD2 envelope, E5 two-sided risk control, E6 coverage, E7 exchangeability"
	@echo "quantum    E8 a-priori screens, E11 simulator parity, circuit structure"
	@echo "mps        E10 tensor-network arm"
	@echo "explain    E12 feature attribution over the calibration band"
	@echo "latency    E13 per-transaction inference latency against the authorisation budget"
	@echo "predictions per-transaction scores and decisions, the statement's first outcome"
	@echo "summaries  the split-arm, bond-dimension and rolling-origin tables the documents quote"
	@echo "figures    submission figures, generated from the tables"
	@echo "walkthrough trace the certificate end to end against the committed tables"
	@echo "freeze     write the SHA-256 manifest"
	@echo "claims     recompute every number quoted in prose from its source table, and check"
	@echo "           that the markdown mathematics survives GitHub's renderer"
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

# Experiments that were planned and are NOT in these targets, each recorded rather than
# silently dropped.  A target that names a script which does not exist makes `make reproduce`
# fail, which is worse than an honest gap: it turns "not run" into "cannot run".
#
#   E7  drift and exchangeability tests    pre-registered, never run   protocol amendment A7
#   E9  quantum kernel re-ranking          stopped by its own screens  protocol section 9
#   E14 issuer economics                   not started                 docs/protocol.md A2
#   E15 ULB stress case                    blocked, dataset not local  docs/PROVENANCE.md 1.2
#
# E9 is the only one that is a result rather than a gap: the a-priori screens rejected every
# configuration, so the arm was stopped by the stopping rule it was pre-registered under.
#
# E12 and E13 were on this list and are not any more: `explain` and `latency` below run them.

baseline:
	$(PY) scripts/make_splits.py
	$(PY) scripts/audit_labels.py
	$(PY) scripts/run_baselines.py
	$(PY) scripts/run_ablations.py

conformal:
	$(PY) scripts/run_conformal.py
	$(PY) scripts/run_power.py
	$(PY) scripts/summarise_coverage.py
	$(PY) scripts/validate_certificate.py

quantum:
	$(PY) scripts/screen_kernels.py
	$(PY) scripts/check_parity.py
	$(PY) scripts/report_circuits.py

mps:
	$(PY) scripts/run_mps.py

# The full-scale arm, kept out of `reproduce` deliberately: 16 fits at roughly 3,000 s each is
# 13 GPU-hours, against seconds for everything else in the pipeline.  A reviewer checking the
# certificate should not have to spend a day re-deriving the tensor-network result, which the
# frozen manifest already covers.  It is a target so that it *can* be re-run, and named in
# `reproduce`'s closing message so its absence is stated rather than silent.
seedsweep:
	$(PY) scripts/run_seed_sweep.py

explain:
	$(PY) scripts/run_explain.py

latency:
	$(PY) scripts/measure_latency.py

predictions:
	$(PY) scripts/export_predictions.py

# Aggregations of tables the long runs produce.  They are targets rather than inline steps
# because each backs a table in the shipped documents and neither had a producer in the tree
# at all: `make reproduce` carried them forward and the manifest check passed them trivially,
# since a file nothing rewrites cannot differ from its own hash.
summaries:
	$(PY) scripts/summarise_split_arms.py
	$(PY) scripts/summarise_seed_sweep.py
	$(PY) scripts/run_rolling_origin.py

figures:
	$(PY) scripts/make_figures.py

# A reviewer's trace of the certificate against the committed tables.  Reads and asserts; it
# refits nothing, so it costs a second and no GPU.  It is a repository artefact -- the portal
# accepts no .ipynb and all five slots are full.
walkthrough:
	$(PY) notebooks/walkthrough.py

freeze:
	$(PY) scripts/freeze.py

# Tables derived from a document rather than from a run.  Both are regenerated before the
# claim gate because a normal editing session changes their sources.  The decision count went
# stale exactly this way: two entries were appended, the table was not regenerated, and every
# gate passed -- because the gate compares claims.yaml to the table, never the table to the
# document it summarises.  Deriving it inside `claims` is what makes that impossible.
derived:
	$(PY) scripts/summarise_decisions.py
	$(PY) scripts/make_crosscheck.py

claims: derived
	$(PY) scripts/check_claims.py
	$(PY) scripts/check_claims.py --citations
	$(PY) scripts/check_claims.py --unused
	$(PY) scripts/check_protocol.py
	$(PY) scripts/check_markdown_math.py $(KATEX)

# `check` depends on `pdf` because it gates *against* the built PDFs: claims.yaml lists them
# as documents and freeze.py hashes them in the scientific class.  Without the dependency,
# `make reproduce && make check` validates documents built before the tables moved.
# Set KATEX to a directory from which `import katex` resolves to turn on the parse half of
# check_markdown_math: KATEX=/path/to/dir make check. Without it the escape half still runs,
# which is what catches the defect class that broke README section 4.2.
KATEX_DIR ?=
KATEX = $(if $(KATEX_DIR),--render $(KATEX_DIR),)

check: pdf claims
	$(PY) scripts/freeze.py --check

reproduce: baseline conformal quantum mps explain latency predictions summaries figures freeze
	@echo
	@echo "Reproduction complete.  Verify a later run against this one with: make check"
	@echo "Not included: the full-scale tensor-network sweep behind section 4.8, which is"
	@echo "13 GPU-hours.  Run it with: make seedsweep"

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
