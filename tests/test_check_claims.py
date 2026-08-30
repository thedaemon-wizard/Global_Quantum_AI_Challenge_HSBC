# SPDX-License-Identifier: Apache-2.0
"""The claim resolver and its ``--update`` formatting.

``--update`` rewrites measured values in place, so a formatting bug in it silently replaces a
measurement with a wrong number that then compares equal to itself forever. It has done this
twice: a YAML round-trip once deleted every comment in the file, and formatting a value of
3.4e-15 with three decimal places once wrote ``0.0000000``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_claims", REPO / "scripts" / "check_claims.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_claims"] = module
    spec.loader.exec_module(module)
    return module


check_claims = _load()


# ------------------------------------------------------------------ precision and notation

@pytest.mark.parametrize(
    ("template", "expected"),
    [("1.494", 3), ("0.0222", 4), ("41", 0), ("3.442e-15", 3), ("1e-9", 0), ("2.5E-3", 1)],
)
def test_precision_is_read_from_how_the_value_is_written(template, expected) -> None:
    """For exponent notation the significant digits are what is printed, not the decimals."""
    assert check_claims._decimals(template) == expected


def test_a_scientific_value_is_not_flattened_to_zero() -> None:
    """The regression: ``f"{3.442e-15:.7f}"`` is ``0.0000000``, which is not the measurement."""
    assert check_claims._format_like(3.442e-15, "3.442e-15") == "3.442e-15"
    assert float(check_claims._format_like(2.887e-15, "3.442e-15")) == pytest.approx(2.887e-15)


def test_a_decimal_value_keeps_its_places() -> None:
    """Plain rounding writes 0.02 where the claim said 0.0222, loosening the next comparison."""
    assert check_claims._format_like(0.02218, "0.0222") == "0.0222"
    assert check_claims._format_like(43.0, "41") == "43"


def test_rounding_matches_the_notation_it_will_be_written_in() -> None:
    assert check_claims._round_like(3.4421e-15, "3.442e-15") == pytest.approx(3.442e-15)
    assert check_claims._round_like(1.4938, "1.494") == pytest.approx(1.494)


# ----------------------------------------------------------------------- derived claims

def test_a_derived_claim_is_a_ratio_of_two_resolved_claims() -> None:
    claim = {"key": "R", "derived": {"op": "ratio", "of": ["A", "B"]}}
    assert check_claims.resolve_derived(claim, {"A": 8.0, "B": 2.0}) == pytest.approx(4.0)


def test_a_derived_claim_can_take_a_power() -> None:
    """The sample-size factor is a squared ratio, because detectable effect scales n^-1/2."""
    claim = {"key": "R", "derived": {"op": "ratio", "of": ["A", "B"], "power": 2}}
    assert check_claims.resolve_derived(claim, {"A": 6.0, "B": 2.0}) == pytest.approx(9.0)


def test_a_derived_claim_refuses_an_unknown_input() -> None:
    claim = {"key": "R", "derived": {"op": "ratio", "of": ["A", "missing"]}}
    with pytest.raises(check_claims.ClaimError, match="missing"):
        check_claims.resolve_derived(claim, {"A": 1.0})


def test_a_derived_claim_refuses_to_divide_by_zero() -> None:
    """A zero denominator means an upstream claim collapsed; inf is not a finding."""
    claim = {"key": "R", "derived": {"op": "ratio", "of": ["A", "B"]}}
    with pytest.raises(check_claims.ClaimError, match="zero"):
        check_claims.resolve_derived(claim, {"A": 1.0, "B": 0.0})


def test_an_unknown_derived_operation_is_refused() -> None:
    claim = {"key": "R", "derived": {"op": "product", "of": ["A", "B"]}}
    with pytest.raises(check_claims.ClaimError, match="product"):
        check_claims.resolve_derived(claim, {"A": 1.0, "B": 2.0})


# ------------------------------------------------------------------- reductions over a table

MARGIN_TABLE = "budget_ms,network_ms,keep\n300,130,True\n300,130,False\n300,200,True\n"


def _margin_claim(name: str, boolean_filter: str | None = None) -> dict:
    reduce_spec: dict = {"op": "min_margin", "columns": ["budget_ms", "network_ms"]}
    if boolean_filter is not None:
        reduce_spec["where"] = boolean_filter
    return {"source": name, "reduce": reduce_spec}


def test_min_margin_without_a_filter_spans_the_whole_table(tmp_path, monkeypatch) -> None:
    """The filter is optional: a margin may have no subset to restrict to.

    Regression for the latency budget, whose residual is the gap between two columns that are
    constant down the table. Requiring ``where`` there would force a boolean column into the
    table for no reason but to satisfy the reducer.
    """
    (tmp_path / "margin.csv").write_text(MARGIN_TABLE)
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    # Rows give margins of 170, 170 and 100; unfiltered, the smallest is 100.
    assert check_claims.resolve(_margin_claim("margin.csv")) == pytest.approx(100.0)


def test_min_margin_still_honours_a_filter_when_given_one(tmp_path, monkeypatch) -> None:
    """The other direction, so making the filter optional cannot silently disable it."""
    (tmp_path / "margin.csv").write_text(
        "budget_ms,network_ms,keep\n300,130,False\n300,140,True\n"
    )
    monkeypatch.setattr(check_claims, "REPO", tmp_path)
    # The 170 row is filtered out, leaving 160 -- which an ignored filter could not produce.
    assert check_claims.resolve(_margin_claim("margin.csv", "keep")) == pytest.approx(160.0)


# ------------------------------------------------------------------- the committed file

def test_every_claim_in_the_repository_resolves() -> None:
    """The gate itself, so a broken claim fails the suite and not only the build."""
    assert check_claims.main([]) == 0
