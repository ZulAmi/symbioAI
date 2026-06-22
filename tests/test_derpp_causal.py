"""Tests for DerppCausal argument parsing and defaults — Mammoth mocked out."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─── Mock Mammoth before importing DerppCausal ───────────────────────────────
# DerppCausal inherits from Derpp (mammoth). We replace the whole mammoth tree
# with mocks so no Mammoth installation is required.

_mammoth = MagicMock()

# Derpp.get_parser must return a real ArgumentParser for our arg-parsing tests.
def _derpp_get_parser(parser):
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--minibatch_size", type=int, default=128)
    return parser

_mammoth.models.derpp.Derpp = MagicMock()
_mammoth.models.derpp.Derpp.get_parser = staticmethod(_derpp_get_parser)

sys.modules.setdefault("mammoth", _mammoth)
sys.modules.setdefault("mammoth.models", _mammoth.models)
sys.modules.setdefault("mammoth.models.derpp", _mammoth.models.derpp)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Import under test ───────────────────────────────────────────────────────

from training.derpp_causal import DerppCausal  # noqa: E402


# ─── Argument parser tests ───────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return DerppCausal.get_parser(parser)


def test_get_parser_returns_parser():
    p = _make_parser()
    assert isinstance(p, argparse.ArgumentParser)


def test_default_causality_off():
    p = _make_parser()
    args = p.parse_args([])
    assert args.use_causal_sampling == 0
    assert args.enable_causal_graph_learning == 0


def test_causal_sampling_flag():
    p = _make_parser()
    args = p.parse_args(["--use_causal_sampling", "3"])
    assert args.use_causal_sampling == 3


def test_true_micro_steps_default():
    p = _make_parser()
    args = p.parse_args([])
    assert args.true_micro_steps == 2


def test_causal_eval_interval_default():
    p = _make_parser()
    args = p.parse_args([])
    assert args.causal_eval_interval == 5


def test_causal_hybrid_candidates_default():
    p = _make_parser()
    args = p.parse_args([])
    assert args.causal_hybrid_candidates == 200


def test_derpp_causal_alias():
    """DERPPCausal must be an alias for DerppCausal for backward compat."""
    from training.derpp_causal import DERPPCausal
    assert DERPPCausal is DerppCausal


def test_name_attribute():
    assert DerppCausal.NAME == "derpp-causal"


def test_compatibility_includes_class_il():
    assert "class-il" in DerppCausal.COMPATIBILITY
