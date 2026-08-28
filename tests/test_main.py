"""Tests du point d'entrée CLI (main).

Seulement les chemins légers : la validation des arguments et l'erreur de
configuration (qui empêche tout chargement des backends lourds).
"""

from __future__ import annotations

import pytest

from ragifix.main import main


def test_main_missing_config_arg_exits(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ragifix"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_main_invalid_config_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["ragifix", "--config", str(tmp_path / "inexistant.yaml")],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
