"""Tests unitaires de `ragifix.vectorstore.base.matches_filters`."""

from __future__ import annotations

from ragifix.vectorstore.base import matches_filters


def test_no_filters_matches_everything():
    assert matches_filters({"source": "a"}, None) is True
    assert matches_filters({"source": "a"}, {}) is True


def test_scalar_equality():
    assert matches_filters({"source": "sharepoint"}, {"source": "sharepoint"}) is True
    assert matches_filters({"source": "sharepoint"}, {"source": "datas_locales"}) is False


def test_list_is_or():
    filters = {"source": ["datas_locales", "sharepoint"]}
    assert matches_filters({"source": "sharepoint"}, filters) is True
    assert matches_filters({"source": "autre"}, filters) is False


def test_missing_metadata_key_does_not_match():
    assert matches_filters({}, {"extension": "pdf"}) is False


def test_filename_glob_scalar_and_list():
    assert matches_filters({"filename": "rapport_2025.pdf"}, {"filename_glob": "*rapport*.pdf"}) is True
    assert matches_filters({"filename": "notes.txt"}, {"filename_glob": "*rapport*.pdf"}) is False
    assert (
        matches_filters(
            {"filename": "notes.txt"}, {"filename_glob": ["*rapport*.pdf", "*notes*.txt"]}
        )
        is True
    )


def test_modified_after_before():
    metadata = {"modified_at": "2025-06-15T00:00:00Z"}
    assert matches_filters(metadata, {"modified_after": "2025-01-01"}) is True
    assert matches_filters(metadata, {"modified_after": "2025-12-01"}) is False
    assert matches_filters(metadata, {"modified_before": "2025-12-01"}) is True
    assert matches_filters(metadata, {"modified_before": "2025-01-01"}) is False
    assert matches_filters({}, {"modified_after": "2025-01-01"}) is False


def test_combined_filters_are_and():
    metadata = {"source": "sharepoint", "extension": "pdf", "filename": "rapport.pdf"}
    assert matches_filters(metadata, {"source": "sharepoint", "extension": "pdf"}) is True
    assert matches_filters(metadata, {"source": "sharepoint", "extension": "txt"}) is False
