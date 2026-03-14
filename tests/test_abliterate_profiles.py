"""Tests for abliterate profiles."""

from __future__ import annotations

from ollama_forge.abliterate_profiles import get_profile, get_profiles, list_profiles


def test_list_profiles_returns_all() -> None:
    names = list_profiles()
    assert "safe" in names
    assert "balanced" in names
    assert "aggressive" in names
    assert "surgical" in names
    assert "optimized" in names
    assert "nuclear" in names


def test_get_profiles_returns_copies() -> None:
    profiles = get_profiles()
    assert len(profiles) >= 6
    # Modifying returned dict should not affect internal state
    profiles["safe"]["strength"] = 999
    fresh = get_profiles()
    assert fresh["safe"]["strength"] != 999


def test_get_profile_returns_empty_for_none() -> None:
    assert get_profile(None) == {}
    assert get_profile("") == {}


def test_get_profile_case_insensitive() -> None:
    p = get_profile("AGGRESSIVE")
    assert p.get("strength") == 1.3


def test_get_profile_unknown_returns_empty() -> None:
    assert get_profile("nonexistent_profile_xyz") == {}


def test_surgical_has_sparse_surgery() -> None:
    p = get_profile("surgical")
    assert p["sparse_surgery"] is True
    assert p["surgery_top_k"] == 0.3
    assert p["moe_expert_scale"] == 0.4


def test_optimized_has_whitened_svd() -> None:
    p = get_profile("optimized")
    assert p["svd_method"] == "whitened"
    assert p["refine_passes"] == 2


def test_nuclear_has_all_techniques() -> None:
    p = get_profile("nuclear")
    assert p["sparse_surgery"] is True
    assert p["svd_method"] == "whitened"
    assert p["refine_passes"] == 3
    assert p["project_bias"] is True
    assert p["output_only"] is True


def test_all_profiles_have_description() -> None:
    for name, values in get_profiles().items():
        assert "description" in values, f"Profile {name} missing description"
        assert len(values["description"]) > 10


def test_all_profiles_have_core_keys() -> None:
    core_keys = {"num_instructions", "agg", "strength", "norm_preserving"}
    for name, values in get_profiles().items():
        missing = core_keys - set(values.keys())
        assert not missing, f"Profile {name} missing keys: {missing}"
