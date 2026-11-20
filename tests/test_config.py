from hst.config import SEED, load_settings, load_sources


def test_seed_and_settings():
    settings = load_settings()
    assert SEED == 1337
    assert settings["seed"] == SEED


def test_sources_are_verified_candidates_and_checked():
    sources = load_sources()
    assert 25 <= len(sources) <= 40
    assert all(source.tos_checked for source in sources if source.enabled)
