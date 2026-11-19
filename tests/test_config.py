from hst.config import SEED, load_settings


def test_seed_and_settings():
    settings = load_settings()
    assert SEED == 1337
    assert settings["seed"] == SEED
