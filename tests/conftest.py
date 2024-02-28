import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


@pytest.fixture(scope="package")
def cfg_pointbev_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model=PointBeV"],
        )

        cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
    return cfg


@pytest.fixture(scope="function")
def cfg_pointbev(cfg_pointbev_global) -> DictConfig:
    cfg = cfg_pointbev_global.copy()

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="package")
def cfg_temporalpointbev_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model=PointBeV_T"],
        )

        cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
    return cfg


@pytest.fixture(scope="function")
def cfg_temporalpointbev(cfg_temporalpointbev_global) -> DictConfig:
    cfg = cfg_temporalpointbev_global.copy()

    yield cfg

    GlobalHydra.instance().clear()
