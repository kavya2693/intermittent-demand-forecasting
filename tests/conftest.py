import pytest

from demand_segments.data import GeneratorConfig, generate


@pytest.fixture(scope="session")
def small_data():
    return generate(GeneratorConfig(n_skus=60, n_weeks=120, seed=3))
