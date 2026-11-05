import pytest


@pytest.fixture(scope="session")
def spark():
    from mlops.data import create_spark

    session = create_spark(app_name="modern-mlops-tests")
    yield session
    session.stop()
