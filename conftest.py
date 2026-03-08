import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="run expensive tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
):
    if config.getoption("--run-expensive"):
        return
    skip_expensive: pytest.MarkDecorator = pytest.mark.skip(
        reason="need --run-expensive option to run"
    )
    for item in items:
        if "expensive" in item.keywords:
            item.add_marker(skip_expensive)
