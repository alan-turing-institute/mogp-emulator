import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fpga", action="store_true", default=False, help="Run FPGA tests"
        )


def pytest_configure(config):
    config.addinivalue_line("markers", "fpga: mark test as involving FPGAs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fpga"):
        return
    skip_fpga = pytest.mark.skip(reason="FPGA test, use --fpga to run")
    for item in items:
        if "fpga" in item.keywords:
            item.add_marker(skip_fpga)
