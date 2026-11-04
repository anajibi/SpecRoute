import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: pulls checkpoints from S3 (~20 GB)")
    config.addinivalue_line("markers", "gpu: needs a working CUDA device")
