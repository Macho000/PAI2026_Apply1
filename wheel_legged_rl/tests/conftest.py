import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "wheel_legged_rl"))

FIXTURES = Path(__file__).parent / "fixtures"
GO2W_URDF = Path(
    os.environ.get(
        "WL_GO2W_URDF",
        ROOT / "third_party" / "unitree_ros" / "robots" / "go2w_description" / "urdf" / "go2w_description.urdf",
    )
)


@pytest.fixture
def go2w_urdf() -> Path:
    if not GO2W_URDF.is_file():
        pytest.skip("Go2-W description not fetched (scripts/fetch_go2w_description.sh)")
    return GO2W_URDF


@pytest.fixture
def toy_urdf() -> Path:
    return FIXTURES / "toy_wheel_leg.urdf"
