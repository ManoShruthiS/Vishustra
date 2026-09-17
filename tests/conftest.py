import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vishustra_core.kernel import VishustraKernel  # noqa: E402


@pytest.fixture(scope="session")
def kernel() -> VishustraKernel:
    return VishustraKernel()


@pytest.fixture
def sample_review() -> str:
    return (
        "Your app is SHIT!! My phone number is +1-555-123-4567 and email john@x.com. "
        "Support took 3 days to reply, the whole experience was absolutely terrible, "
        "and I will never use this product again."
    )