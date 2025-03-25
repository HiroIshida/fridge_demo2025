import pickle
import time
from pathlib import Path

import pytest

from fd2025.perception.detect_cylinders import model_content_in_fridge

table = {
    1: [
        "input-of-model_content_in_fridge-20231227_131212.pkl",
        "input-of-model_content_in_fridge-20231227_131224.pkl",
        "input-of-model_content_in_fridge-20231227_131234.pkl",
        "input-of-model_content_in_fridge-20231227_131247.pkl",
    ],
    2: [
        "input-of-model_content_in_fridge-20231227_131451.pkl",
        "input-of-model_content_in_fridge-20231227_131513.pkl",
        "input-of-model_content_in_fridge-20231227_131532.pkl",
        "input-of-model_content_in_fridge-20231227_131552.pkl",
        "input-of-model_content_in_fridge-20231227_131602.pkl",
        "input-of-model_content_in_fridge-20231227_131612.pkl",
        "input-of-model_content_in_fridge-20231227_131633.pkl",
        "input-of-model_content_in_fridge-20231227_131723.pkl",
    ],
    3: [
        "input-of-model_content_in_fridge-20231227_131857.pkl",
        "input-of-model_content_in_fridge-20231227_131924.pkl",
        "input-of-model_content_in_fridge-20231227_131939.pkl",
        "input-of-model_content_in_fridge-20231227_131954.pkl",
        "input-of-model_content_in_fridge-20231227_132011.pkl",
        "input-of-model_content_in_fridge-20231227_132031.pkl",
        "input-of-model_content_in_fridge-20231227_132055.pkl",
        "input-of-model_content_in_fridge-20231227_132113.pkl",
        "input-of-model_content_in_fridge-20231227_132134.pkl",
        "input-of-model_content_in_fridge-20231227_132148.pkl",
        "input-of-model_content_in_fridge-20231227_132207.pkl",
        "input-of-model_content_in_fridge-20231227_132227.pkl",
    ],
    4: [
        "input-of-model_content_in_fridge-20231227_132403.pkl",
        "input-of-model_content_in_fridge-20231227_132519.pkl",
        "input-of-model_content_in_fridge-20231227_132540.pkl",
        "input-of-model_content_in_fridge-20231227_132603.pkl",
        "input-of-model_content_in_fridge-20231227_132616.pkl",
        "input-of-model_content_in_fridge-20231227_132639.pkl",
        "input-of-model_content_in_fridge-20231227_132714.pkl",
        "input-of-model_content_in_fridge-20231227_132752.pkl",
    ],
    5: [
        "input-of-model_content_in_fridge-20231227_132911.pkl",
        "input-of-model_content_in_fridge-20231227_132932.pkl",
        "input-of-model_content_in_fridge-20231227_133000.pkl",
        "input-of-model_content_in_fridge-20231227_133021.pkl",
    ],
}

# prepare test caeses
cases = []
for key, values in table.items():
    for value in values:
        full_path = Path(__file__).parent / "data" / "fridge_env_pickles" / value
        cases.append((key, full_path))


@pytest.mark.parametrize("expected, file_path", cases)
def test_content_modeling(expected, file_path):
    with file_path.open("rb") as f:
        args = pickle.load(f)
    ts = time.time()
    cylinders = model_content_in_fridge(*args)
    print(f"elapsed time: {time.time() - ts}")
    success = len(cylinders) == expected
    print(f"expected: {expected}, actual: {len(cylinders)}")
    assert success, f"expected: {expected}, actual: {len(cylinders)}"


if __name__ == "__main__":
    test_content_modeling()
