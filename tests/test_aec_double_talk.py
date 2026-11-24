import pytest
import media.aec_double_talk as mod

CASES = [
    ("double_talk", (1,)),
    ("clock_drift", (1,)),
    ("delay_reset", (1,)),
    ("nonlinear_suppress", (1,)),
    ("near_gain", (1,)),
    ("determinism", (1,)),
    ("device_switch", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("reference_align", (1,)),
    ("metrics", (1,)),
    ("residual_echo", (1,)),
    ("comfort_noise", (1,)),
    ("profile", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
