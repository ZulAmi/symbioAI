import time
from monitoring.failure_monitor import FailureMonitor


def test_failure_monitor_basic():
    fm = FailureMonitor(window_size=10, confidence_threshold=0.7, entropy_threshold=1.5, failure_ratio_trigger=0.5, min_failures_trigger=2)
    ctx = {"task_type": "analysis", "domain": "general"}

    # One success
    ev = fm.record(
        model_id="m1",
        request_id="r1",
        inputs={"text": "ok"},
        outputs={"generated_text": "foo", "confidence": 0.9},
        metadata={"confidence": 0.9},
        context=ctx,
        is_failure=False,
    )
    assert ev is None

    # Two failures
    fm.record(
        model_id="m1",
        request_id="r2",
        inputs={"text": "hard"},
        outputs={"generated_text": "bar", "confidence": 0.4},
        metadata={"confidence": 0.4},
        context=ctx,
        is_failure=True,
    )
    fm.record(
        model_id="m1",
        request_id="r3",
        inputs={"text": "hard2"},
        outputs={"generated_text": "baz", "confidence": 0.3},
        metadata={"confidence": 0.3},
        context=ctx,
        is_failure=True,
    )

    # Now the trigger should fire (2 failures out of 3 => 0.66)
    assert fm.should_trigger_finetune("m1", "analysis", "general") is True

    # Build dataset
    ds = fm.build_training_dataset("m1", task_type="analysis", domain="general")
    assert isinstance(ds, list)
    assert len(ds) >= 2