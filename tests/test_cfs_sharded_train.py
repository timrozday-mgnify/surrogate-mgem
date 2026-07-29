"""Training with the organism axis sharded must still evaluate and save.

`_shard_organisms` is an opt-in speedup (1.8x on the deepset) that only engages
when the caller asks for extra CPU devices. That made it invisible to every other
test, which runs on the default single device -- and the first real run of it
trained a deepset for 2.9 h and then died in `evaluate` with "inconsistent axis
specs: org vs None", because the heads came back sharded and `ds.x_val` did not.
A crash *after* training costs the entire run, so this is the check worth having.

`XLA_FLAGS` is read when jax first initialises its backend, so this has to be a
subprocess: by the time a test function runs, jax is already up.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")

# 3 organisms over 3 devices, so the axis divides and sharding actually engages.
_SCRIPT = textwrap.dedent("""
    import sys
    from cfs.surrogate.train import evaluate, train_value_heads
    sys.path.insert(0, {tests!r})
    from test_cfs_value_head import _synthetic

    import jax
    assert jax.device_count() == 3, jax.device_count()

    ds = _synthetic()
    heads = train_value_heads(ds, arch="deepset", width=16, depth=2, epochs=2,
                              batch=64, lr=1e-2)
    # The bug: these two lines are what `run` does after training, and they are
    # what the sharding broke.
    d = evaluate(heads, ds, arch="deepset")
    assert 0.0 <= d["worst_grad_cosine"] <= 1.0, d["worst_grad_cosine"]
    print("OK")
""")


def test_sharded_training_survives_evaluate(tmp_path):
    script = tmp_path / "shard_check.py"
    script.write_text(_SCRIPT.format(tests=str(__import__("pathlib").Path(__file__).parent)))
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        env={**__import__("os").environ,
             "XLA_FLAGS": "--xla_force_host_platform_device_count=3"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout, proc.stdout + proc.stderr
