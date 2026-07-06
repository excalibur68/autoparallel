# Fast Optimizer Build

AutoParallel builds an optimization problem before solving for placements. Fast
optimizer build reduces the time and memory needed for that construction while
preserving the same placement result.

Fast build is enabled by default.

## Controls

`AP_FAST_BUILD=0` disables the enumeration-time shortcut used during optimizer
construction. This is useful for A/B validation.

```bash
AP_FAST_BUILD=0 python examples/example_autoparallel.py
```

`AP_PARALLEL_BUILD` controls how many worker processes are used for optimizer
edge-cost construction.

```bash
AP_PARALLEL_BUILD=1 python examples/example_autoparallel.py
AP_PARALLEL_BUILD=8 python examples/example_autoparallel.py
```

Use `AP_PARALLEL_BUILD=1` to force serial construction. AutoParallel also falls
back to serial construction when CUDA is already initialized, because forking
after CUDA initialization is not safe in real GPU training processes.

## Validation

To compare fast build with the reference construction path, run the same model
twice and compare the returned placements:

```bash
AP_FAST_BUILD=0 AP_PARALLEL_BUILD=1 python your_autoparallel_script.py
AP_FAST_BUILD=1 AP_PARALLEL_BUILD=1 python your_autoparallel_script.py
```

The placements should match. If they do not, treat it as a correctness bug.
