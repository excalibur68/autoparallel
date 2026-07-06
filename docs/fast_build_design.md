# Fast Optimizer Build Design

Fast optimizer build changes how AutoParallel constructs the sharding ILP. It
does not change the objective, the feasible placements, or the selected
placement result.

## Equivalence

Enumeration-time redistribute costs are skipped because the optimizer later
computes and stores AutoParallel's own communication costs for every surviving
strategy edge. No solver-visible cost uses the skipped values.

Infinite-cost strategy edges are omitted before PuLP variable creation. This is
equivalent to the previous formulation that created those variables and added
constraints forcing them to zero.

Repeated-subgraph cluster links are stored per node instead of per option. The
option indices are identical between a repeated node and its cluster root, so the
same root variable can be resolved from the node-level link.

`DecisionVar` uses `slots=True` to reduce Python object overhead. This changes
memory layout only.

Edge-cost computation can run in forked workers. Results are consumed in the
same node order as the serial path, and PuLP variables are still assembled in
the parent process.

## Safety

`AP_PARALLEL_BUILD=1` forces the serial path.

When CUDA is already initialized, AutoParallel uses the serial path even if
`AP_PARALLEL_BUILD` requests multiple workers. This avoids fork-after-CUDA
failures in real training runs.

`AP_FAST_BUILD=0` disables the enumeration-time shortcut for A/B validation.

## Test Strategy

The fast-build regression test compares public optimizer behavior between the
reference and fast paths: the optimizer must solve successfully and return the
same placements. It does not rely on private optimizer data structures such as
PuLP variables or decision-variable key sets.
