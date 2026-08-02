# CLI-142 Direct Map Independence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan.

**Goal:** Close the remaining CLI-142 semantic-admission hole so a compact
direct initialization Map cannot prove itself with `target - target`, a
transitive target-dependent right-hand side, or multiple outputs.

**Architecture:** `rumoca-ir-solve` remains the single pure admission owner.
It proves executable register flow from immutable Solve IR data.
JSON/bincode deserialization and `rumoca-sim` settlement reuse that gate;
runtime code must not maintain a second weaker validator.

**Base:** `5f19c8019dccdd0bb107cd828a3c5e01fd479ab6`

## Global Constraints

- Work only in the existing linked worktree
  `/Users/hechuan/dev-home/worktrees/rumoca/cli-142-structured-gpu-initialization`
  on `recovery/cli-142-structured-gpu-initialization`.
- This is a new implementation/review cycle. Prior reports are evidence, not
  proof that the remaining behavior is correct.
- Fix the earliest owner: the pure semantic gate in `rumoca-ir-solve`.
  Do not add a runtime-only rejection or weaken producer/admission contracts.
- A direct family is admissible only when the terminal residual has exactly
  one target-dependent side. The non-target operand's complete reaching-
  definition closure must not depend on the validated target `LoadY`.
- Direct and transitive dependencies through `Move`, unary, binary, select,
  indexed loads, tables, linear solve, external calls, and every current
  `LinearOp` source form must be handled conservatively and fail closed.
- A compact direct Map must contain exactly one terminal `StoreOutput`.
- Keep the existing SSA/single-definition, definition-before-use, affine
  target, residual-sign, projection DAG, source-span, coverage, and
  complete-or-error guarantees.
- Do not change scheduled-event/codegen behavior, PR #11 colon-slice work,
  structural fallback policy, generated sources, MSL baselines, or other
  recovery scope.
- Update SPEC_0007, SPEC_0029, or SPEC_0032 only if their current contract
  text does not already describe the final mechanism. Keep spec budgets green.
- New commits must be SSH-signed, contain `Signed-off-by`, and contain no AI
  `Co-Authored-By` trailer.

---

### Task 1: Prove target-independent direct residual inputs

**Expected files:**

- Modify: `crates/rumoca-ir-solve/src/direct_map_semantics.rs`
- Modify: `crates/rumoca-ir-solve/src/tests.rs`
- Modify: `crates/rumoca-sim/src/gpu_initialization.rs`
- Modify relevant active specs only if needed.

#### Step 1 — Establish RED

Add owner tests before production edits. Each filter must run a real test and
fail because the current gate accepts the counterfeit:

1. JSON and bincode reject direct `target - target`.
2. JSON and bincode reject an RHS that depends on target through `Move`.
3. Cover at least one deeper chain using unary, binary, or select so the test
   cannot be satisfied by checking only direct operand equality.
4. JSON and bincode reject more than one `StoreOutput`.
5. `rumoca-sim` settlement rejects the direct and transitive counterfeits
   before returning vectors; assert the original unsettled `y0` is not
   returned as success.

Record exact commands, expected assertion failures, and counts in the task
report.

#### Step 2 — Implement the pure semantic proof

In `rumoca-ir-solve`:

1. Reuse the already-built SSA definition map.
2. Prove that exactly one terminal `Sub` operand is the validated target
   register.
3. Walk the other operand's complete reaching-definition graph and reject if
   any path reaches the validated target `LoadY`.
4. Make traversal deterministic, cycle-safe, checked for register ranges and
   fail closed for unsupported or malformed source shapes.
5. Require exactly one `StoreOutput`, and require it to be terminal.
6. Keep `residual_sign` derived from the sole target side.

Do not encode the rule as only `lhs != rhs`; indirect target flow is the root
cause.

#### Step 3 — Verify owner and runtime behavior

Run at minimum:

```bash
cargo test -p rumoca-ir-solve compact_initialization_ -- --nocapture
cargo test -p rumoca-sim --all-features settlement_semantic_gate_rejects_ -- --nocapture
cargo test -p rumoca-ir-solve
cargo test -p rumoca-sim --all-features
```

Then run the affected five-package gate and exact N=50 test:

```bash
cargo test -p rumoca-ir-solve -p rumoca-eval-solve \
  -p rumoca-phase-solve -p rumoca-sim -p rumoca-bind-wasm --all-features

cargo test -p rumoca-bind-wasm --all-features \
  tests::simulation_runtime_tests::test_prepare_gpu_simulation_settles_wave_initial_equations_n50_in_linear_budget \
  -- --exact --nocapture
```

#### Step 4 — Repository gates

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p rumoca --test architecture_hardening_test -- --nocapture
cargo test -p rumoca --test history_policy_test -- --nocapture
cargo test -p rumoca --test spec_budget_test -- --nocapture
cargo doc --workspace --all-features --no-deps
cargo test --workspace -- --test-threads=1 \
  --skip encapsulated_scope_rejection_matches_omc
```

Attempt the exact workspace command as well. If the known Docker/OMC
`/tmp/containerd-mount... input/output error` recurs, preserve the exact
evidence and do not classify the skip-one command as the exact gate.

MSL and ModelicaTest commands remain required when prerequisites are available;
otherwise report the missing source trees and failing `omc --version`
prerequisite precisely.

#### Step 5 — Commit and report

Create one focused SSH-signed DCO commit:

```text
fix(solve): prove direct residual independence
```

The task report must include root cause, competing hypotheses, RED/GREEN
evidence, exact modified scope, excluded scope, commands not run, and residual
risks.
