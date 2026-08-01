# CLI-142 Structured GPU Initialization Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the single structured GPU initialization mechanism from Draft
PR #13 onto current `origin/main`, so compact initial-equation families are
preserved through Solve IR and are applied and verified before GPU simulation.

**Architecture:** DAE remains the semantic owner of user and fixed-start
initialization equations. Solve IR carries only backend-neutral compact
initialization ownership, ranges, projection metadata, and executable tensor
programs. `rumoca-phase-solve` proves that contract while lowering;
`rumoca-eval-solve` evaluates compact Maps with the same runtime/table context
used by verification; `rumoca-sim` owns GPU settlement/admission; WASM remains a
thin adapter. No codegen target or solver backend may reconstruct lost
initialization semantics.

**Tech Stack:** Rust 2024 workspace, Cargo-native tests, DAE/Solve IR serde,
structured tensor Maps, Rumoca simulation facade, WASM GPU preparation.

## Global Constraints

- Base the recovery on `origin/main=7bde715b96c3a5b7cfff1872443152e833c35910`.
- Historical evidence is Draft PR #13 old owner head
  `841d2dde74356f890e7d378a65448667aa114e22`, final mixed head
  `299e0ede56dadc536da1da42731216992b412821`, and the initialization-only
  corrective head `fe752731045c027087418d5c4e0f7adab26ae209`.
- Do not cherry-pick or merge the mixed owner history. Reconstruct the smallest
  current-main diff and preserve a clean, reviewable task branch.
- Explicitly exclude PR #11 colon-slice lowering, scheduled-event
  provenance/codegen changes, recovery integration, preservation-only merges,
  generated-source patches, validators weakened to accept invalid input, and
  changes owned by any other recovery issue.
- Governing contracts are SPEC_0007 (DAE/Solve ownership and schema),
  SPEC_0029 (IR purity and runtime layering), SPEC_0032 (structured families
  and Map scalar views), SPEC_0033 (first-divergence/root-cause proof), and
  SPEC_0025 (verification and review gate).
- DAE owns semantic initialization equations and typed provenance. Solve IR may
  carry compact executable ownership and range metadata, but IR crates must
  remain pure data/validation and must not evaluate expressions or choose a
  backend.
- Structured initialization must remain O(number of families/ranges), not
  O(number of scalar cells), for the N=50 proof. No scalar-row reassembly or
  string-parsed identity is allowed.
- Every required solver-Y target must have exactly one valid owner: a proven
  direct family, a declared fixed-start range, or an explicit projection path.
  Partial coverage, overlaps, malformed affine ranges, non-finite values,
  unsupported/random/impure operations, and eventful GPU models fail closed
  with the owning source span.
- Apply and verify a direct family with one shared runtime/table context.
  Verification must not observe different external-table or runtime state.
- Preserve current non-GPU initialization behavior and all pre-existing
  affected tests that pass at the base commit.
- TDD evidence already established on the untouched base:
  `cargo test -p rumoca-bind-wasm --all-features
  test_prepare_gpu_simulation_settles_wave_initial_equations -- --nocapture`
  fails with `u[3,3]=0` (1 failed, 68 filtered, exit 101).
- Final commits must include `Signed-off-by` and must not include AI
  `Co-Authored-By` trailers.

---

### Task 1: Reconstruct compact structured initialization end to end

**Files expected to be created or modified:**

- Modify: `crates/rumoca-ir-dae/src/lib.rs`
- Modify: `crates/rumoca-phase-dae/src/initial.rs`
- Modify as required by the DAE schema fixture surface:
  `crates/rumoca-phase-dae/src/algorithm_lowering.rs`,
  `crates/rumoca-phase-dae/src/appendix_b_validation.rs`,
  `crates/rumoca-phase-dae/src/dae_lowering.rs`,
  `crates/rumoca-phase-dae/src/dae_lowering/tests.rs`
- Create: `crates/rumoca-ir-solve/src/initialization_validation.rs`
- Modify: `crates/rumoca-ir-solve/src/lib.rs`
- Modify: `crates/rumoca-ir-solve/src/tests.rs`
- Create: `crates/rumoca-phase-solve/src/gpu_initialization.rs`
- Modify: `crates/rumoca-phase-solve/src/lib.rs`
- Modify: `crates/rumoca-phase-solve/src/lower.rs`
- Modify: `crates/rumoca-phase-solve/src/lower/initial_residual.rs`
- Modify: `crates/rumoca-phase-solve/src/residual_compute_block.rs`
- Modify: `crates/rumoca-phase-solve/src/solve_model.rs`
- Modify: `crates/rumoca-phase-solve/src/tests/gpu_preparation.rs`
- Create if decomposition is still required by SPEC_0021:
  `crates/rumoca-phase-solve/src/projection_plan.rs`
- Create: `crates/rumoca-eval-solve/src/native_map.rs`
- Modify: `crates/rumoca-eval-solve/src/lib.rs`
- Create: `crates/rumoca-sim/src/gpu_initialization.rs`
- Modify: `crates/rumoca-sim/src/lib.rs`
- Modify: `crates/rumoca-sim/src/solve_lowering/direct.rs`
- Modify: `crates/rumoca-sim/src/solve_lowering/entry.rs`
- Modify: `crates/rumoca-bind-wasm/src/gpu_api.rs`
- Modify: `crates/rumoca-bind-wasm/src/tests/simulation_runtime_tests.rs`
- Modify only the directly affected serde/constructor fixtures required by the
  new DAE/Solve schema contract.
- Modify: `spec/SPEC_0007_IR_PIPELINE.md`
- Modify: `spec/SPEC_0029_CRATE_BOUNDARIES.md`
- Modify: `spec/SPEC_0032_RANGE_PRESERVING_TENSORS.md`

**Interfaces:**

- Produces typed DAE initialization provenance distinguishing user equations
  from generated fixed-start equations without parsing `Equation::origin`.
- Produces a versioned `InitializationSolveSystem` with compact direct-family
  ownership and source-spanned required/fixed target ranges.
- Produces fallible compact Map evaluation through `rumoca-eval-solve`, sharing
  the normal runtime and external-table context.
- Produces `rumoca-sim` GPU settlement that admits, applies, and verifies the
  complete initialization contract before exposing `y0`/`p0`.
- Preserves the existing public WASM GPU preparation surface.

- [ ] **Step 1: Preserve the RED and add owner-level failing tests**

Keep the existing Wave2D RED unchanged. Add focused tests before production
changes for:

- structured direct-family preservation and replay;
- N=50 compact/linear metadata;
- descending binder steps;
- required/fixed range completeness and unique ownership;
- overlap, malformed/non-affine range, and source-span failures;
- shared external-table context for apply and verify;
- random/impure operation rejection;
- eventful/non-finite GPU admission rejection.

Run the new owner-level filters and confirm each fails for the intended missing
contract, not from a typo or filtered-out test.

- [ ] **Step 2: Add the versioned DAE and Solve contracts**

Add typed initialization provenance to the DAE initialization partition and
populate it at the DAE producer. Bump the DAE schema because old payloads do not
have equivalent meaning.

Add compact direct-family and source-spanned target-range metadata to Solve IR,
with strict shape/range/coverage validation and a Solve schema bump. Keep IR
logic limited to pure validation/query behavior.

Run:

```bash
cargo test -p rumoca-ir-dae
cargo test -p rumoca-ir-solve
cargo test -p rumoca-phase-dae
```

Expected: PASS, including JSON/bincode round trips, old/future schema rejection,
typed provenance, overlap, malformed range, and incomplete coverage tests.

- [ ] **Step 3: Prove and lower structured initialization in the owning phase**

Teach `rumoca-phase-solve` to preserve source-proven structured initial
families as compact Maps, prove affine replay and target ownership, distinguish
fixed-start coverage, and build the existing algebraic projection contract
without scalar-row recovery. Unsupported or ambiguous families fail with the
first owning span.

Run:

```bash
cargo test -p rumoca-phase-solve gpu_initial -- --nocapture
cargo test -p rumoca-phase-solve gpu_preparation -- --nocapture
cargo test -p rumoca-phase-solve
```

Expected: PASS, including compact N=50, descending binder, unique ownership,
coverage, failure-span, and random/impure rejection tests.

- [ ] **Step 4: Evaluate, apply, and verify the compact contract**

Add native Map evaluation in `rumoca-eval-solve` using the ordinary row
evaluation context. In `rumoca-sim`, admit the compact initialization artifact,
apply direct families once, settle the explicit projection path where required,
and verify residuals using the same context. Reject unsupported runtime features
before returning partially settled vectors.

Keep `rumoca-bind-wasm` a thin adapter that requests GPU preparation and exposes
the settled vectors.

Run:

```bash
cargo test -p rumoca-eval-solve
cargo test -p rumoca-sim --all-features
cargo test -p rumoca-bind-wasm --all-features \
  test_prepare_gpu_simulation_settles_wave_initial_equations -- --nocapture
cargo test -p rumoca-bind-wasm --all-features \
  test_prepare_gpu_simulation_lowers_and_settles_descending_initial_binder -- --nocapture
cargo test -p rumoca-bind-wasm --all-features \
  test_prepare_gpu_simulation_settles_wave_initial_equations_n50_in_linear_budget -- --nocapture
```

Expected: PASS. The original Wave2D center is greater than `0.9`; the N=50
artifact stays compact and within the test's linear-size budget.

- [ ] **Step 5: Synchronize specs and audit scope**

Update only the active specs that describe the final DAE/Solve/runtime
initialization contract. Verify the branch diff contains no scheduled-event
codegen, colon-slice lowering, generated source, baseline promotion, or other
recovery issue.

Run:

```bash
git diff --check origin/main...HEAD
cargo test -p rumoca --test architecture_hardening_test -- --nocapture
```

Expected: PASS; architecture inventory and spec budgets remain valid.

- [ ] **Step 6: Run affected and repository gates**

Run the focused tests first, then:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo doc --no-deps
```

For compiler/simulator semantics, run the SPEC_0025 full MSL gate and
ModelicaTest semantic gate when the local exact OpenModelica/MSL prerequisites
are available. If they are not available, preserve the exact attempted command
and prerequisite failure, and use hosted CI as the authoritative gate. Record
all unrelated existing failures with exact command, test name, exit code, and
base/head comparison.

- [ ] **Step 7: Commit the scoped recovery**

Commit only the plan and CLI-142 mechanism files:

```bash
git commit -s -m "fix(solve): preserve structured GPU initialization"
```

Expected: signed commit(s), no AI co-author trailer, and a diff whose user-facing
scope is only structured GPU initialization recovery.
