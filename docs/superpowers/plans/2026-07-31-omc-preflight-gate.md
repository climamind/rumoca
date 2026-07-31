# OMC Preflight Gate Implementation Plan

## Context

`cargo test --workspace` reached `omc_differential_semantics`, spawned the
local `omc` wrapper, and failed only after Docker/containerd returned
`input/output error`. The previous availability check treated any successful
process spawn as an available OMC, so the failure surfaced as a misleading
semantic mismatch. A non-destructive restart of the default Colima profile
restored `/bin/bash`, container extraction, `omc --version`, and the original
differential test, proving the first failing layer was the Colima VM storage
stack rather than OMC or Rumoca semantics.

## Governing sources

- `spec/SPEC_0033_DEVELOPMENT_PROCESS.md` sections 2, 3, and 6: prove the
  first failing layer, fix the owning tooling layer, and verify the smallest
  behavior-proving path first.
- `spec/SPEC_0025_PR_REVIEW_PROCESS.md` section 4 and 4a: keep verification
  Cargo-native and expose the underlying Cargo test while allowing `xtask` to
  orchestrate reusable repository gates.
- `toolchains/openmodelica-version`: canonical OpenModelica package pin.
- `CONTRIBUTING.md`: local and CI OpenModelica versions must match the pinned
  major.minor.patch release before parity results are comparable.

## Global Constraints

- The gate MUST be reusable as `cargo xtask verify omc`.
- Direct `cargo xtask verify msl-parity` MUST invoke the same gate before
  expensive MSL setup; no duplicated OMC probing logic.
- The gate MUST check both the pinned OpenModelica release and execution of a
  real `.mos` smoke script from a mounted workspace directory.
- A missing `omc` executable, a non-zero version command, a version mismatch,
  and a smoke-script failure MUST be distinct failures.
- Docker/containerd `input/output error` output MUST be classified as a
  container-runtime storage failure and include a non-destructive Colima
  restart remedy. The gate MUST NOT restart, prune, delete, or recreate a
  runtime automatically.
- Native Linux/CI OpenModelica installations MUST remain supported; Docker or
  Colima MUST NOT become mandatory.
- The optional workspace differential test may skip only when `omc` cannot be
  spawned because it is absent. A present but unhealthy OMC MUST fail with its
  captured output and point to `cargo xtask verify omc`.
- Tests MUST exercise real temporary executables and process results where
  practical; do not assert only on source text or mocks.
- Update `README.md` and `CONTRIBUTING.md` with the canonical gate and failure
  boundary. Do not create `docs/ARCHITECTURE.md`; this repository has no such
  source of truth and routes architecture/process rules through `spec/`.
- Commit with `git commit -s`; do not push.

### Task 1: Implement and wire the reusable OMC preflight gate

**Files:**

- Create `crates/xtask/src/omc_preflight.rs`.
- Modify `crates/xtask/src/main.rs` and `crates/xtask/src/verify_cmd.rs`.
- Modify `crates/rumoca/tests/omc_differential_semantics.rs`.
- Modify `README.md` and `CONTRIBUTING.md`.
- Include this plan file in the commit.

**RED:**

1. Add tests that run the planned gate against temporary `omc` executables
   for: pinned healthy version plus successful smoke; missing executable;
   non-zero version command with Docker/containerd I/O output; wrong release;
   and smoke-script failure.
2. Add or amend xtask wiring tests proving the standalone `verify omc`
   command exists and `msl-parity` delegates to the same preflight before
   expensive setup.
3. Add a focused test seam for the differential-test prerequisite decision so
   absent OMC is the only skip outcome and unhealthy OMC is an actionable
   failure.
4. Run the focused tests before production implementation and record the
   expected failures caused by the missing gate/behavior.

**GREEN:**

1. Implement one OMC preflight module owned by `xtask`:
   - read and canonicalize `toolchains/openmodelica-version`;
   - run `omc --version`, capture status/stdout/stderr, and require the pinned
     major.minor.patch release;
   - create a temporary workspace-local `.mos` script, execute it with `omc`,
     and require an unambiguous successful result;
   - classify container-runtime storage I/O failures and render the safe
     Colima restart guidance without mutating the runtime.
2. Expose it as `cargo xtask verify omc` and call it at the start of direct
   `cargo xtask verify msl-parity` runs that actually require OMC. A merge-only
   shard fan-in that performs no OMC execution may skip the preflight.
3. Harden `omc_differential_semantics` prerequisite handling: only not-found
   skips; present-but-broken fails with captured diagnostic and the canonical
   gate command.
4. Document the gate and its relationship to Cargo-native tests.

**Verification:**

```bash
cargo test -p xtask omc_preflight -- --nocapture
cargo test -p xtask verify_cmd::tests -- --nocapture
cargo test -p rumoca --test omc_differential_semantics -- --nocapture
cargo xtask verify omc
cargo fmt --all -- --check
cargo clippy -p xtask -p rumoca --all-targets --all-features -- -D warnings
cargo test --workspace -- --test-threads=1
cargo doc --no-deps
git diff --check origin/main...HEAD
```

The exact workspace command must pass with the repaired local OMC runtime. If
the environment regresses, the new gate must fail first with the classified
storage-layer diagnosis; do not hide the failure by skipping the test.
