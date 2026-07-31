use std::fs;
use std::io;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Output};

use tempfile::tempdir_in;

const ENCAPSULATED_SCOPE_SOURCE: &str = r#"
package P
  constant Real c = 1;
  encapsulated model M
    Real x = c;
  end M;
end P;
"#;
const OMC_PREFLIGHT_GUIDANCE: &str = "Run `cargo xtask verify omc` to diagnose the OpenModelica runtime before retrying this differential test.";

#[derive(Debug, PartialEq, Eq)]
enum OmcPrerequisite {
    SkipMissing,
    Ready,
}

fn omc_prerequisite_decision(result: io::Result<Output>) -> Result<OmcPrerequisite, String> {
    match result {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(OmcPrerequisite::SkipMissing),
        Err(error) => Err(format!(
            "failed to start `omc --version`: {error}\n{OMC_PREFLIGHT_GUIDANCE}"
        )),
        Ok(output) if output.status.success() => Ok(OmcPrerequisite::Ready),
        Ok(output) => Err(format!(
            "`omc --version` failed with status={}.\n{}\n{OMC_PREFLIGHT_GUIDANCE}",
            output.status,
            captured_output(&output),
        )),
    }
}

fn omc_semantic_decision(result: io::Result<Output>) -> Result<(), String> {
    let output = result.map_err(|error| {
        format!(
            "failed to run the OMC differential `.mos` script: {error}\n{OMC_PREFLIGHT_GUIDANCE}"
        )
    })?;
    let captured = captured_output(&output);
    if !output.status.success() {
        return Err(format!(
            "OMC differential `.mos` script failed with status={}.\n{captured}\n{OMC_PREFLIGHT_GUIDANCE}",
            output.status,
        ));
    }
    if !captured.contains("Variable c not found in scope M") {
        return Err(format!(
            "OMC differential `.mos` script omitted the expected encapsulated-scope diagnostic.\n{captured}\n{OMC_PREFLIGHT_GUIDANCE}"
        ));
    }
    Ok(())
}

fn captured_output(output: &Output) -> String {
    format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim(),
    )
}

#[test]
fn encapsulated_scope_rejection_matches_omc() {
    match omc_prerequisite_decision(Command::new("omc").arg("--version").output()) {
        Ok(OmcPrerequisite::SkipMissing) => {
            eprintln!("skipping OMC differential test: omc executable not found");
            return;
        }
        Ok(OmcPrerequisite::Ready) => {}
        Err(diagnostic) => panic!("OMC differential-test prerequisite is unhealthy:\n{diagnostic}"),
    }

    let cwd = std::env::current_dir().expect("current dir");
    let dir = tempdir_in(cwd).expect("tempdir in mounted workspace");
    let model_path = dir.path().join("EncapsulatedScope.mo");
    let script_path = dir.path().join("check.mos");
    fs::write(&model_path, ENCAPSULATED_SCOPE_SOURCE).expect("write model");
    fs::write(
        &script_path,
        format!(
            r#"loadFile("{}");
checkModel(P.M);
getErrorString();
"#,
            model_path.display()
        ),
    )
    .expect("write OMC script");

    omc_semantic_decision(Command::new("omc").arg(&script_path).output())
        .unwrap_or_else(|diagnostic| panic!("OMC differential-test run failed:\n{diagnostic}"));

    let rumoca = rumoca::Compiler::new()
        .model("P.M")
        .compile_str(ENCAPSULATED_SCOPE_SOURCE, "EncapsulatedScope.mo");
    let err = rumoca.expect_err("Rumoca should reject the same encapsulated lookup");
    let err_text = format!("{err:?}");
    assert!(
        err_text.contains("unresolved component reference: 'c'"),
        "expected Rumoca unresolved-name diagnostic, got:\n{err_text}"
    );
}

#[cfg(unix)]
#[test]
fn omc_prerequisite_skips_only_when_the_executable_is_absent() {
    let missing = omc_prerequisite_decision(Err(io::Error::new(
        io::ErrorKind::NotFound,
        "omc not found",
    )))
    .expect("missing OMC should be the explicit skip case");
    assert_eq!(missing, OmcPrerequisite::SkipMissing);

    let unhealthy = Command::new("sh")
        .args([
            "-c",
            "printf '%s\\n' 'containerd: input/output error' >&2; exit 23",
        ])
        .output();
    let error = omc_prerequisite_decision(unhealthy)
        .expect_err("present but unhealthy OMC must be actionable");
    assert!(error.contains("status=exit status: 23"), "{error}");
    assert!(error.contains("containerd: input/output error"), "{error}");
    assert!(error.contains("cargo xtask verify omc"), "{error}");
}

#[cfg(unix)]
#[test]
fn omc_prerequisite_points_to_gate_when_present_but_not_executable() {
    let temp = tempfile::tempdir().expect("create temporary OMC directory");
    let omc = temp.path().join("omc");
    fs::write(&omc, "#!/bin/sh\nexit 0\n").expect("write non-executable OMC file");

    let launch = Command::new(&omc).arg("--version").output();
    assert!(
        matches!(launch.as_ref(), Err(error) if error.kind() == io::ErrorKind::PermissionDenied),
        "expected a real permission-denied launch error, got {launch:?}"
    );
    let error = omc_prerequisite_decision(launch)
        .expect_err("present but non-executable OMC must be actionable");
    assert!(error.contains("failed to start `omc --version`"), "{error}");
    assert!(error.contains("cargo xtask verify omc"), "{error}");
}

#[cfg(unix)]
#[test]
fn omc_semantic_run_failure_captures_output_and_points_to_gate() {
    let temp = tempfile::tempdir().expect("create temporary OMC directory");
    let omc = temp.path().join("omc");
    fs::write(
        &omc,
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0~1-gd7e2907'
  exit 0
fi
printf '%s\n' 'recognizable OMC stdout'
printf '%s\n' 'recognizable OMC stderr' >&2
exit 37
"#,
    )
    .expect("write failing OMC executable");
    let mut permissions = fs::metadata(&omc)
        .expect("read OMC executable metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&omc, permissions).expect("make OMC executable");

    let inherited_path = std::env::var("PATH").expect("read inherited PATH");
    let path = format!("{}:{inherited_path}", temp.path().display());
    let output = Command::new(std::env::current_exe().expect("current test executable"))
        .args([
            "encapsulated_scope_rejection_matches_omc",
            "--exact",
            "--nocapture",
        ])
        .env("PATH", path)
        .output()
        .expect("run differential test with fake OMC");
    let diagnostic = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "fake OMC must fail the child test"
    );
    assert!(
        diagnostic.contains("recognizable OMC stdout"),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("recognizable OMC stderr"),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("cargo xtask verify omc"),
        "{diagnostic}"
    );
}
