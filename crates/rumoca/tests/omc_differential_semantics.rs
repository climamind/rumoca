use std::fs;
use std::io;
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

#[derive(Debug, PartialEq, Eq)]
enum OmcPrerequisite {
    SkipMissing,
    Ready,
}

fn omc_prerequisite_decision(result: io::Result<Output>) -> Result<OmcPrerequisite, String> {
    match result {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(OmcPrerequisite::SkipMissing),
        Err(error) => Err(format!("failed to start `omc --version`: {error}")),
        Ok(output) if output.status.success() => Ok(OmcPrerequisite::Ready),
        Ok(output) => Err(format!(
            "`omc --version` failed with status={}.\nstdout:\n{}\nstderr:\n{}\nRun `cargo xtask verify omc` to diagnose the OpenModelica runtime before retrying this differential test.",
            output.status,
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
        )),
    }
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

    let omc = Command::new("omc")
        .arg(&script_path)
        .output()
        .expect("run omc");
    let omc_output = format!(
        "{}{}",
        String::from_utf8_lossy(&omc.stdout),
        String::from_utf8_lossy(&omc.stderr)
    );
    assert!(
        omc.status.success() && omc_output.contains("Variable c not found in scope M"),
        "expected OMC to reject unqualified enclosing-scope lookup, got:\n{omc_output}"
    );

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
