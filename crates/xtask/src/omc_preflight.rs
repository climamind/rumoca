use anyhow::{Context, Result, bail, ensure};
use std::ffi::OsStr;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::process::{Command, Output};

use crate::util::split_path_with_indices;

const PIN_PATH: &str = "toolchains/openmodelica-version";
const SMOKE_MARKER: &str = "OMC_PREFLIGHT_OK";
const SMOKE_SCRIPT: &str = r#"getVersion();
print("OMC_PREFLIGHT_OK\n");
"#;

pub(crate) fn run(root: &Path) -> Result<()> {
    run_with_program(root, Path::new("omc"))
}

fn run_with_program(root: &Path, program: &Path) -> Result<()> {
    run_with_program_arguments(root, program, &[])
}

fn run_with_program_arguments(
    root: &Path,
    program: &Path,
    program_arguments: &[&OsStr],
) -> Result<()> {
    let expected_identity = pinned_runtime_identity(root)?;
    let version = run_version_command(program, program_arguments)?;
    let version_output = captured_output(&version);
    ensure!(
        version.status.success(),
        "OpenModelica version command failed (status={}).{}",
        version.status,
        failure_guidance(&version_output),
    );

    let observed_identity =
        runtime_identity_from_version_output(&version_output).with_context(|| {
            format!(
                "could not identify an exact OpenModelica runtime identity from:{}",
                format_output(&version_output)
            )
        })?;
    ensure!(
        observed_identity == expected_identity,
        "OpenModelica runtime identity mismatch: expected {expected_identity}, observed {observed_identity}.{}",
        format_output(&version_output),
    );

    run_workspace_smoke(root, program, program_arguments)
}

fn pinned_runtime_identity(root: &Path) -> Result<String> {
    let pin_path = root.join(PIN_PATH);
    let pin = fs::read_to_string(&pin_path)
        .with_context(|| format!("failed to read OpenModelica pin {}", pin_path.display()))?;
    runtime_identity_from_package_pin(&pin)
        .with_context(|| format!("invalid OpenModelica pin {}", pin_path.display()))
}

fn runtime_identity_from_package_pin(raw: &str) -> Result<String> {
    let package_pin = raw.trim();
    let (identity, debian_revision) = package_pin
        .rsplit_once('-')
        .context("expected a Debian package revision suffix")?;
    ensure!(
        !debian_revision.is_empty() && debian_revision.bytes().all(|byte| byte.is_ascii_digit()),
        "invalid Debian package revision `{debian_revision}`"
    );
    parse_runtime_identity(identity)?;
    Ok(identity.to_owned())
}

fn parse_runtime_identity(identity: &str) -> Result<()> {
    let (release, git_revision) = identity
        .split_once("~1-g")
        .context("expected `major.minor.patch~1-g<git-revision>`")?;
    let mut components = split_path_with_indices(release).into_iter();
    release_component(components.next(), "major")?;
    release_component(components.next(), "minor")?;
    release_component(components.next(), "patch")?;
    ensure!(
        components.next().is_none(),
        "expected exactly major.minor.patch, got `{release}`"
    );
    ensure!(
        !git_revision.is_empty() && git_revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid OpenModelica git revision `{git_revision}`"
    );
    Ok(())
}

fn release_component(component: Option<&str>, label: &str) -> Result<u64> {
    let component = component.unwrap_or_default();
    ensure!(
        !component.is_empty() && component.bytes().all(|byte| byte.is_ascii_digit()),
        "invalid {label} release component `{component}`"
    );
    component
        .parse::<u64>()
        .with_context(|| format!("invalid {label} release component `{component}`"))
}

fn command_with_arguments(program: &Path, program_arguments: &[&OsStr]) -> Command {
    let mut command = Command::new(program);
    command.args(program_arguments);
    command
}

fn run_version_command(program: &Path, program_arguments: &[&OsStr]) -> Result<Output> {
    command_with_arguments(program, program_arguments)
        .arg("--version")
        .output()
        .map_err(|error| match error.kind() {
            ErrorKind::NotFound => anyhow::anyhow!(
                "OpenModelica executable `{}` was not found. Install the pinned OpenModelica release, then run `cargo xtask verify omc`.",
                program.display()
            ),
            _ => anyhow::anyhow!(
                "failed to start OpenModelica executable `{}` for `--version`: {error}",
                program.display()
            ),
        })
}

fn runtime_identity_from_version_output(output: &str) -> Result<String> {
    let line = output
        .lines()
        .find(|line| line.trim_start().starts_with("OpenModelica"))
        .unwrap_or_default();
    let version = line
        .trim_start()
        .strip_prefix("OpenModelica")
        .unwrap_or_default()
        .trim_start();
    let identity = version.split_whitespace().next().unwrap_or_default();
    parse_runtime_identity(identity)?;
    Ok(identity.to_owned())
}

fn run_workspace_smoke(root: &Path, program: &Path, program_arguments: &[&OsStr]) -> Result<()> {
    let workspace_root = root
        .canonicalize()
        .with_context(|| format!("failed to canonicalize workspace root {}", root.display()))?;
    let smoke_dir = tempfile::tempdir_in(&workspace_root).with_context(|| {
        format!(
            "failed to create workspace-local OMC smoke directory in {}",
            workspace_root.display()
        )
    })?;
    let script = smoke_dir.path().join("rumoca-omc-preflight.mos");
    fs::write(&script, SMOKE_SCRIPT)
        .with_context(|| format!("failed to write OMC smoke script {}", script.display()))?;
    let output = command_with_arguments(program, program_arguments)
        .arg(&script)
        .current_dir(&workspace_root)
        .output()
        .with_context(|| format!("failed to run OMC smoke script {}", script.display()))?;
    let captured = captured_output(&output);
    if !output.status.success() {
        bail!(
            "OpenModelica smoke script failed (status={}).{}",
            output.status,
            failure_guidance(&captured),
        );
    }
    ensure!(
        captured.contains(SMOKE_MARKER),
        "OpenModelica smoke script completed without `{SMOKE_MARKER}`.{}",
        format_output(&captured),
    );
    Ok(())
}

fn captured_output(output: &Output) -> String {
    format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn format_output(output: &str) -> String {
    format!("\nCaptured output:\n{output}")
}

fn failure_guidance(output: &str) -> String {
    if is_container_runtime_storage_failure(output) {
        format!(
            "\nDetected container-runtime storage I/O failure. Restart the default Colima profile without deleting data: `colima stop && colima start`; then rerun `cargo xtask verify omc`.{}",
            format_output(output)
        )
    } else {
        format_output(output)
    }
}

fn is_container_runtime_storage_failure(output: &str) -> bool {
    let output = output.to_ascii_lowercase();
    output.contains("input/output error")
        && (output.contains("docker") || output.contains("containerd"))
}

#[cfg(all(test, unix))]
mod tests {
    use super::{run_with_program, run_with_program_arguments};
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const PIN: &str = "1.27.0~1-gd7e2907-1";
    const SYSTEM_SHELL: &str = "/bin/sh";

    fn test_root() -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("create workspace root");
        fs::create_dir_all(root.path().join("toolchains")).expect("create toolchains dir");
        fs::write(root.path().join("toolchains/openmodelica-version"), PIN)
            .expect("write OpenModelica pin");
        root
    }

    fn write_omc(root: &Path, name: &str, body: &str) -> PathBuf {
        write_omc_with_staged_executable(root, name, body, |_, _| {})
    }

    fn run_with_omc_script(root: &Path, script: &Path) -> anyhow::Result<()> {
        run_with_program_arguments(root, Path::new(SYSTEM_SHELL), &[script.as_os_str()])
    }

    fn write_omc_with_staged_executable(
        root: &Path,
        name: &str,
        body: &str,
        inspect_staged: impl FnOnce(&Path, &Path),
    ) -> PathBuf {
        let path = root.join(name);
        let mut staged = tempfile::NamedTempFile::new_in(root).expect("create staged fake omc");
        staged
            .write_all(format!("#!/bin/sh\n{body}\n").as_bytes())
            .expect("write staged fake omc");
        let mut permissions = staged
            .as_file()
            .metadata()
            .expect("read staged fake omc metadata")
            .permissions();
        permissions.set_mode(0o755);
        staged
            .as_file()
            .set_permissions(permissions)
            .expect("make staged fake omc executable");
        let staged_path = staged.into_temp_path();
        inspect_staged(staged_path.as_ref(), &path);
        let staged_path = staged_path.keep().expect("keep closed staged fake omc");
        fs::rename(&staged_path, &path).expect("atomically publish fake omc");
        path
    }

    #[test]
    fn fake_omc_is_published_only_after_its_staged_writer_closes() {
        let root = test_root();
        let omc = write_omc_with_staged_executable(
            root.path(),
            "staged-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0~1-gd7e2907'
  exit 0
fi
printf '%s\n' 'OMC_PREFLIGHT_OK'
"#,
            |staged, published| {
                assert!(staged.is_file(), "staged executable must exist");
                assert!(
                    !published.exists(),
                    "published executable must stay hidden until publication"
                );
                let output = Command::new(SYSTEM_SHELL)
                    .arg(staged)
                    .arg("--version")
                    .output()
                    .expect("run closed staged fake OMC script");
                assert!(output.status.success(), "{output:?}");
            },
        );

        assert!(omc.is_file(), "published executable must exist");
        let output = Command::new(SYSTEM_SHELL)
            .arg(&omc)
            .arg("--version")
            .output()
            .expect("run published fake OMC script");
        assert!(output.status.success(), "{output:?}");
    }

    #[test]
    fn fake_omc_script_runs_through_the_system_shell() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "shell-runner-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0~1-gd7e2907'
  exit 0
fi
printf '%s\n' 'OMC_PREFLIGHT_OK'
"#,
        );

        run_with_omc_script(root.path(), &omc)
            .expect("system shell must run the fake OMC script as data");
    }

    #[test]
    fn accepts_pinned_healthy_version_and_workspace_smoke_script() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "healthy-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0~1-gd7e2907'
  exit 0
fi
case "$1" in
  "$PWD"/*.mos) ;;
  *) printf '%s\n' 'smoke script was not workspace-local' >&2; exit 13 ;;
esac
printf '%s\n' 'OMC_PREFLIGHT_OK'
"#,
        );

        run_with_omc_script(root.path(), &omc).expect("healthy OMC must pass preflight");
    }

    #[test]
    fn reports_a_missing_omc_executable() {
        let root = test_root();
        let error = run_with_program(root.path(), &root.path().join("missing-omc"))
            .expect_err("missing OMC must fail preflight");

        assert!(error.to_string().contains("not found"), "{error:#}");
    }

    #[test]
    fn classifies_container_runtime_storage_io_from_version_failure() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "storage-io-omc",
            r#"
printf '%s\n' 'docker: failed to create task: containerd: input/output error' >&2
exit 23
"#,
        );

        let error = run_with_omc_script(root.path(), &omc)
            .expect_err("container storage failure must fail preflight");
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("container-runtime storage"),
            "{diagnostic}"
        );
        assert!(
            diagnostic.contains("colima stop && colima start"),
            "{diagnostic}"
        );
    }

    #[test]
    fn rejects_a_wrong_openmodelica_release() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "wrong-release-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.26.3~1-gd7e2907'
  exit 0
fi
printf '%s\n' 'OMC_PREFLIGHT_OK'
"#,
        );

        let error = run_with_omc_script(root.path(), &omc)
            .expect_err("wrong OMC release must fail preflight");
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("runtime identity mismatch"),
            "{diagnostic}"
        );
        assert!(
            diagnostic.contains("expected 1.27.0~1-gd7e2907"),
            "{diagnostic}"
        );
        assert!(
            diagnostic.contains("observed 1.26.3~1-gd7e2907"),
            "{diagnostic}"
        );
    }

    #[test]
    fn reports_smoke_script_failure_after_a_healthy_version() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "smoke-failure-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0~1-gd7e2907'
  exit 0
fi
printf '%s\n' 'fake smoke failure' >&2
exit 19
"#,
        );

        let error =
            run_with_omc_script(root.path(), &omc).expect_err("smoke failure must fail preflight");
        let diagnostic = format!("{error:#}");
        assert!(diagnostic.contains("smoke script failed"), "{diagnostic}");
        assert!(diagnostic.contains("fake smoke failure"), "{diagnostic}");
    }

    #[test]
    fn rejects_plain_semver_version_output() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "plain-semver-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0'
  exit 0
fi
printf '%s\n' 'OMC_PREFLIGHT_OK'
"#,
        );

        let error = run_with_omc_script(root.path(), &omc)
            .expect_err("plain semver must not satisfy the pinned runtime identity");
        assert!(
            format!("{error:#}").contains("OpenModelica 1.27.0"),
            "{error:#}"
        );
    }

    #[test]
    fn rejects_a_wrong_openmodelica_git_revision() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "wrong-revision-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0~1-gbadc0de'
  exit 0
fi
printf '%s\n' 'OMC_PREFLIGHT_OK'
"#,
        );

        let error = run_with_omc_script(root.path(), &omc)
            .expect_err("wrong git revision must not satisfy the pinned runtime identity");
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("runtime identity mismatch"),
            "{diagnostic}"
        );
        assert!(diagnostic.contains("gbadc0de"), "{diagnostic}");
    }

    #[test]
    fn rejects_a_development_openmodelica_build() {
        let root = test_root();
        let omc = write_omc(
            root.path(),
            "development-build-omc",
            r#"
if [ "$1" = "--version" ]; then
  printf '%s\n' 'OpenModelica 1.27.0~dev-123'
  exit 0
fi
printf '%s\n' 'OMC_PREFLIGHT_OK'
"#,
        );

        let error = run_with_omc_script(root.path(), &omc)
            .expect_err("development build must not satisfy the pinned runtime identity");
        assert!(
            format!("{error:#}").contains("OpenModelica 1.27.0~dev-123"),
            "{error:#}"
        );
    }
}
