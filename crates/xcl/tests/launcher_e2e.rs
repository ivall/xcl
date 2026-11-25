use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn dot_only_identities_keep_configuration_in_a_title_subdirectory() {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root = std::env::temp_dir().join(format!("xcl-launcher-title-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let root = TempDir(root);

    for (tag, identity) in [("normal", "Contoso.Test"), ("empty", ""), ("current", "."), ("parent", ".."), ("dots", "...")] {
        let case = root.0.join(tag);
        let game = case.join("game");
        let data = case.join("data");
        std::fs::create_dir_all(&game).unwrap();
        std::fs::create_dir(&data).unwrap();
        // The missing executable stops the launcher after configuration creation, before starting a host.
        std::fs::write(
            game.join("MicrosoftGame.config"),
            format!(r#"<Game><Identity Name="{identity}"/><ExecutableList><Executable Name="missing.exe"/></ExecutableList></Game>"#),
        ).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("run")
            .arg(&game)
            .env("XCL_DATA_ROOT", &data)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{}", String::from_utf8_lossy(&output.stderr));
        assert!(!case.join("xcl.toml").exists(), "{identity:?} wrote configuration above the data root");
        assert!(!data.join("xcl.toml").exists(), "{identity:?} wrote configuration directly in the data root");
        assert!(
            std::fs::read_dir(&data).unwrap().any(|entry| entry.unwrap().path().join("xcl.toml").is_file()),
            "{identity:?} did not create configuration in a title subdirectory"
        );
        if tag == "normal" {
            assert!(data.join(identity).join("xcl.toml").is_file());
        }
    }
}
