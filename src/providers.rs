use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SavedProvider {
    #[serde(default)]
    pub name: String,
    pub base_url: String,
    pub api_key: String,
}

pub(crate) fn config_path() -> Result<PathBuf> {
    let directory = if cfg!(target_os = "windows") {
        PathBuf::from(std::env::var_os("APPDATA").context("APPDATA is not set")?).join("Harness")
    } else if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?)
            .join("Library/Application Support/Harness")
    } else {
        match std::env::var_os("XDG_CONFIG_HOME") {
            Some(directory) => PathBuf::from(directory),
            None => {
                PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?).join(".config")
            }
        }
        .join("harness")
    };
    Ok(directory.join("providers.json"))
}

pub(crate) fn load(path: &Path) -> Result<Vec<SavedProvider>> {
    let contents = match fs::read(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("could not read {}", path.display()));
        }
    };
    let mut providers: Vec<SavedProvider> = serde_json::from_slice(&contents)
        .with_context(|| format!("invalid provider config in {}", path.display()))?;
    for provider in &mut providers {
        provider.base_url = validate_url(&provider.base_url).map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            !provider.api_key.trim().is_empty() && !provider.api_key.contains(";;"),
            "invalid API key in provider config"
        );
    }
    Ok(providers)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SavedModel {
    pub model: String,
    pub base_url: Option<String>,
}

pub(crate) fn load_model(path: &Path) -> Result<Option<SavedModel>> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ModelConfig {
        Legacy(String),
        Selection(SavedModel),
    }

    match fs::read(path) {
        Ok(contents) => {
            let config: ModelConfig = serde_json::from_slice(&contents)
                .with_context(|| format!("invalid model config in {}", path.display()))?;
            let selection = match config {
                ModelConfig::Legacy(model) => SavedModel {
                    model,
                    base_url: None,
                },
                ModelConfig::Selection(selection) => selection,
            };
            Ok((!selection.model.trim().is_empty()).then_some(selection))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("could not read {}", path.display())),
    }
}

pub(crate) fn save_model(path: &Path, model: &str, base_url: Option<&str>) -> Result<()> {
    save_json(
        path,
        &SavedModel {
            model: model.to_owned(),
            base_url: base_url.map(str::to_owned),
        },
    )
}

pub(crate) fn save(path: &Path, providers: &[SavedProvider]) -> Result<()> {
    save_json(path, &providers)
}

pub(crate) fn load_reasoning_effort(path: &Path) -> Result<Option<String>> {
    let contents = match fs::read(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("could not read {}", path.display()));
        }
    };
    let effort: String = serde_json::from_slice(&contents)
        .with_context(|| format!("invalid reasoning effort config in {}", path.display()))?;
    anyhow::ensure!(
        crate::constants::REASONING_OPTIONS.contains(&effort.as_str()),
        "invalid reasoning effort in {}",
        path.display()
    );
    Ok(Some(effort))
}

pub(crate) fn save_reasoning_effort(path: &Path, effort: &str) -> Result<()> {
    anyhow::ensure!(
        crate::constants::REASONING_OPTIONS.contains(&effort),
        "invalid reasoning effort: {effort}"
    );
    save_json(path, &effort)
}

fn save_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let directory = path
        .parent()
        .context("provider config has no parent directory")?;
    fs::create_dir_all(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
        if path.exists() {
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
    }
    let contents = serde_json::to_vec_pretty(value)?;
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(&contents)?;
    file.as_file().sync_all()?;
    file.persist(path)
        .with_context(|| format!("could not save {}", path.display()))?;
    Ok(())
}

pub(crate) fn validate_url(input: &str) -> Result<String, String> {
    let input = input.trim();
    let url = reqwest::Url::parse(input).map_err(|_| {
        "Enter an absolute provider URL, such as https://api.example.com/v1.".to_owned()
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("Provider URL must use http or https and include a host.".to_owned());
    }
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || input.contains(";;")
    {
        return Err(
            "Enter one provider base URL without credentials, a query, or a fragment.".to_owned(),
        );
    }
    Ok(url.as_str().trim_end_matches('/').to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_survives_restarts_and_rejects_invalid_values() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("harness/last-reasoning-effort.json");
        assert_eq!(load_reasoning_effort(&path).unwrap(), None);
        for effort in crate::constants::REASONING_OPTIONS {
            save_reasoning_effort(&path, effort).unwrap();
            assert_eq!(
                load_reasoning_effort(&path).unwrap().as_deref(),
                Some(*effort)
            );
        }
        assert!(save_reasoning_effort(&path, "unknown").is_err());
        assert_eq!(
            load_reasoning_effort(&path).unwrap().as_deref(),
            Some("max")
        );
        for invalid in ["invalid json", r#""unknown""#, "null"] {
            fs::write(&path, invalid).unwrap();
            assert!(load_reasoning_effort(&path).is_err());
        }
    }

    #[test]
    fn last_selected_model_survives_restarts_and_updates() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("harness/last-model.json");
        assert_eq!(load_model(&path).unwrap(), None);
        save_model(&path, "first-model", None).unwrap();
        assert_eq!(
            load_model(&path)
                .unwrap()
                .as_ref()
                .map(|selection| selection.model.as_str()),
            Some("first-model")
        );
        save_model(&path, "last-model", Some("https://provider.example/v1")).unwrap();
        assert_eq!(
            load_model(&path)
                .unwrap()
                .as_ref()
                .map(|selection| selection.model.as_str()),
            Some("last-model")
        );
        save_model(&path, "", None).unwrap();
        assert_eq!(load_model(&path).unwrap(), None);
        fs::write(&path, "invalid json").unwrap();
        assert!(load_model(&path).is_err());
    }

    #[test]
    fn legacy_configs_load_and_selected_provider_survives_reload() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("providers.json");
        fs::write(
            &path,
            r#"[{"base_url":"https://example.com/v1","api_key":"key"}]"#,
        )
        .unwrap();
        let providers = load(&path).unwrap();
        assert!(providers[0].name.is_empty());
        let path = directory.path().join("last-model.json");
        fs::write(&path, r#""legacy-model""#).unwrap();
        assert_eq!(
            load_model(&path).unwrap(),
            Some(SavedModel {
                model: "legacy-model".to_owned(),
                base_url: None,
            })
        );
        save_model(&path, "shared", Some("https://example.com/v1")).unwrap();
        assert_eq!(
            load_model(&path).unwrap(),
            Some(SavedModel {
                model: "shared".to_owned(),
                base_url: Some("https://example.com/v1".to_owned()),
            })
        );
    }

    #[test]
    fn credentials_survive_reload_and_have_private_permissions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("harness/providers.json");
        assert!(load(&path).unwrap().is_empty());
        let mut providers = vec![SavedProvider {
            name: "Test provider".to_owned(),
            base_url: "https://provider.example/v1".to_owned(),
            api_key: "test-key".to_owned(),
        }];
        save(&path, &providers).unwrap();
        assert!(load(&path).unwrap() == providers);
        providers[0].api_key = "updated-key".to_owned();
        save(&path, &providers).unwrap();
        assert!(load(&path).unwrap() == providers);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(path.parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn urls_are_normalized_and_invalid_endpoints_are_rejected() {
        assert_eq!(
            validate_url(" https://provider.example/v1/ ").unwrap(),
            "https://provider.example/v1"
        );
        assert_eq!(
            validate_url("http://localhost:8080/v1").unwrap(),
            "http://localhost:8080/v1"
        );
        for url in [
            "example.com",
            "file:///tmp/provider",
            "https://user:secret@example.com/v1",
            "https://example.com/v1?key=secret",
            "https://example.com/v1#fragment",
            "https://example.com/v1;;https://other.example/v1",
        ] {
            assert!(validate_url(url).is_err());
        }
    }

    #[test]
    fn malformed_saved_credentials_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("providers.json");
        fs::write(&path, "invalid json").unwrap();
        assert!(load(&path).is_err());
        fs::write(
            &path,
            r#"[{"base_url":"https://example.com/v1","api_key":""}]"#,
        )
        .unwrap();
        assert!(load(&path).is_err());
    }
}
