#[must_use]
pub const fn extension_name() -> &'static str {
    "fts5"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentMode {
    Stored,
    Contentless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteAction {
    Reject,
    Tombstone,
    PhysicalPurge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
pub struct Fts5Config {
    secure_delete: bool,
    content_mode: ContentMode,
    contentless_delete: bool,
}

impl Fts5Config {
    #[must_use]
    pub const fn new(content_mode: ContentMode) -> Self {
        Self {
            secure_delete: false,
            content_mode,
            contentless_delete: false,
        }
    }

    #[must_use]
    pub const fn secure_delete_enabled(self) -> bool {
        self.secure_delete
    }

    #[must_use]
    pub const fn contentless_delete_enabled(self) -> bool {
        self.contentless_delete
    }

    #[must_use]
    pub const fn content_mode(self) -> ContentMode {
        self.content_mode
    }

    #[must_use]
    pub const fn delete_action(self) -> DeleteAction {
        match self.content_mode {
            ContentMode::Stored => {
                if self.secure_delete {
                    DeleteAction::PhysicalPurge
                } else {
                    DeleteAction::Tombstone
                }
            }
            ContentMode::Contentless => {
                if !self.contentless_delete {
                    DeleteAction::Reject
                } else if self.secure_delete {
                    DeleteAction::PhysicalPurge
                } else {
                    DeleteAction::Tombstone
                }
            }
        }
    }

    /// Apply SQLite-style FTS5 table control commands like `secure-delete=1`.
    ///
    /// Returns `true` when the command key/value is recognized and applied.
    /// Returns `false` for unknown command keys or invalid values.
    pub fn apply_control_command(&mut self, command: &str) -> bool {
        let trimmed = command.trim();
        let Some((raw_key, raw_value)) = trimmed.split_once('=') else {
            return false;
        };

        let key = raw_key.trim().to_ascii_lowercase();
        let Some(value) = parse_bool_like(raw_value) else {
            return false;
        };

        match key.as_str() {
            "secure-delete" | "secure_delete" => {
                self.secure_delete = value;
                true
            }
            "contentless_delete" => {
                self.contentless_delete = value;
                true
            }
            _ => false,
        }
    }
}

impl Default for Fts5Config {
    fn default() -> Self {
        Self::new(ContentMode::Stored)
    }
}

fn parse_bool_like(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Some(true),
        "0" | "off" | "false" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{ContentMode, DeleteAction, Fts5Config, extension_name};

    #[test]
    fn test_extension_name_matches_crate_suffix() {
        let expected = env!("CARGO_PKG_NAME")
            .strip_prefix("fsqlite-ext-")
            .expect("extension crates should use fsqlite-ext-* naming");
        assert_eq!(extension_name(), expected);
    }

    #[test]
    fn test_secure_delete_enable_command() {
        let mut config = Fts5Config::default();
        assert!(config.apply_control_command("secure-delete=1"));
        assert!(config.secure_delete_enabled());
        assert_eq!(config.delete_action(), DeleteAction::PhysicalPurge);
    }

    #[test]
    fn test_secure_delete_disable_command() {
        let mut config = Fts5Config::default();
        assert!(config.apply_control_command("secure_delete=true"));
        assert!(config.secure_delete_enabled());
        assert!(config.apply_control_command("secure-delete=0"));
        assert!(!config.secure_delete_enabled());
        assert_eq!(config.delete_action(), DeleteAction::Tombstone);
    }

    #[test]
    fn test_invalid_control_command_is_ignored() {
        let mut config = Fts5Config::default();
        assert!(!config.apply_control_command("secure-delete=maybe"));
        assert!(!config.apply_control_command("integrity-check=1"));
        assert_eq!(config.delete_action(), DeleteAction::Tombstone);
    }

    #[test]
    fn test_contentless_delete_rejects_without_toggle() {
        let config = Fts5Config::new(ContentMode::Contentless);
        assert_eq!(config.delete_action(), DeleteAction::Reject);
    }

    #[test]
    fn test_contentless_delete_tombstone_mode() {
        let mut config = Fts5Config::new(ContentMode::Contentless);
        assert!(config.apply_control_command("contentless_delete=1"));
        assert_eq!(config.delete_action(), DeleteAction::Tombstone);
    }

    #[test]
    fn test_contentless_delete_secure_delete_combo() {
        let mut config = Fts5Config::new(ContentMode::Contentless);
        assert!(config.apply_control_command("contentless_delete=1"));
        assert!(config.apply_control_command("secure-delete=on"));
        assert_eq!(config.delete_action(), DeleteAction::PhysicalPurge);
    }
}
