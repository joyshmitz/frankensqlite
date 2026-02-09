#[must_use]
pub const fn extension_name() -> &'static str {
    "session"
}

#[cfg(test)]
mod tests {
    use super::extension_name;

    #[test]
    fn test_extension_name_matches_crate_suffix() {
        let expected = env!("CARGO_PKG_NAME")
            .strip_prefix("fsqlite-ext-")
            .expect("extension crates should use fsqlite-ext-* naming");
        assert_eq!(extension_name(), expected);
    }
}
