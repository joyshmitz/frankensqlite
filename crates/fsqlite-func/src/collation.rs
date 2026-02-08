//! Collation callback trait and built-in collations (§9.4).
//!
//! Collations are pure comparators used by ORDER BY, GROUP BY, DISTINCT,
//! and index traversal. They are open extension points.
//!
//! `compare` is intentionally CPU-only and does not accept `&Cx`.
//!
//! # Contract
//!
//! Implementations **must** be:
//! - **Deterministic**: same inputs always produce the same output.
//! - **Antisymmetric**: `compare(a, b)` is the reverse of `compare(b, a)`.
//! - **Transitive**: if `a < b` and `b < c`, then `a < c`.
#![allow(clippy::unnecessary_literal_bound)]

use std::cmp::Ordering;

/// A collation comparator.
///
/// Implementations define total ordering over UTF-8 byte strings.
///
/// Built-in collations: [`BinaryCollation`] (memcmp), [`NoCaseCollation`]
/// (ASCII case-insensitive), [`RtrimCollation`] (trailing-space-insensitive).
pub trait CollationFunction: Send + Sync {
    /// Collation name (for `COLLATE name`).
    fn name(&self) -> &str;

    /// Compare two UTF-8 byte slices.
    ///
    /// Must be deterministic, antisymmetric, and transitive.
    fn compare(&self, left: &[u8], right: &[u8]) -> Ordering;
}

// ── Built-in collations ──────────────────────────────────────────────────

/// BINARY collation: raw `memcmp` byte comparison.
///
/// This is SQLite's default collation. Comparison is byte-by-byte with no
/// locale or case folding.
pub struct BinaryCollation;

impl CollationFunction for BinaryCollation {
    fn name(&self) -> &str {
        "BINARY"
    }

    fn compare(&self, left: &[u8], right: &[u8]) -> Ordering {
        left.cmp(right)
    }
}

/// NOCASE collation: ASCII case-insensitive comparison.
///
/// Only folds ASCII letters (`A-Z` → `a-z`). Non-ASCII bytes are compared
/// as-is. For full Unicode case folding, use the ICU extension (§14.6).
pub struct NoCaseCollation;

impl CollationFunction for NoCaseCollation {
    fn name(&self) -> &str {
        "NOCASE"
    }

    fn compare(&self, left: &[u8], right: &[u8]) -> Ordering {
        let l = left.iter().map(u8::to_ascii_lowercase);
        let r = right.iter().map(u8::to_ascii_lowercase);
        l.cmp(r)
    }
}

/// RTRIM collation: trailing-space-insensitive comparison.
///
/// Trailing ASCII spaces (`0x20`) are stripped before comparison.
/// All other characters (including tabs, non-breaking spaces) are significant.
pub struct RtrimCollation;

impl CollationFunction for RtrimCollation {
    fn name(&self) -> &str {
        "RTRIM"
    }

    fn compare(&self, left: &[u8], right: &[u8]) -> Ordering {
        let l = strip_trailing_spaces(left);
        let r = strip_trailing_spaces(right);
        l.cmp(r)
    }
}

fn strip_trailing_spaces(s: &[u8]) -> &[u8] {
    let mut end = s.len();
    while end > 0 && s[end - 1] == b' ' {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collation_binary_memcmp() {
        let coll = BinaryCollation;
        assert_eq!(coll.compare(b"abc", b"abc"), Ordering::Equal);
        assert_eq!(coll.compare(b"abc", b"abd"), Ordering::Less);
        assert_eq!(coll.compare(b"abd", b"abc"), Ordering::Greater);
        // Mixed case: uppercase < lowercase in byte ordering
        assert_eq!(coll.compare(b"ABC", b"abc"), Ordering::Less);
        // Non-ASCII UTF-8: multibyte sequences
        assert_eq!(
            coll.compare("café".as_bytes(), "café".as_bytes()),
            Ordering::Equal
        );
        assert_ne!(coll.compare("über".as_bytes(), b"uber"), Ordering::Equal);
    }

    #[test]
    fn test_collation_nocase_ascii() {
        let coll = NoCaseCollation;
        assert_eq!(coll.compare(b"ABC", b"abc"), Ordering::Equal);
        assert_eq!(coll.compare(b"Alice", b"alice"), Ordering::Equal);
        // A (0x41) < b (0x62) normally, but NOCASE: a (0x61) < b (0x62)
        assert_eq!(coll.compare(b"A", b"b"), Ordering::Less);
    }

    #[test]
    fn test_collation_rtrim() {
        let coll = RtrimCollation;
        // Trailing spaces are ignored
        assert_eq!(coll.compare(b"hello   ", b"hello"), Ordering::Equal);
        assert_eq!(coll.compare(b"hello", b"hello   "), Ordering::Equal);
        assert_eq!(coll.compare(b"hello   ", b"hello   "), Ordering::Equal);
        // Non-space trailing chars are NOT ignored
        assert_ne!(coll.compare(b"hello!", b"hello"), Ordering::Equal);
        // Trailing space + different content
        assert_ne!(coll.compare(b"hello ", b"hello!"), Ordering::Equal);
    }

    #[test]
    fn test_collation_properties_antisymmetric() {
        let collations: Vec<Box<dyn CollationFunction>> = vec![
            Box::new(BinaryCollation),
            Box::new(NoCaseCollation),
            Box::new(RtrimCollation),
        ];

        let pairs: &[(&[u8], &[u8])] = &[
            (b"abc", b"def"),
            (b"hello", b"world"),
            (b"ABC", b"abc"),
            (b"hello   ", b"hello"),
        ];

        for coll in &collations {
            for &(a, b) in pairs {
                let forward = coll.compare(a, b);
                let reverse = coll.compare(b, a);
                assert_eq!(
                    forward,
                    reverse.reverse(),
                    "{}: compare({:?}, {:?}) = {forward:?}, but reverse = {reverse:?}",
                    coll.name(),
                    std::str::from_utf8(a).unwrap_or("?"),
                    std::str::from_utf8(b).unwrap_or("?"),
                );
            }
        }
    }

    #[test]
    fn test_collation_properties_transitive() {
        let coll = BinaryCollation;
        let a = b"apple";
        let b = b"banana";
        let c = b"cherry";

        // a < b and b < c => a < c
        assert_eq!(coll.compare(a, b), Ordering::Less);
        assert_eq!(coll.compare(b, c), Ordering::Less);
        assert_eq!(coll.compare(a, c), Ordering::Less);
    }

    #[test]
    fn test_collation_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BinaryCollation>();
        assert_send_sync::<NoCaseCollation>();
        assert_send_sync::<RtrimCollation>();
    }
}
