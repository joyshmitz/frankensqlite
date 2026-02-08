//! ECS (Erasure-Coded Stream) substrate types.
//!
//! This module defines foundational identity primitives for Native mode:
//! - [`ObjectId`] / [`PayloadHash`]: content-addressed identity (§3.5.1)
//! - [`SymbolRecord`] / [`SymbolRecordFlags`]: physical storage envelope (§3.5.2)
//!
//! Spec: COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md §3.5.1–§3.5.2.

use std::fmt;

use crate::glossary::{OTI_WIRE_SIZE, Oti};

/// Domain separation prefix for ECS ObjectIds (spec: `"fsqlite:ecs:v1"`).
const ECS_OBJECT_ID_DOMAIN_SEPARATOR: &[u8] = b"fsqlite:ecs:v1";

/// Canonical 32-byte hash of an ECS object's payload.
///
/// The spec refers to this as `payload_hash` in:
/// `ObjectId = Trunc128(BLAKE3("fsqlite:ecs:v1" || canonical_object_header || payload_hash))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(transparent)]
pub struct PayloadHash([u8; 32]);

impl PayloadHash {
    /// Construct from raw bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the hash bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Hash a payload using BLAKE3-256.
    #[must_use]
    pub fn blake3(payload: &[u8]) -> Self {
        let hash = blake3::hash(payload);
        Self(*hash.as_bytes())
    }
}

/// 16-byte truncated content-addressed identity for an ECS object.
///
/// Spec:
/// `ObjectId = Trunc128(BLAKE3("fsqlite:ecs:v1" || canonical_object_header || payload_hash))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(transparent)]
pub struct ObjectId([u8; 16]);

impl ObjectId {
    /// ObjectId length in bytes.
    pub const LEN: usize = 16;

    /// Domain separation prefix from the spec.
    pub const DOMAIN_SEPARATOR: &'static [u8] = ECS_OBJECT_ID_DOMAIN_SEPARATOR;

    /// Construct from raw bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Return the raw bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Derive an ObjectId from already-canonicalized bytes.
    ///
    /// `canonical_bytes` must be a deterministic, versioned wire-format blob
    /// (spec: "not serde vibes") representing the object's header plus its
    /// `payload_hash`.
    #[must_use]
    pub fn derive_from_canonical_bytes(canonical_bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(Self::DOMAIN_SEPARATOR);
        hasher.update(canonical_bytes);
        let digest = hasher.finalize();

        let mut out = [0u8; Self::LEN];
        out.copy_from_slice(&digest.as_bytes()[..Self::LEN]);
        Self(out)
    }

    /// Derive an ObjectId from canonical header bytes and a payload hash.
    #[must_use]
    pub fn derive(canonical_object_header: &[u8], payload_hash: PayloadHash) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(Self::DOMAIN_SEPARATOR);
        hasher.update(canonical_object_header);
        hasher.update(payload_hash.as_bytes());
        let digest = hasher.finalize();

        let mut out = [0u8; Self::LEN];
        out.copy_from_slice(&digest.as_bytes()[..Self::LEN]);
        Self(out)
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl AsRef<[u8]> for ObjectId {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<[u8; 16]> for ObjectId {
    fn from(value: [u8; 16]) -> Self {
        Self(value)
    }
}

// ---------------------------------------------------------------------------
// §3.5.2 SymbolRecord Envelope and Auth Tags
// ---------------------------------------------------------------------------

/// Magic bytes identifying a SymbolRecord: `"FSEC"` (0x46 0x53 0x45 0x43).
pub const SYMBOL_RECORD_MAGIC: [u8; 4] = [0x46, 0x53, 0x45, 0x43];

/// Current envelope version.
pub const SYMBOL_RECORD_VERSION: u8 = 1;

/// Domain separation prefix for symbol auth tags.
const SYMBOL_AUTH_DOMAIN: &[u8] = b"fsqlite:symbol-auth:v1";

bitflags::bitflags! {
    /// Flags for a [`SymbolRecord`].
    ///
    /// Additional local flags MAY be defined but MUST be treated as advisory
    /// optimization hints. Correctness never depends on them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SymbolRecordFlags: u8 {
        /// This record is the first source symbol (esi = 0) and the writer
        /// attempted to place the entire systematic run contiguously.
        const SYSTEMATIC_RUN_START = 0x01;
    }
}

/// Validation error when deserializing or checking a [`SymbolRecord`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolRecordError {
    /// Input too short to contain a complete record.
    TooShort { expected_min: usize, actual: usize },
    /// Magic bytes do not match `"FSEC"`.
    BadMagic([u8; 4]),
    /// Envelope version is unsupported.
    UnsupportedVersion(u8),
    /// `symbol_size != OTI.T` — key invariant violated.
    SymbolSizeMismatch { symbol_size: u32, oti_t: u32 },
    /// `frame_xxh3` integrity check failed.
    IntegrityFailure { expected: u64, computed: u64 },
    /// Auth tag verification failed.
    AuthTagFailure,
}

impl fmt::Display for SymbolRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort {
                expected_min,
                actual,
            } => {
                write!(
                    f,
                    "symbol record too short: need {expected_min}, got {actual}"
                )
            }
            Self::BadMagic(m) => write!(f, "bad magic: {m:02x?}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::SymbolSizeMismatch { symbol_size, oti_t } => {
                write!(f, "symbol_size ({symbol_size}) != OTI.T ({oti_t})")
            }
            Self::IntegrityFailure { expected, computed } => {
                write!(
                    f,
                    "frame_xxh3 mismatch: stored {expected:#018x}, computed {computed:#018x}"
                )
            }
            Self::AuthTagFailure => write!(f, "auth tag verification failed"),
        }
    }
}

/// Fixed header size before `symbol_data`:
/// magic(4) + version(1) + object_id(16) + OTI(22) + esi(4) + symbol_size(4) = 51.
const HEADER_BEFORE_DATA: usize = 4 + 1 + 16 + OTI_WIRE_SIZE + 4 + 4;

/// Fixed trailer size after `symbol_data`:
/// flags(1) + frame_xxh3(8) + auth_tag(16) = 25.
const TRAILER_AFTER_DATA: usize = 1 + 8 + 16;

/// The atomic unit of physical storage for ECS objects (§3.5.2).
///
/// A `SymbolRecord` is self-describing: a decoder collecting K' symbols with
/// the same `ObjectId` can reconstruct the original object without any
/// external metadata.
///
/// Wire layout (all integers little-endian):
/// ```text
/// magic[4] | version[1] | object_id[16] | OTI[22] | esi[4] | symbol_size[4]
/// | symbol_data[T] | flags[1] | frame_xxh3[8] | auth_tag[16]
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolRecord {
    /// Content-addressed identity of the parent ECS object.
    pub object_id: ObjectId,
    /// RaptorQ Object Transmission Information.
    pub oti: Oti,
    /// Encoding Symbol Identifier — which symbol this is.
    pub esi: u32,
    /// The actual RaptorQ encoding symbol payload.
    pub symbol_data: Vec<u8>,
    /// Advisory flags.
    pub flags: SymbolRecordFlags,
    /// xxhash3 of all preceding fields for fast integrity checking.
    pub frame_xxh3: u64,
    /// Optional BLAKE3-keyed auth tag for authenticated transport.
    /// All-zero when `symbol_auth = off`.
    pub auth_tag: [u8; 16],
}

impl SymbolRecord {
    /// Compute the `frame_xxh3` digest over the header + symbol_data + flags.
    ///
    /// This covers everything from `magic` through `flags`, i.e. all fields
    /// preceding `frame_xxh3` in the wire layout.
    #[must_use]
    fn compute_frame_xxh3(pre_hash_bytes: &[u8]) -> u64 {
        xxhash_rust::xxh3::xxh3_64(pre_hash_bytes)
    }

    /// Build the byte region that `frame_xxh3` covers (magic..flags inclusive).
    fn pre_hash_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            HEADER_BEFORE_DATA + self.symbol_data.len() + 1, /* flags */
        );
        buf.extend_from_slice(&SYMBOL_RECORD_MAGIC);
        buf.push(SYMBOL_RECORD_VERSION);
        buf.extend_from_slice(self.object_id.as_bytes());
        buf.extend_from_slice(&self.oti.to_bytes());
        buf.extend_from_slice(&self.esi.to_le_bytes());
        #[allow(clippy::cast_possible_truncation)]
        let symbol_size = self.symbol_data.len() as u32;
        buf.extend_from_slice(&symbol_size.to_le_bytes());
        buf.extend_from_slice(&self.symbol_data);
        buf.push(self.flags.bits());
        buf
    }

    /// Create a new `SymbolRecord`, computing `frame_xxh3` automatically.
    ///
    /// `auth_tag` is set to all-zero (symbol_auth off). Use
    /// [`Self::with_auth_tag`] to set an authenticated tag.
    #[must_use]
    pub fn new(
        object_id: ObjectId,
        oti: Oti,
        esi: u32,
        symbol_data: Vec<u8>,
        flags: SymbolRecordFlags,
    ) -> Self {
        let mut rec = Self {
            object_id,
            oti,
            esi,
            symbol_data,
            flags,
            frame_xxh3: 0,
            auth_tag: [0u8; 16],
        };
        let pre_hash = rec.pre_hash_bytes();
        rec.frame_xxh3 = Self::compute_frame_xxh3(&pre_hash);
        rec
    }

    /// Set the auth tag using a BLAKE3-keyed MAC.
    ///
    /// `epoch_key` is the 32-byte key derived from `SymbolSegmentHeader.epoch_id`
    /// per §4.18.2.
    ///
    /// ```text
    /// auth_tag = Trunc128(BLAKE3_KEYED(epoch_key,
    ///     "fsqlite:symbol-auth:v1" || bytes(magic..frame_xxh3)))
    /// ```
    #[must_use]
    pub fn with_auth_tag(mut self, epoch_key: &[u8; 32]) -> Self {
        self.auth_tag = Self::compute_auth_tag(epoch_key, &self.pre_hash_bytes(), self.frame_xxh3);
        self
    }

    /// Compute the 16-byte auth tag.
    fn compute_auth_tag(epoch_key: &[u8; 32], pre_hash: &[u8], frame_xxh3: u64) -> [u8; 16] {
        let mut keyed_hasher = blake3::Hasher::new_keyed(epoch_key);
        keyed_hasher.update(SYMBOL_AUTH_DOMAIN);
        keyed_hasher.update(pre_hash);
        keyed_hasher.update(&frame_xxh3.to_le_bytes());
        let digest = keyed_hasher.finalize();
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&digest.as_bytes()[..16]);
        tag
    }

    /// Serialize to canonical wire bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let total = HEADER_BEFORE_DATA + self.symbol_data.len() + TRAILER_AFTER_DATA;
        let mut buf = Vec::with_capacity(total);

        // Header
        buf.extend_from_slice(&SYMBOL_RECORD_MAGIC);
        buf.push(SYMBOL_RECORD_VERSION);
        buf.extend_from_slice(self.object_id.as_bytes());
        buf.extend_from_slice(&self.oti.to_bytes());
        buf.extend_from_slice(&self.esi.to_le_bytes());
        #[allow(clippy::cast_possible_truncation)]
        let symbol_size = self.symbol_data.len() as u32;
        buf.extend_from_slice(&symbol_size.to_le_bytes());

        // Payload
        buf.extend_from_slice(&self.symbol_data);

        // Trailer
        buf.push(self.flags.bits());
        buf.extend_from_slice(&self.frame_xxh3.to_le_bytes());
        buf.extend_from_slice(&self.auth_tag);

        debug_assert_eq!(buf.len(), total);
        buf
    }

    /// Deserialize from canonical wire bytes, validating all invariants.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolRecordError`] if the data is malformed, the magic is
    /// wrong, the version is unsupported, `symbol_size != OTI.T`, or the
    /// `frame_xxh3` integrity check fails.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SymbolRecordError> {
        // Need at least the fixed header to read symbol_size.
        if data.len() < HEADER_BEFORE_DATA {
            return Err(SymbolRecordError::TooShort {
                expected_min: HEADER_BEFORE_DATA,
                actual: data.len(),
            });
        }

        // Magic
        let magic: [u8; 4] = data[0..4].try_into().expect("4 bytes");
        if magic != SYMBOL_RECORD_MAGIC {
            return Err(SymbolRecordError::BadMagic(magic));
        }

        // Version
        let version = data[4];
        if version != SYMBOL_RECORD_VERSION {
            return Err(SymbolRecordError::UnsupportedVersion(version));
        }

        // ObjectId
        let object_id = ObjectId::from_bytes(data[5..21].try_into().expect("16 bytes"));

        // OTI
        let oti =
            Oti::from_bytes(&data[21..43]).expect("already checked length >= HEADER_BEFORE_DATA");

        // ESI + symbol_size
        let esi = u32::from_le_bytes(data[43..47].try_into().expect("4 bytes"));
        let symbol_size = u32::from_le_bytes(data[47..51].try_into().expect("4 bytes"));

        // Key invariant: symbol_size == OTI.T
        if symbol_size != oti.t {
            return Err(SymbolRecordError::SymbolSizeMismatch {
                symbol_size,
                oti_t: oti.t,
            });
        }

        let total_size = HEADER_BEFORE_DATA + symbol_size as usize + TRAILER_AFTER_DATA;
        if data.len() < total_size {
            return Err(SymbolRecordError::TooShort {
                expected_min: total_size,
                actual: data.len(),
            });
        }

        // Symbol data
        let data_start = HEADER_BEFORE_DATA;
        let data_end = data_start + symbol_size as usize;
        let symbol_data = data[data_start..data_end].to_vec();

        // Trailer
        let flags = SymbolRecordFlags::from_bits_truncate(data[data_end]);
        let frame_xxh3 = u64::from_le_bytes(
            data[data_end + 1..data_end + 9]
                .try_into()
                .expect("8 bytes"),
        );
        let auth_tag: [u8; 16] = data[data_end + 9..data_end + 25]
            .try_into()
            .expect("16 bytes");

        // Verify frame_xxh3 integrity
        let pre_hash_end = data_end + 1; // magic..flags inclusive
        let computed = Self::compute_frame_xxh3(&data[..pre_hash_end]);
        if computed != frame_xxh3 {
            return Err(SymbolRecordError::IntegrityFailure {
                expected: frame_xxh3,
                computed,
            });
        }

        Ok(Self {
            object_id,
            oti,
            esi,
            symbol_data,
            flags,
            frame_xxh3,
            auth_tag,
        })
    }

    /// Verify `frame_xxh3` integrity without full deserialization.
    #[must_use]
    pub fn verify_integrity(&self) -> bool {
        let pre_hash = self.pre_hash_bytes();
        Self::compute_frame_xxh3(&pre_hash) == self.frame_xxh3
    }

    /// Verify the auth tag using the given epoch key.
    ///
    /// Returns `true` if the auth tag matches, or if the tag is all-zero
    /// (symbol_auth off — tag is ignored per spec).
    #[must_use]
    pub fn verify_auth(&self, epoch_key: &[u8; 32]) -> bool {
        if self.auth_tag == [0u8; 16] {
            return true; // auth off: tag ignored
        }
        let expected = Self::compute_auth_tag(epoch_key, &self.pre_hash_bytes(), self.frame_xxh3);
        self.auth_tag == expected
    }

    /// Total serialized size of this record in bytes.
    #[must_use]
    pub fn wire_size(&self) -> usize {
        HEADER_BEFORE_DATA + self.symbol_data.len() + TRAILER_AFTER_DATA
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_id_blake3_derivation() {
        let header = b"hdr:v1\x00";
        let payload = b"hello world";
        let payload_hash = PayloadHash::blake3(payload);

        let derived = ObjectId::derive(header, payload_hash);

        let mut canonical = Vec::new();
        canonical.extend_from_slice(header);
        canonical.extend_from_slice(payload_hash.as_bytes());
        let derived2 = ObjectId::derive_from_canonical_bytes(&canonical);

        assert_eq!(derived, derived2);

        let mut hasher = blake3::Hasher::new();
        hasher.update(ObjectId::DOMAIN_SEPARATOR);
        hasher.update(&canonical);
        let digest = hasher.finalize();
        let mut expected = [0u8; 16];
        expected.copy_from_slice(&digest.as_bytes()[..16]);

        assert_eq!(derived.as_bytes(), &expected);
    }

    #[test]
    fn test_object_id_collision_resistance() {
        let header = b"hdr:v1\x00";
        let payload_a = b"payload-a";
        let payload_b = b"payload-b";
        let id_a = ObjectId::derive(header, PayloadHash::blake3(payload_a));
        let id_b = ObjectId::derive(header, PayloadHash::blake3(payload_b));
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn test_object_id_deterministic() {
        let header = b"hdr:v1\x00";
        let payload = b"payload";
        let hash = PayloadHash::blake3(payload);
        let id1 = ObjectId::derive(header, hash);
        let id2 = ObjectId::derive(header, hash);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_object_id_display_hex() {
        let id = ObjectId::from_bytes([0u8; 16]);
        let s = id.to_string();
        assert_eq!(s.len(), 32);
        assert!(s.chars().all(|ch| matches!(ch, '0'..='9' | 'a'..='f')));

        // A stable known-value check (16 zero bytes => 32 zero hex chars).
        assert_eq!(s, "00000000000000000000000000000000");
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn test_oti(symbol_size: u32) -> Oti {
        Oti {
            f: 16384,
            al: 4,
            t: symbol_size,
            z: 1,
            n: 1,
        }
    }

    fn test_record(symbol_size: u32) -> SymbolRecord {
        let data = vec![0xAB; symbol_size as usize];
        let oid = ObjectId::from_bytes([1u8; 16]);
        SymbolRecord::new(
            oid,
            test_oti(symbol_size),
            0,
            data,
            SymbolRecordFlags::empty(),
        )
    }

    // -----------------------------------------------------------------------
    // §3.5.2 SymbolRecord tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbol_record_serialize_deserialize() {
        let rec = test_record(4096);
        let bytes = rec.to_bytes();
        let rec2 = SymbolRecord::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(rec, rec2);
    }

    #[test]
    fn test_symbol_record_magic_validation() {
        let rec = test_record(64);
        let mut bytes = rec.to_bytes();
        bytes[0] = 0xFF;
        let err = SymbolRecord::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, SymbolRecordError::BadMagic(_)));
    }

    #[test]
    fn test_symbol_record_frame_xxh3_integrity() {
        let rec = test_record(128);
        let mut bytes = rec.to_bytes();
        // Flip one bit in symbol_data
        bytes[HEADER_BEFORE_DATA] ^= 0x01;
        let err = SymbolRecord::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, SymbolRecordError::IntegrityFailure { .. }));
    }

    #[test]
    fn test_symbol_record_invariant_symbol_size_eq_oti_t() {
        let oid = ObjectId::from_bytes([2u8; 16]);
        let oti = test_oti(100);
        // Manually build wire bytes with symbol_size=200 but OTI.T=100
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&SYMBOL_RECORD_MAGIC);
        bytes.push(SYMBOL_RECORD_VERSION);
        bytes.extend_from_slice(oid.as_bytes());
        bytes.extend_from_slice(&oti.to_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // esi
        bytes.extend_from_slice(&200u32.to_le_bytes()); // symbol_size != oti.t
        bytes.extend_from_slice(&[0u8; 200]);
        bytes.push(0); // flags
        let hash = xxhash_rust::xxh3::xxh3_64(&bytes);
        bytes.extend_from_slice(&hash.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 16]);

        let err = SymbolRecord::from_bytes(&bytes).unwrap_err();
        assert!(matches!(
            err,
            SymbolRecordError::SymbolSizeMismatch {
                symbol_size: 200,
                oti_t: 100
            }
        ));
    }

    #[test]
    fn test_symbol_record_auth_tag_verification() {
        let epoch_key = [0x42u8; 32];
        let rec = test_record(64).with_auth_tag(&epoch_key);
        assert_ne!(rec.auth_tag, [0u8; 16]);
        assert!(rec.verify_auth(&epoch_key));

        // Tamper: change one data byte, recompute frame_xxh3 but NOT auth_tag
        let mut tampered = rec;
        tampered.symbol_data[0] ^= 0x01;
        let pre_hash = tampered.pre_hash_bytes();
        tampered.frame_xxh3 = xxhash_rust::xxh3::xxh3_64(&pre_hash);
        assert!(!tampered.verify_auth(&epoch_key));
    }

    #[test]
    fn test_symbol_record_auth_tag_ignored_when_off() {
        let rec = test_record(64);
        assert_eq!(rec.auth_tag, [0u8; 16]);
        let any_key = [0xFFu8; 32];
        assert!(rec.verify_auth(&any_key));
    }

    #[test]
    fn test_symbol_record_systematic_flag() {
        let oid = ObjectId::from_bytes([3u8; 16]);
        let rec = SymbolRecord::new(
            oid,
            test_oti(64),
            0,
            vec![0u8; 64],
            SymbolRecordFlags::SYSTEMATIC_RUN_START,
        );
        assert!(rec.flags.contains(SymbolRecordFlags::SYSTEMATIC_RUN_START));
        assert_eq!(rec.esi, 0);

        let bytes = rec.to_bytes();
        let rec2 = SymbolRecord::from_bytes(&bytes).unwrap();
        assert!(rec2.flags.contains(SymbolRecordFlags::SYSTEMATIC_RUN_START));
    }

    #[test]
    fn test_oti_field_widths() {
        let oti = Oti {
            f: 1_000_000,
            al: 4,
            t: 65536,
            z: 10,
            n: 1,
        };
        let bytes = oti.to_bytes();
        let oti2 = Oti::from_bytes(&bytes).unwrap();
        assert_eq!(oti, oti2);
        assert_eq!(oti2.t, 65536);
    }

    #[test]
    fn test_systematic_fast_path_happy() {
        let oid = ObjectId::from_bytes([4u8; 16]);
        let oti = Oti {
            f: 256,
            al: 4,
            t: 64,
            z: 1,
            n: 1,
        };

        let records: Vec<_> = (0u32..4)
            .map(|i| {
                let flags = if i == 0 {
                    SymbolRecordFlags::SYSTEMATIC_RUN_START
                } else {
                    SymbolRecordFlags::empty()
                };
                let fill = u8::try_from(i).expect("i < 4");
                SymbolRecord::new(oid, oti, i, vec![fill; 64], flags)
            })
            .collect();

        assert!(
            records[0]
                .flags
                .contains(SymbolRecordFlags::SYSTEMATIC_RUN_START)
        );
        for rec in &records[1..] {
            assert!(!rec.flags.contains(SymbolRecordFlags::SYSTEMATIC_RUN_START));
        }

        // Reconstruct via systematic fast path
        let mut reconstructed = Vec::new();
        for rec in &records {
            assert!(rec.verify_integrity());
            reconstructed.extend_from_slice(&rec.symbol_data);
        }
        let f = usize::try_from(oti.f).expect("OTI transfer length fits in usize");
        reconstructed.truncate(f);
        assert_eq!(reconstructed.len(), 256);

        for (i, chunk) in reconstructed.chunks(64).enumerate() {
            let expected = u8::try_from(i).expect("i < 4");
            assert!(chunk.iter().all(|&b| b == expected));
        }
    }

    #[test]
    fn test_systematic_fast_path_fallback() {
        let oid = ObjectId::from_bytes([5u8; 16]);
        let oti = Oti {
            f: 256,
            al: 4,
            t: 64,
            z: 1,
            n: 1,
        };

        let rec2 = SymbolRecord::new(oid, oti, 2, vec![2u8; 64], SymbolRecordFlags::empty());
        let mut bytes = rec2.to_bytes();
        bytes[HEADER_BEFORE_DATA] ^= 0xFF; // corrupt data

        let result = SymbolRecord::from_bytes(&bytes);
        assert!(matches!(
            result.unwrap_err(),
            SymbolRecordError::IntegrityFailure { .. }
        ));
    }

    #[test]
    fn test_symbol_record_version_validation() {
        let rec = test_record(64);
        let mut bytes = rec.to_bytes();
        bytes[4] = 99;
        let err = SymbolRecord::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, SymbolRecordError::UnsupportedVersion(99)));
    }

    #[test]
    fn test_symbol_record_too_short() {
        let err = SymbolRecord::from_bytes(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, SymbolRecordError::TooShort { .. }));
    }

    #[test]
    fn test_symbol_record_wire_size() {
        let rec = test_record(4096);
        assert_eq!(
            rec.wire_size(),
            HEADER_BEFORE_DATA + 4096 + TRAILER_AFTER_DATA
        );
        assert_eq!(rec.wire_size(), rec.to_bytes().len());
    }

    #[test]
    fn test_symbol_record_verify_integrity() {
        let rec = test_record(128);
        assert!(rec.verify_integrity());

        let mut bad = rec;
        bad.symbol_data[0] ^= 0x01;
        assert!(!bad.verify_integrity());
    }

    #[test]
    fn test_oti_roundtrip() {
        let oti = Oti {
            f: u64::MAX,
            al: u16::MAX,
            t: u32::MAX,
            z: u32::MAX,
            n: u32::MAX,
        };
        let bytes = oti.to_bytes();
        assert_eq!(bytes.len(), OTI_WIRE_SIZE);
        let oti2 = Oti::from_bytes(&bytes).unwrap();
        assert_eq!(oti, oti2);
    }

    #[test]
    fn test_oti_from_bytes_too_short() {
        assert!(Oti::from_bytes(&[0u8; 10]).is_none());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_oti() -> impl Strategy<Value = Oti> {
        (
            any::<u64>(),
            any::<u16>(),
            1..=65536u32,
            1..=100u32,
            1..=100u32,
        )
            .prop_map(|(f, al, t, z, n)| Oti { f, al, t, z, n })
    }

    proptest! {
        #[test]
        fn prop_symbol_record_roundtrip(
            oti in arb_oti(),
            esi in any::<u32>(),
            data_byte in any::<u8>(),
        ) {
            let oid = ObjectId::from_bytes([7u8; 16]);
            let data = vec![data_byte; oti.t as usize];
            let rec = SymbolRecord::new(oid, oti, esi, data, SymbolRecordFlags::empty());
            let bytes = rec.to_bytes();
            let rec2 = SymbolRecord::from_bytes(&bytes).unwrap();
            prop_assert_eq!(rec, rec2);
        }

        #[test]
        fn prop_frame_xxh3_collision_resistance(
            a in proptest::collection::vec(any::<u8>(), 64..=64),
            b in proptest::collection::vec(any::<u8>(), 64..=64),
        ) {
            if a != b {
                let oid = ObjectId::from_bytes([8u8; 16]);
                let oti = Oti { f: 64, al: 4, t: 64, z: 1, n: 1 };
                let rec_a = SymbolRecord::new(oid, oti, 0, a, SymbolRecordFlags::empty());
                let rec_b = SymbolRecord::new(oid, oti, 0, b, SymbolRecordFlags::empty());
                prop_assert_ne!(rec_a.frame_xxh3, rec_b.frame_xxh3);
            }
        }
    }
}
