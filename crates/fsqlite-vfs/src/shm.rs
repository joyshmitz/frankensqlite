//! Safe shared-memory region handle for WAL index coordination.
//!
//! This replaces raw `*mut u8` pointers in the [`VfsFile`] SHM API with a safe,
//! bounds-checked wrapper.
//!
//! Note: this type is intentionally backend-agnostic. Concrete VFS backends can
//! construct `ShmRegion` from their own backing storage (in-process heap buffers
//! for `MemoryVfs`, mmap-backed regions for `UnixVfs`, etc.).
//!
//! [`VfsFile`]: crate::traits::VfsFile

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};

/// A handle to a mapped shared-memory region.
///
/// Provides safe, bounds-checked byte-level access to SHM regions used for
/// WAL index coordination. No raw pointers in the public API.
///
/// # Region semantics
///
/// Each region is a fixed-size chunk (typically 32 KB) of the SHM file.
/// Regions are 0-indexed and grow on demand when `VfsFile::shm_map` is
/// called with `extend = true`.
#[derive(Debug, Clone)]
pub struct ShmRegion {
    len: usize,
    data: Arc<Mutex<Vec<u8>>>,
}

impl ShmRegion {
    /// Create a new zeroed SHM region of the given size.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            len: size,
            data: Arc::new(Mutex::new(vec![0; size])),
        }
    }

    /// Create a region from existing data.
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            len,
            data: Arc::new(Mutex::new(data)),
        }
    }

    /// The size of this region in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this region is empty (zero-length).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Acquire a lock and borrow the region as a byte slice.
    ///
    /// The returned guard derefs to `&[u8]` / `&mut [u8]` and releases the lock
    /// on drop.
    #[must_use]
    pub fn lock(&self) -> ShmRegionGuard<'_> {
        ShmRegionGuard {
            guard: self.data.lock().expect("shm region mutex poisoned"),
        }
    }

    /// Read a little-endian `u32` at the given byte offset.
    ///
    /// # Panics
    ///
    /// Panics if `offset + 4 > self.len()`.
    #[must_use]
    pub fn read_u32_le(&self, offset: usize) -> u32 {
        let bytes: [u8; 4] = {
            let guard = self.lock();
            guard[offset..offset + 4]
                .try_into()
                .expect("slice is exactly 4 bytes")
        };
        u32::from_le_bytes(bytes)
    }

    /// Write a little-endian `u32` at the given byte offset.
    ///
    /// # Panics
    ///
    /// Panics if `offset + 4 > self.len()`.
    pub fn write_u32_le(&self, offset: usize, val: u32) {
        let mut guard = self.lock();
        guard[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }

    /// Read a little-endian `u64` at the given byte offset.
    ///
    /// # Panics
    ///
    /// Panics if `offset + 8 > self.len()`.
    #[must_use]
    pub fn read_u64_le(&self, offset: usize) -> u64 {
        let bytes: [u8; 8] = {
            let guard = self.lock();
            guard[offset..offset + 8]
                .try_into()
                .expect("slice is exactly 8 bytes")
        };
        u64::from_le_bytes(bytes)
    }

    /// Write a little-endian `u64` at the given byte offset.
    ///
    /// # Panics
    ///
    /// Panics if `offset + 8 > self.len()`.
    pub fn write_u64_le(&self, offset: usize, val: u64) {
        let mut guard = self.lock();
        guard[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    }
}

/// Locked SHM region access guard.
pub struct ShmRegionGuard<'a> {
    guard: MutexGuard<'a, Vec<u8>>,
}

impl Deref for ShmRegionGuard<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.guard.as_slice()
    }
}

impl DerefMut for ShmRegionGuard<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.guard.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_region_new_zeroed() {
        let region = ShmRegion::new(4096);
        assert_eq!(region.len(), 4096);
        assert!(region.lock().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_shm_region_read_write_u32() {
        let region = ShmRegion::new(64);
        region.write_u32_le(0, 0xDEAD_BEEF);
        region.write_u32_le(4, 42);
        assert_eq!(region.read_u32_le(0), 0xDEAD_BEEF);
        assert_eq!(region.read_u32_le(4), 42);
    }

    #[test]
    fn test_shm_region_read_write_u64() {
        let region = ShmRegion::new(64);
        region.write_u64_le(0, 0x0102_0304_0506_0708);
        assert_eq!(region.read_u64_le(0), 0x0102_0304_0506_0708);
    }

    #[test]
    fn test_shm_region_deref() {
        let region = ShmRegion::new(8);
        {
            let mut g = region.lock();
            g[0] = 0xFF;
        }
        assert_eq!(region.lock()[0], 0xFF);
    }

    #[test]
    fn test_shm_region_from_vec() {
        let data = vec![1, 2, 3, 4];
        let region = ShmRegion::from_vec(data);
        assert_eq!(region.len(), 4);
        assert_eq!(&*region.lock(), &[1, 2, 3, 4]);
    }
}
