//! FrankenSQLite verification harness.
//!
//! This crate is intentionally not "just tests": it contains reusable
//! verification tooling (trace exporters, schedule exploration harnesses, etc.)
//! that other crates can call into from their own tests.

pub mod tla;
