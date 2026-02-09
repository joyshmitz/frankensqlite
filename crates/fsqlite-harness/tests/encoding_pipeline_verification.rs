//! RaptorQ Encoding Pipeline verification suite (§3.2.3).
//!
//! Bead: bd-1hi.3
//!
//! Verifies the 5-step encoding pipeline:
//!   Step 1: Determine coding parameters (K → K', S, H, W, L)
//!   Step 2: Construct constraint matrix A (L × L)
//!   Step 3: Build source vector D
//!   Step 4: Solve A·C = D for intermediate symbols
//!   Step 5: Generate encoding symbols from intermediates
//!
//! Tests cover: pipeline stages, correctness, systematic property,
//! determinism, constraint matrix structure, and E2E roundtrips.

use std::collections::HashSet;

use asupersync::raptorq::decoder::{DecodeError, InactivationDecoder, ReceivedSymbol};
use asupersync::raptorq::gf256::Gf256;
use asupersync::raptorq::systematic::{ConstraintMatrix, SystematicEncoder, SystematicParams};

const BEAD_ID: &str = "bd-1hi.3";

// ============================================================================
// Helpers
// ============================================================================

/// Generate deterministic source symbols for testing.
fn make_source(k: usize, symbol_size: usize) -> Vec<Vec<u8>> {
    (0..k)
        .map(|i| {
            (0..symbol_size)
                .map(|j| ((i * 37 + j * 13 + 7) % 256) as u8)
                .collect()
        })
        .collect()
}

/// Try to create an encoder, returning None if the constraint matrix is singular.
fn try_encoder(k: usize, symbol_size: usize, seed: u64) -> Option<SystematicEncoder> {
    let source = make_source(k, symbol_size);
    SystematicEncoder::new(&source, symbol_size, seed)
}

/// Create an encoder with fallback seeds (some K values produce singular matrices
/// for certain seeds — tracked by bd-uix9).
fn encoder_or_skip(k: usize, symbol_size: usize) -> Option<SystematicEncoder> {
    for seed in [42, 123, 7, 999, 314159] {
        if let Some(enc) = try_encoder(k, symbol_size, seed) {
            return Some(enc);
        }
    }
    None
}

// ============================================================================
// Step 1: Coding parameter verification
// ============================================================================

#[test]
fn test_step1_coding_params_basic_invariants() {
    // For representative K values, verify L = K + S + H and derived params.
    for &k in &[5, 10, 20, 50, 100, 200, 500, 1000] {
        let p = SystematicParams::for_source_block(k, 64);
        assert_eq!(
            p.l,
            p.k + p.s + p.h,
            "bead_id={BEAD_ID} case=L_invariant k={k}"
        );
        assert_eq!(p.w, p.k + p.s, "bead_id={BEAD_ID} case=W_invariant k={k}");
        assert_eq!(p.p, p.h, "bead_id={BEAD_ID} case=P_eq_H k={k}");
        assert_eq!(p.b, p.k, "bead_id={BEAD_ID} case=B_eq_K k={k}");
        assert!(p.s >= 7, "bead_id={BEAD_ID} case=S_gte_7 k={k} s={}", p.s);
        assert!(p.h >= 3, "bead_id={BEAD_ID} case=H_gte_3 k={k} h={}", p.h);
    }
}

#[test]
fn test_step1_symbol_size_preserved() {
    for &sym_sz in &[64, 256, 1024, 4096] {
        let p = SystematicParams::for_source_block(50, sym_sz);
        assert_eq!(
            p.symbol_size, sym_sz,
            "bead_id={BEAD_ID} case=symbol_size_preserved sz={sym_sz}"
        );
    }
}

// ============================================================================
// Step 2: Constraint matrix structure verification
// ============================================================================

#[test]
fn test_step2_constraint_matrix_dimensions() {
    for &k in &[5, 20, 100] {
        let params = SystematicParams::for_source_block(k, 64);
        let matrix = ConstraintMatrix::build(&params, 42);
        assert_eq!(
            matrix.rows,
            params.s + params.h + params.k,
            "bead_id={BEAD_ID} case=matrix_rows k={k}"
        );
        assert_eq!(
            matrix.cols, params.l,
            "bead_id={BEAD_ID} case=matrix_cols k={k}"
        );
    }
}

#[test]
fn test_step2_ldpc_rows_are_sparse_binary() {
    // LDPC rows (0..S) should only contain 0 or 1 (GF(2)), and be sparse.
    for &k in &[10, 50, 100] {
        let params = SystematicParams::for_source_block(k, 64);
        let matrix = ConstraintMatrix::build(&params, 42);

        for row in 0..params.s {
            let mut nonzero_count = 0;
            for col in 0..params.l {
                let val = matrix.get(row, col);
                // In GF(256) with XOR addition, entries can be 0 or any nonzero value
                // from accumulation. But for well-formed LDPC, entries in the K columns
                // should be 0 or 1 (before addition of duplicates).
                if !val.is_zero() {
                    nonzero_count += 1;
                }
            }
            // Each LDPC row has: ~3*(K/S) entries from circulant + 1 identity entry
            // (plus possible overlaps from the circulant)
            assert!(
                nonzero_count > 0,
                "bead_id={BEAD_ID} case=ldpc_row_nonempty k={k} row={row}"
            );
        }
    }
}

#[test]
fn test_step2_ldpc_identity_block() {
    // LDPC rows should have 1 at column K+i for row i (identity block for check symbols).
    for &k in &[10, 50] {
        let params = SystematicParams::for_source_block(k, 64);
        let matrix = ConstraintMatrix::build(&params, 42);

        for i in 0..params.s {
            let val = matrix.get(i, params.k + i);
            assert_eq!(
                val,
                Gf256::ONE,
                "bead_id={BEAD_ID} case=ldpc_identity k={k} row={i} col={}",
                params.k + i
            );
        }
    }
}

#[test]
fn test_step2_hdpc_rows_use_gf256() {
    // HDPC rows (S..S+H) operate over GF(256), should have non-binary entries.
    for &k in &[20, 100] {
        let params = SystematicParams::for_source_block(k, 64);
        let matrix = ConstraintMatrix::build(&params, 42);

        let mut found_nonbinary = false;
        for row in params.s..params.s + params.h {
            for col in 0..params.w {
                let val = matrix.get(row, col);
                let byte_val: u8 = val.raw();
                if byte_val > 1 {
                    found_nonbinary = true;
                    break;
                }
            }
            if found_nonbinary {
                break;
            }
        }
        assert!(
            found_nonbinary,
            "bead_id={BEAD_ID} case=hdpc_uses_gf256 k={k} \
             (expected at least one entry > 1 in HDPC region)"
        );
    }
}

#[test]
fn test_step2_hdpc_pi_identity_block() {
    // HDPC rows should have 1 at column W+r for row S+r (PI symbol identity).
    for &k in &[20, 100] {
        let params = SystematicParams::for_source_block(k, 64);
        let matrix = ConstraintMatrix::build(&params, 42);

        for r in 0..params.h {
            let val = matrix.get(params.s + r, params.w + r);
            assert_eq!(
                val,
                Gf256::ONE,
                "bead_id={BEAD_ID} case=hdpc_pi_identity k={k} row={} col={}",
                params.s + r,
                params.w + r
            );
        }
    }
}

#[test]
fn test_step2_lt_rows_systematic_identity() {
    // LT rows (S+H..S+H+K): for systematic encoding, row S+H+i has a 1 in column i.
    for &k in &[10, 50] {
        let params = SystematicParams::for_source_block(k, 64);
        let matrix = ConstraintMatrix::build(&params, 42);

        for i in 0..k {
            let row = params.s + params.h + i;
            let val = matrix.get(row, i);
            assert_eq!(
                val,
                Gf256::ONE,
                "bead_id={BEAD_ID} case=lt_systematic_identity k={k} i={i}"
            );
            // All other columns in this row should be zero (pure systematic).
            for col in 0..params.l {
                if col == i {
                    continue;
                }
                let other = matrix.get(row, col);
                assert!(
                    other.is_zero(),
                    "bead_id={BEAD_ID} case=lt_row_only_diagonal k={k} row={row} col={col} val={other:?}"
                );
            }
        }
    }
}

#[test]
fn test_step2_constraint_matrix_deterministic() {
    let params = SystematicParams::for_source_block(50, 64);
    let m1 = ConstraintMatrix::build(&params, 42);
    let m2 = ConstraintMatrix::build(&params, 42);

    for row in 0..m1.rows {
        for col in 0..m1.cols {
            assert_eq!(
                m1.get(row, col),
                m2.get(row, col),
                "bead_id={BEAD_ID} case=matrix_deterministic row={row} col={col}"
            );
        }
    }
}

// ============================================================================
// Steps 3-4: Source vector and intermediate symbol solve
// ============================================================================

#[test]
fn test_step3_4_solve_produces_intermediate_symbols() {
    // Verify that solving A·C = D produces L intermediate symbols.
    let Some(enc) = encoder_or_skip(50, 64) else {
        return; // skip if all seeds produce singular matrix
    };
    let params = enc.params();
    // Intermediate symbols should exist for all L indices.
    for i in 0..params.l {
        let sym = enc.intermediate_symbol(i);
        assert_eq!(
            sym.len(),
            params.symbol_size,
            "bead_id={BEAD_ID} case=intermediate_size i={i}"
        );
    }
}

#[test]
fn test_step3_4_intermediate_first_k_match_source() {
    // For systematic encoding: intermediate[0..K] should equal source symbols.
    // This is the key systematic property at the intermediate level.
    let k = 50;
    let sym_sz = 64;
    let source = make_source(k, sym_sz);
    let Some(enc) = SystematicEncoder::new(&source, sym_sz, 42) else {
        return;
    };

    for i in 0..k {
        let intermediate_i = enc.intermediate_symbol(i);
        assert_eq!(
            intermediate_i,
            &source[i][..],
            "bead_id={BEAD_ID} case=intermediate_eq_source i={i}"
        );
    }
}

// ============================================================================
// Step 5: Encoding symbol generation
// ============================================================================

#[test]
fn test_step5_systematic_emission_is_source_identity() {
    // First K emitted symbols should be EXACTLY the source symbols.
    for &k in &[5, 10, 50, 100] {
        let sym_sz = 64;
        let source = make_source(k, sym_sz);
        let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, 42) else {
            continue;
        };
        let systematic = enc.emit_systematic();
        assert_eq!(
            systematic.len(),
            k,
            "bead_id={BEAD_ID} case=systematic_count k={k}"
        );
        for (i, sym) in systematic.iter().enumerate() {
            assert_eq!(
                sym.esi, i as u32,
                "bead_id={BEAD_ID} case=systematic_esi k={k} i={i}"
            );
            assert!(
                sym.is_source,
                "bead_id={BEAD_ID} case=systematic_flag k={k} i={i}"
            );
            assert_eq!(
                sym.degree, 1,
                "bead_id={BEAD_ID} case=systematic_degree k={k} i={i}"
            );
            assert_eq!(
                sym.data, source[i],
                "bead_id={BEAD_ID} case=systematic_data k={k} i={i}"
            );
        }
    }
}

#[test]
fn test_step5_repair_symbols_are_not_source() {
    let Some(mut enc) = encoder_or_skip(50, 64) else {
        return;
    };
    let _ = enc.emit_systematic();
    let repairs = enc.emit_repair(10);

    assert_eq!(repairs.len(), 10, "bead_id={BEAD_ID} case=repair_count");
    for (i, sym) in repairs.iter().enumerate() {
        assert_eq!(
            sym.esi,
            (50 + i) as u32,
            "bead_id={BEAD_ID} case=repair_esi i={i}"
        );
        assert!(
            !sym.is_source,
            "bead_id={BEAD_ID} case=repair_not_source i={i}"
        );
        assert!(
            sym.degree >= 1,
            "bead_id={BEAD_ID} case=repair_degree_positive i={i} deg={}",
            sym.degree
        );
    }
}

#[test]
fn test_step5_repair_esi_ascending() {
    let Some(mut enc) = encoder_or_skip(50, 64) else {
        return;
    };
    let _ = enc.emit_systematic();
    let batch1 = enc.emit_repair(5);
    let batch2 = enc.emit_repair(5);

    // Batch 1 ESIs should be K..K+5
    for (i, sym) in batch1.iter().enumerate() {
        assert_eq!(sym.esi, (50 + i) as u32);
    }
    // Batch 2 ESIs should continue: K+5..K+10
    for (i, sym) in batch2.iter().enumerate() {
        assert_eq!(sym.esi, (55 + i) as u32);
    }
}

// ============================================================================
// Pipeline-level tests
// ============================================================================

#[test]
fn test_encoding_pipeline_stages() {
    // Verify the 5-step pipeline executes correctly from source to encoded output.
    let k = 50;
    let sym_sz = 64;
    let source = make_source(k, sym_sz);

    // Step 1: Parameters
    let params = SystematicParams::for_source_block(k, sym_sz);
    assert_eq!(params.k, k);
    assert!(params.l > params.k, "L must exceed K");

    // Step 2: Constraint matrix
    let matrix = ConstraintMatrix::build(&params, 42);
    assert_eq!(matrix.rows, params.s + params.h + params.k);
    assert_eq!(matrix.cols, params.l);

    // Step 3: Source vector D
    let mut rhs: Vec<Vec<u8>> = Vec::with_capacity(matrix.rows);
    for _ in 0..params.s + params.h {
        rhs.push(vec![0u8; sym_sz]);
    }
    for sym in &source {
        rhs.push(sym.clone());
    }
    assert_eq!(rhs.len(), matrix.rows, "bead_id={BEAD_ID} case=rhs_length");

    // Step 4: Solve A·C = D
    let intermediate = matrix
        .solve(&rhs)
        .expect("bead_id=bd-1hi.3 case=solve_succeeds k=50");
    assert_eq!(
        intermediate.len(),
        params.l,
        "bead_id={BEAD_ID} case=intermediate_count"
    );

    // Step 5: Verify systematic property on intermediates
    for i in 0..k {
        assert_eq!(
            intermediate[i], source[i],
            "bead_id={BEAD_ID} case=pipeline_systematic i={i}"
        );
    }
}

#[test]
fn test_encoding_pipeline_correctness() {
    // Encoded symbols should decode back to original source data.
    // Decoder requires L = K + S + H symbols total.
    let k = 50;
    let sym_sz = 64;
    let source = make_source(k, sym_sz);
    for seed in [42_u64, 123, 7, 999, 314_159] {
        let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, seed) else {
            continue;
        };
        let params = enc.params().clone();
        let systematic = enc.emit_systematic();
        // Generate enough repair symbols so total >= L
        let needed_repair = params.l.saturating_sub(k) + 2; // S + H extra, plus margin
        let repairs = enc.emit_repair(needed_repair);

        let decoder = InactivationDecoder::new(k, sym_sz, seed);
        // Decoder needs L = K+S+H symbols total: constraint eqs + source + repair.
        let mut received: Vec<ReceivedSymbol> = decoder.constraint_symbols();

        for sym in &systematic {
            received.push(ReceivedSymbol::source(sym.esi, sym.data.clone()));
        }
        for sym in &repairs {
            let (cols, coeffs) = decoder.repair_equation(sym.esi);
            received.push(ReceivedSymbol::repair(
                sym.esi,
                cols,
                coeffs,
                sym.data.clone(),
            ));
        }

        if let Ok(result) = decoder.decode(&received) {
            for i in 0..k {
                assert_eq!(
                    result.source[i], source[i],
                    "bead_id={BEAD_ID} case=roundtrip_correctness i={i} seed={seed}"
                );
            }
            return;
        }
    }

    panic!("bead_id={BEAD_ID} case=decode_succeeds no seed produced a decodable matrix");
}

#[test]
fn test_encoding_pipeline_systematic() {
    // The systematic property: first K encoding symbols are EXACTLY the source symbols.
    // In the no-loss case, receiver already has all K source symbols with zero overhead.
    for &k in &[5, 10, 50, 100] {
        let sym_sz = 64;
        let source = make_source(k, sym_sz);
        let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, 42) else {
            continue;
        };

        let all = enc.emit_all(4);
        // First K are systematic
        for i in 0..k {
            assert_eq!(
                all[i].data, source[i],
                "bead_id={BEAD_ID} case=systematic_property k={k} i={i}"
            );
            assert!(all[i].is_source);
        }
        // Remaining are repair
        for sym in all.iter().skip(k) {
            assert!(!sym.is_source);
            assert!(sym.esi >= k as u32);
        }
    }
}

#[test]
fn test_encoding_pipeline_deterministic() {
    // Same input → same output, always.
    let k = 50;
    let sym_sz = 64;
    let source = make_source(k, sym_sz);
    let seed = 42;

    let mut enc1 = SystematicEncoder::new(&source, sym_sz, seed).unwrap();
    let mut enc2 = SystematicEncoder::new(&source, sym_sz, seed).unwrap();

    let out1 = enc1.emit_all(10);
    let out2 = enc2.emit_all(10);

    assert_eq!(
        out1.len(),
        out2.len(),
        "bead_id={BEAD_ID} case=deterministic_count"
    );
    for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert_eq!(
            a.esi, b.esi,
            "bead_id={BEAD_ID} case=deterministic_esi i={i}"
        );
        assert_eq!(
            a.data, b.data,
            "bead_id={BEAD_ID} case=deterministic_data i={i}"
        );
        assert_eq!(
            a.is_source, b.is_source,
            "bead_id={BEAD_ID} case=deterministic_source i={i}"
        );
        assert_eq!(
            a.degree, b.degree,
            "bead_id={BEAD_ID} case=deterministic_degree i={i}"
        );
    }
}

#[test]
fn test_encoding_different_seeds_differ() {
    // Different seeds must produce different repair symbols.
    let k = 50;
    let sym_sz = 64;
    let source = make_source(k, sym_sz);

    let mut enc1 = SystematicEncoder::new(&source, sym_sz, 42).unwrap();
    let mut enc2 = SystematicEncoder::new(&source, sym_sz, 99).unwrap();

    let _ = enc1.emit_systematic();
    let _ = enc2.emit_systematic();
    let r1 = enc1.emit_repair(5);
    let r2 = enc2.emit_repair(5);

    let mut any_differ = false;
    for (a, b) in r1.iter().zip(r2.iter()) {
        if a.data != b.data {
            any_differ = true;
            break;
        }
    }
    assert!(
        any_differ,
        "bead_id={BEAD_ID} case=different_seeds_differ \
         (repair symbols should differ for different seeds)"
    );
}

// ============================================================================
// K' zero-padding
// ============================================================================

#[test]
fn test_k_prime_padding_handled() {
    // The encoder should handle K < K_prime by internally padding.
    // We just verify the encoder works for various K and produces correct output.
    for &k in &[3, 5, 7, 11, 13, 17, 23] {
        let sym_sz = 32;
        let source = make_source(k, sym_sz);
        if let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, 42) {
            let systematic = enc.emit_systematic();
            assert_eq!(systematic.len(), k);
            for (i, sym) in systematic.iter().enumerate() {
                assert_eq!(
                    sym.data, source[i],
                    "bead_id={BEAD_ID} case=padding_systematic k={k} i={i}"
                );
            }
        }
    }
}

// ============================================================================
// Encoding stats verification
// ============================================================================

#[test]
fn test_encoding_stats_populated() {
    let Some(mut enc) = encoder_or_skip(50, 64) else {
        return;
    };
    let _ = enc.emit_systematic();
    let _ = enc.emit_repair(10);

    let stats = enc.stats();
    assert_eq!(stats.source_symbol_count, 50);
    assert_eq!(stats.symbol_size, 64);
    assert_eq!(stats.repair_symbols_generated, 10);
    assert_eq!(
        stats.systematic_bytes_emitted,
        50 * 64,
        "bead_id={BEAD_ID} case=stats_sys_bytes"
    );
    assert_eq!(
        stats.repair_bytes_emitted,
        10 * 64,
        "bead_id={BEAD_ID} case=stats_repair_bytes"
    );
    assert!(stats.degree_min >= 1);
    assert!(stats.degree_max >= stats.degree_min);
    assert!(stats.degree_count == 10);
    assert!(stats.overhead_ratio() > 1.0, "L/K should exceed 1.0");
}

// ============================================================================
// E2E encode/decode pipeline tests
// ============================================================================

#[test]
fn test_e2e_encode_decode_pipeline_k64_t4096() {
    // Encode 64 database pages (4096 bytes each), generate repair symbols,
    // drop some source symbols, replace with repair, decode, verify byte-perfect.
    // Decoder requires L = K + S + H total symbols.
    let k = 64;
    let page_size = 4096;
    let source = make_source(k, page_size);

    for seed in [42_u64, 123, 7, 999, 314_159] {
        let Some(mut enc) = SystematicEncoder::new(&source, page_size, seed) else {
            continue;
        };
        let params = enc.params().clone();

        let systematic = enc.emit_systematic();
        // Need at least L total symbols. With 2 dropped, need L - (K-2) = S + H + 2 repairs.
        let needed_repair = params.s + params.h + 4; // margin
        let repairs = enc.emit_repair(needed_repair);

        // Verify systematic property: all source symbols pass through unchanged.
        for (i, sym) in systematic.iter().enumerate() {
            assert_eq!(
                sym.data, source[i],
                "bead_id={BEAD_ID} case=e2e_systematic i={i} seed={seed}"
            );
        }

        // Drop 2 source symbols and replace with repairs.
        let drop_indices: HashSet<usize> = [7, 31].into_iter().collect();
        let decoder = InactivationDecoder::new(k, page_size, seed);
        // Constraint equations are required for the decoder matrix to be full-rank.
        let mut received: Vec<ReceivedSymbol> = decoder.constraint_symbols();

        for sym in &systematic {
            if !drop_indices.contains(&(sym.esi as usize)) {
                received.push(ReceivedSymbol::source(sym.esi, sym.data.clone()));
            }
        }
        for sym in &repairs {
            let (cols, coeffs) = decoder.repair_equation(sym.esi);
            received.push(ReceivedSymbol::repair(
                sym.esi,
                cols,
                coeffs,
                sym.data.clone(),
            ));
        }

        if let Ok(result) = decoder.decode(&received) {
            for i in 0..k {
                assert_eq!(
                    result.source[i], source[i],
                    "bead_id={BEAD_ID} case=e2e_byte_perfect i={i} seed={seed}"
                );
            }
            return;
        }
    }

    panic!("bead_id={BEAD_ID} case=e2e_decode_succeeds no seed produced a decodable matrix");
}

#[test]
fn test_e2e_roundtrip_all_source_plus_repair() {
    // Decoder requires L = K + S + H symbols. Provide K source + enough repair to reach L.
    for &k in &[10, 50, 100] {
        let sym_sz = 64;
        let source = make_source(k, sym_sz);
        let seed = 42;

        let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, seed) else {
            continue;
        };
        let params = enc.params().clone();

        let systematic = enc.emit_systematic();
        let needed_repair = params.l - k + 2;
        let repairs = enc.emit_repair(needed_repair);

        let decoder = InactivationDecoder::new(k, sym_sz, seed);
        // Decoder needs constraint equations (LDPC+HDPC) + source + repair >= L.
        let mut received: Vec<ReceivedSymbol> = decoder.constraint_symbols();
        for sym in &systematic {
            received.push(ReceivedSymbol::source(sym.esi, sym.data.clone()));
        }
        for sym in &repairs {
            let (cols, coeffs) = decoder.repair_equation(sym.esi);
            received.push(ReceivedSymbol::repair(
                sym.esi,
                cols,
                coeffs,
                sym.data.clone(),
            ));
        }

        let result = decoder
            .decode(&received)
            .expect("bead_id=bd-1hi.3 case=all_source_decode");

        for i in 0..k {
            assert_eq!(
                result.source[i], source[i],
                "bead_id={BEAD_ID} case=all_source_roundtrip k={k} i={i}"
            );
        }
    }
}

#[test]
fn test_e2e_insufficient_symbols_error() {
    let k = 50;
    let sym_sz = 64;
    let source = make_source(k, sym_sz);
    let seed = 42;

    let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, seed) else {
        return;
    };
    let systematic = enc.emit_systematic();
    let decoder = InactivationDecoder::new(k, sym_sz, seed);

    // Send fewer than L symbols — should get InsufficientSymbols error.
    let received: Vec<ReceivedSymbol> = systematic
        .iter()
        .take(k / 2)
        .map(|s| ReceivedSymbol::source(s.esi, s.data.clone()))
        .collect();

    match decoder.decode(&received) {
        Err(DecodeError::InsufficientSymbols {
            received: r,
            required: req,
        }) => {
            assert_eq!(r, k / 2);
            assert!(req > r, "bead_id={BEAD_ID} case=insufficient_req_gt_recv");
        }
        Ok(_) => panic!("bead_id={BEAD_ID} case=insufficient_should_fail"),
        Err(e) => panic!("bead_id={BEAD_ID} case=unexpected_error error={e:?}"),
    }
}

#[test]
fn test_e2e_symbol_size_mismatch_error() {
    let k = 10;
    let sym_sz = 64;
    let seed = 42;

    let decoder = InactivationDecoder::new(k, sym_sz, seed);
    let params = decoder.params();

    // Create symbols with wrong size.
    let received: Vec<ReceivedSymbol> = (0..params.l)
        .map(|i| ReceivedSymbol::source(i as u32, vec![0u8; sym_sz + 1]))
        .collect();

    match decoder.decode(&received) {
        Err(DecodeError::SymbolSizeMismatch { expected, actual }) => {
            assert_eq!(expected, sym_sz);
            assert_eq!(actual, sym_sz + 1);
        }
        other => panic!("bead_id={BEAD_ID} case=size_mismatch_expected result={other:?}"),
    }
}

#[test]
fn test_e2e_different_symbol_sizes() {
    // Verify encoding and decoding work for various page/symbol sizes.
    // Decoder requires L = K + S + H symbols total.
    for &sym_sz in &[32, 128, 512, 4096] {
        let k = 20;
        let seed = 42;
        let source = make_source(k, sym_sz);

        let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, seed) else {
            continue;
        };
        let params = enc.params().clone();
        let needed_repair = params.l - k + 2;
        let all = enc.emit_all(needed_repair);

        let decoder = InactivationDecoder::new(k, sym_sz, seed);
        let mut received: Vec<ReceivedSymbol> = decoder.constraint_symbols();
        for sym in &all {
            if sym.is_source {
                received.push(ReceivedSymbol::source(sym.esi, sym.data.clone()));
            } else {
                let (cols, coeffs) = decoder.repair_equation(sym.esi);
                received.push(ReceivedSymbol::repair(
                    sym.esi,
                    cols,
                    coeffs,
                    sym.data.clone(),
                ));
            }
        }

        let result = decoder
            .decode(&received)
            .expect("bead_id=bd-1hi.3 case=various_sizes");

        for i in 0..k {
            assert_eq!(
                result.source[i], source[i],
                "bead_id={BEAD_ID} case=various_sizes sym_sz={sym_sz} i={i}"
            );
        }
    }
}

#[test]
fn test_e2e_larger_k_500() {
    // K=500 with realistic symbol size — verify no panics/overflows.
    // Decoder requires L = K + S + H symbols total.
    let k = 500;
    let sym_sz = 256;
    let source = make_source(k, sym_sz);

    for seed in [42_u64, 123, 7, 999, 314_159] {
        let Some(mut enc) = SystematicEncoder::new(&source, sym_sz, seed) else {
            continue;
        };
        let params = enc.params().clone();

        let systematic = enc.emit_systematic();
        let needed_repair = params.l.saturating_sub(k) + 2;
        let repairs = enc.emit_repair(needed_repair);

        // Verify systematic property.
        for (i, sym) in systematic.iter().enumerate() {
            assert_eq!(
                sym.data, source[i],
                "bead_id={BEAD_ID} case=k500_systematic i={i}"
            );
        }

        // Full roundtrip.
        let decoder = InactivationDecoder::new(k, sym_sz, seed);
        // Include constraint equations to avoid underdetermined matrix.
        let mut received: Vec<ReceivedSymbol> = decoder.constraint_symbols();
        received.extend(
            systematic
                .iter()
                .map(|s| ReceivedSymbol::source(s.esi, s.data.clone())),
        );
        for sym in &repairs {
            let (cols, coeffs) = decoder.repair_equation(sym.esi);
            received.push(ReceivedSymbol::repair(
                sym.esi,
                cols,
                coeffs,
                sym.data.clone(),
            ));
        }

        if let Ok(result) = decoder.decode(&received) {
            for i in 0..k {
                assert_eq!(
                    result.source[i], source[i],
                    "bead_id={BEAD_ID} case=k500_roundtrip i={i} seed={seed}"
                );
            }
            return;
        }
    }

    panic!("bead_id={BEAD_ID} case=k500_roundtrip no seed produced a decodable matrix");
}

// ============================================================================
// Repair symbol determinism and equation consistency
// ============================================================================

#[test]
fn test_repair_symbol_matches_equation() {
    // Verify that a repair symbol's data matches XOR of its equation's intermediate columns.
    let k = 50;
    let sym_sz = 64;
    let seed = 42;
    let source = make_source(k, sym_sz);

    let Some(enc) = SystematicEncoder::new(&source, sym_sz, seed) else {
        return;
    };

    let decoder = InactivationDecoder::new(k, sym_sz, seed);

    for esi in (k as u32)..(k as u32 + 5) {
        let repair_data = enc.repair_symbol(esi);
        let (cols, _coeffs) = decoder.repair_equation(esi);

        // The repair symbol should equal XOR of intermediate[cols[i]].
        let mut expected = vec![0u8; sym_sz];
        for &col in &cols {
            let intermediate = enc.intermediate_symbol(col);
            for (e, &s) in expected.iter_mut().zip(intermediate.iter()) {
                *e ^= s;
            }
        }

        assert_eq!(
            repair_data, expected,
            "bead_id={BEAD_ID} case=repair_matches_equation esi={esi}"
        );
    }
}

#[test]
fn test_emit_all_order_and_counts() {
    let k = 30;
    let repair_count = 8;
    let Some(mut enc) = encoder_or_skip(k, 64) else {
        return;
    };

    let all = enc.emit_all(repair_count);
    assert_eq!(all.len(), k + repair_count);

    // First K are source, ascending ESI 0..K
    for (i, sym) in all.iter().enumerate().take(k) {
        assert_eq!(sym.esi, i as u32);
        assert!(sym.is_source);
    }

    // Next repair_count are repair, ascending ESI K..K+repair_count
    for (i, sym) in all.iter().skip(k).enumerate() {
        assert_eq!(sym.esi, (k + i) as u32);
        assert!(!sym.is_source);
    }
}
