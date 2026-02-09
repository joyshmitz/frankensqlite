//! §3.3 Asupersync RaptorQ Pipeline Integration (bd-1hi.5).
//!
//! This module provides the FrankenSQLite-side wrapper types for the
//! asupersync RaptorQ pipeline.  Production code uses abstract traits
//! (`PageSymbolSink`, `PageSymbolSource`, `SymbolCodec`) so that the
//! actual asupersync dependency remains dev-only.
//!
//! # Cx Cancellation
//!
//! All long-running encode/decode loops call `cx.checkpoint()` every
//! `checkpoint_interval` symbols (§4.12.1).  If the context is cancelled
//! the operation returns `FrankenError::Abort`.

use fsqlite_error::{FrankenError, Result};
use fsqlite_types::cx::Cx;
use tracing::{debug, error, info, warn};

const BEAD_ID: &str = "bd-1hi.5";

// ---------------------------------------------------------------------------
// Pipeline Configuration (§3.3)
// ---------------------------------------------------------------------------

/// Minimum allowed symbol size (bytes).
pub const MIN_PIPELINE_SYMBOL_SIZE: u32 = 512;

/// Maximum allowed symbol size (bytes).
pub const MAX_PIPELINE_SYMBOL_SIZE: u32 = 65_536;

/// Default Cx checkpoint interval (symbols between cancellation checks).
pub const DEFAULT_CHECKPOINT_INTERVAL: u32 = 64;

/// FrankenSQLite-side RaptorQ pipeline configuration (§3.3).
///
/// Mirrors the needed subset of asupersync's `RaptorQConfig` so that
/// production code does not depend on asupersync directly.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineConfig {
    /// Symbol size T in bytes.  Must be a power of two in
    /// `[MIN_PIPELINE_SYMBOL_SIZE, MAX_PIPELINE_SYMBOL_SIZE]`.
    pub symbol_size: u32,
    /// Maximum source block size (max K per source block) in bytes.
    pub max_block_size: u32,
    /// Repair overhead factor.  E.g. `1.25` means 25 % extra repair symbols.
    pub repair_overhead: f64,
    /// Symbols between `Cx::checkpoint()` calls (§4.12.1).
    pub checkpoint_interval: u32,
}

impl PipelineConfig {
    /// Create a configuration for page-sized symbols (T = page_size).
    #[must_use]
    pub fn for_page_size(page_size: u32) -> Self {
        Self {
            symbol_size: page_size,
            max_block_size: 64 * 1024,
            repair_overhead: 1.25,
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
        }
    }

    /// Validate this configuration.
    ///
    /// Rejects:
    /// - `symbol_size == 0`
    /// - `symbol_size` not a power of two
    /// - `symbol_size` outside `[MIN, MAX]`
    /// - `max_block_size == 0`
    /// - `repair_overhead < 1.0`
    /// - `checkpoint_interval == 0`
    pub fn validate(&self) -> Result<()> {
        if self.symbol_size == 0 {
            return Err(FrankenError::OutOfRange {
                what: "pipeline symbol_size".to_owned(),
                value: "0".to_owned(),
            });
        }
        if !self.symbol_size.is_power_of_two() {
            return Err(FrankenError::OutOfRange {
                what: "pipeline symbol_size (must be power of 2)".to_owned(),
                value: self.symbol_size.to_string(),
            });
        }
        if self.symbol_size < MIN_PIPELINE_SYMBOL_SIZE
            || self.symbol_size > MAX_PIPELINE_SYMBOL_SIZE
        {
            return Err(FrankenError::OutOfRange {
                what: format!(
                    "pipeline symbol_size (must be in [{MIN_PIPELINE_SYMBOL_SIZE}, {MAX_PIPELINE_SYMBOL_SIZE}])"
                ),
                value: self.symbol_size.to_string(),
            });
        }
        if self.max_block_size == 0 {
            return Err(FrankenError::OutOfRange {
                what: "pipeline max_block_size".to_owned(),
                value: "0".to_owned(),
            });
        }
        if self.repair_overhead < 1.0 {
            return Err(FrankenError::OutOfRange {
                what: "pipeline repair_overhead (must be >= 1.0)".to_owned(),
                value: self.repair_overhead.to_string(),
            });
        }
        if self.checkpoint_interval == 0 {
            return Err(FrankenError::OutOfRange {
                what: "pipeline checkpoint_interval".to_owned(),
                value: "0".to_owned(),
            });
        }
        Ok(())
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::for_page_size(4096)
    }
}

// ---------------------------------------------------------------------------
// Page Symbol Sink / Source Traits (§3.3)
// ---------------------------------------------------------------------------

/// Writes encoded page symbols to WAL/ECS storage.
pub trait PageSymbolSink {
    /// Write a single encoded symbol.
    fn write_symbol(&mut self, esi: u32, data: &[u8]) -> Result<()>;

    /// Flush all buffered symbols to durable storage.
    fn flush(&mut self) -> Result<()>;

    /// Number of symbols written so far.
    fn written_count(&self) -> u32;
}

/// Reads symbols from WAL/ECS storage for decoding.
pub trait PageSymbolSource {
    /// Read a symbol by its ESI.  Returns `None` if unavailable (erased).
    fn read_symbol(&mut self, esi: u32) -> Result<Option<Vec<u8>>>;

    /// All available ESIs in this source.
    fn available_esis(&self) -> Vec<u32>;

    /// Number of available symbols.
    fn available_count(&self) -> u32;
}

// ---------------------------------------------------------------------------
// Symbol Codec Trait (§3.3)
// ---------------------------------------------------------------------------

/// Abstraction over the actual RaptorQ encode/decode engine.
///
/// In production, this wraps asupersync's `RaptorQSenderBuilder` /
/// `RaptorQReceiverBuilder`.  In tests, it may be a mock.
pub trait SymbolCodec: Send + Sync {
    /// Encode source data into source + repair symbols.
    fn encode(
        &self,
        source_data: &[u8],
        symbol_size: u32,
        repair_overhead: f64,
    ) -> Result<CodecEncodeResult>;

    /// Decode from received symbols.
    fn decode(
        &self,
        symbols: &[(u32, Vec<u8>)],
        k_source: u32,
        symbol_size: u32,
    ) -> Result<CodecDecodeResult>;
}

/// Raw encode result from the codec.
#[derive(Debug, Clone)]
pub struct CodecEncodeResult {
    /// Source symbols: `(esi, data)`.
    pub source_symbols: Vec<(u32, Vec<u8>)>,
    /// Repair symbols: `(esi, data)`.
    pub repair_symbols: Vec<(u32, Vec<u8>)>,
    /// Number of source symbols K.
    pub k_source: u32,
}

/// Raw decode result from the codec.
#[derive(Debug, Clone)]
pub enum CodecDecodeResult {
    /// Decode succeeded.
    Success {
        /// Recovered source data.
        data: Vec<u8>,
        /// Number of symbols consumed.
        symbols_used: u32,
        /// Symbols resolved by peeling.
        peeled_count: u32,
        /// Symbols resolved by Gaussian elimination (inactive subsystem).
        inactivated_count: u32,
    },
    /// Decode failed.
    Failure {
        /// Reason for failure.
        reason: DecodeFailureReason,
        /// Number of symbols that were received.
        symbols_received: u32,
        /// Source symbols required.
        k_required: u32,
    },
}

// ---------------------------------------------------------------------------
// Outcome Types (§3.3)
// ---------------------------------------------------------------------------

/// Result of a pipeline encode operation.
#[derive(Debug, Clone)]
pub struct EncodeOutcome {
    /// Number of source symbols produced.
    pub source_count: u32,
    /// Number of repair symbols produced.
    pub repair_count: u32,
    /// Symbol size in bytes.
    pub symbol_size: u32,
}

/// Result of a pipeline decode operation.
#[derive(Debug, Clone)]
pub enum DecodeOutcome {
    /// Successful decode with recovered pages.
    Success(DecodeSuccess),
    /// Failed decode with diagnostic information.
    Failure(DecodeFailure),
}

/// Successful decode metadata.
#[derive(Debug, Clone)]
pub struct DecodeSuccess {
    /// Recovered page data, concatenated.
    pub data: Vec<u8>,
    /// Number of symbols used for decoding.
    pub symbols_used: u32,
    /// Symbols resolved during the peeling phase.
    pub peeled_count: u32,
    /// Symbols resolved during the Gaussian elimination phase.
    pub inactivated_count: u32,
}

/// Failed decode metadata.
#[derive(Debug, Clone)]
pub struct DecodeFailure {
    /// Why the decode failed.
    pub reason: DecodeFailureReason,
    /// Number of symbols that were available.
    pub symbols_received: u32,
    /// Source symbols required (K).
    pub k_required: u32,
}

/// Reasons a decode can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeFailureReason {
    /// Fewer symbols than K available.
    InsufficientSymbols,
    /// The decoding matrix is singular (rank deficient).
    SingularMatrix,
    /// Symbol sizes do not match the expected T.
    SymbolSizeMismatch,
    /// Cancelled via `Cx::checkpoint()`.
    Cancelled,
}

// ---------------------------------------------------------------------------
// Pipeline Encoder (§3.3)
// ---------------------------------------------------------------------------

/// RaptorQ page encoder that wraps a [`SymbolCodec`] and writes through
/// a [`PageSymbolSink`] with Cx cancellation checkpoints.
pub struct RaptorQPageEncoder<C: SymbolCodec> {
    config: PipelineConfig,
    codec: C,
}

impl<C: SymbolCodec> RaptorQPageEncoder<C> {
    /// Create a new encoder.  Validates the config eagerly.
    pub fn new(config: PipelineConfig, codec: C) -> Result<Self> {
        config.validate()?;
        info!(
            bead_id = BEAD_ID,
            symbol_size = config.symbol_size,
            max_block_size = config.max_block_size,
            repair_overhead = config.repair_overhead,
            "RaptorQ page encoder created"
        );
        Ok(Self { config, codec })
    }

    /// Encode page data and write symbols through the sink.
    ///
    /// `cx.checkpoint()` is called every `checkpoint_interval` symbols.
    #[allow(clippy::cast_possible_truncation)]
    pub fn encode_pages(
        &self,
        cx: &Cx,
        page_data: &[u8],
        sink: &mut dyn PageSymbolSink,
    ) -> Result<EncodeOutcome> {
        cx.checkpoint().map_err(|_| FrankenError::Abort)?;

        let symbol_size = self.config.symbol_size;
        debug!(
            bead_id = BEAD_ID,
            data_len = page_data.len(),
            symbol_size,
            "starting page encode"
        );

        let result = self
            .codec
            .encode(page_data, symbol_size, self.config.repair_overhead)?;

        // Write source symbols with checkpoints.
        let interval = self.config.checkpoint_interval as usize;
        for (idx, (esi, data)) in result.source_symbols.iter().enumerate() {
            if idx > 0 && idx % interval == 0 {
                cx.checkpoint().map_err(|_| FrankenError::Abort)?;
            }
            sink.write_symbol(*esi, data)?;
        }

        // Write repair symbols with checkpoints.
        for (idx, (esi, data)) in result.repair_symbols.iter().enumerate() {
            if idx > 0 && idx % interval == 0 {
                cx.checkpoint().map_err(|_| FrankenError::Abort)?;
            }
            sink.write_symbol(*esi, data)?;
        }

        sink.flush()?;

        let outcome = EncodeOutcome {
            source_count: result.k_source,
            repair_count: result.repair_symbols.len() as u32,
            symbol_size,
        };

        info!(
            bead_id = BEAD_ID,
            source_count = outcome.source_count,
            repair_count = outcome.repair_count,
            symbol_size = outcome.symbol_size,
            "page encode complete"
        );

        Ok(outcome)
    }

    /// Reference to the pipeline config.
    #[must_use]
    pub const fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Pipeline Decoder (§3.3)
// ---------------------------------------------------------------------------

/// RaptorQ page decoder that wraps a [`SymbolCodec`] and reads from
/// a [`PageSymbolSource`] with Cx cancellation checkpoints.
pub struct RaptorQPageDecoder<C: SymbolCodec> {
    config: PipelineConfig,
    codec: C,
}

impl<C: SymbolCodec> RaptorQPageDecoder<C> {
    /// Create a new decoder.  Validates the config eagerly.
    pub fn new(config: PipelineConfig, codec: C) -> Result<Self> {
        config.validate()?;
        info!(
            bead_id = BEAD_ID,
            symbol_size = config.symbol_size,
            "RaptorQ page decoder created"
        );
        Ok(Self { config, codec })
    }

    /// Decode pages from the source.
    ///
    /// Reads available symbols, delegates to the codec, and returns the
    /// outcome.  Cx checkpoint is called at read boundaries.
    #[allow(clippy::cast_possible_truncation)]
    pub fn decode_pages(
        &self,
        cx: &Cx,
        source: &mut dyn PageSymbolSource,
        k_source: u32,
    ) -> Result<DecodeOutcome> {
        cx.checkpoint().map_err(|_| FrankenError::Abort)?;

        let available = source.available_count();
        debug!(
            bead_id = BEAD_ID,
            k_source, available, "starting page decode"
        );

        if available < k_source {
            warn!(
                bead_id = BEAD_ID,
                k_source, available, "fewer symbols than K_source — decode likely to fail"
            );
        }

        // Collect symbols from source with checkpoints.
        let esis = source.available_esis();
        let interval = self.config.checkpoint_interval as usize;
        let mut symbols = Vec::with_capacity(esis.len());
        for (idx, esi) in esis.iter().enumerate() {
            if idx > 0 && idx % interval == 0 {
                cx.checkpoint().map_err(|_| FrankenError::Abort)?;
            }
            if let Some(data) = source.read_symbol(*esi)? {
                symbols.push((*esi, data));
            }
        }

        // Delegate to codec.
        let codec_result = self
            .codec
            .decode(&symbols, k_source, self.config.symbol_size)?;

        match codec_result {
            CodecDecodeResult::Success {
                data,
                symbols_used,
                peeled_count,
                inactivated_count,
            } => {
                info!(
                    bead_id = BEAD_ID,
                    k_source,
                    symbols_used,
                    peeled_count,
                    inactivated_count,
                    "page decode succeeded"
                );
                if symbols_used == k_source {
                    warn!(
                        bead_id = BEAD_ID,
                        k_source,
                        symbols_used,
                        "fragile recovery: decoded with minimum symbol count"
                    );
                }
                Ok(DecodeOutcome::Success(DecodeSuccess {
                    data,
                    symbols_used,
                    peeled_count,
                    inactivated_count,
                }))
            }
            CodecDecodeResult::Failure {
                reason,
                symbols_received,
                k_required,
            } => {
                error!(
                    bead_id = BEAD_ID,
                    k_source,
                    symbols_received,
                    k_required,
                    reason = ?reason,
                    "page decode failed"
                );
                Ok(DecodeOutcome::Failure(DecodeFailure {
                    reason,
                    symbols_received,
                    k_required,
                }))
            }
        }
    }

    /// Reference to the pipeline config.
    #[must_use]
    pub const fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use asupersync::raptorq::decoder::InactivationDecoder;
    use asupersync::raptorq::systematic::ConstraintMatrix;
    use asupersync::raptorq::RaptorQSenderBuilder;
    use asupersync::security::AuthenticationTag;
    use asupersync::security::authenticated::AuthenticatedSymbol;
    use asupersync::transport::error::{SinkError, StreamError};
    use asupersync::transport::sink::SymbolSink;
    use asupersync::transport::stream::SymbolStream;
    use asupersync::types::{ObjectId as AsObjectId, Symbol};
    use asupersync::{Cx as AsCx, RaptorQConfig};

    use super::*;

    // -----------------------------------------------------------------------
    // Mock PageSymbolSink / PageSymbolSource
    // -----------------------------------------------------------------------

    struct VecPageSink {
        symbols: BTreeMap<u32, Vec<u8>>,
        flushed: bool,
    }

    impl VecPageSink {
        fn new() -> Self {
            Self {
                symbols: BTreeMap::new(),
                flushed: false,
            }
        }
    }

    impl PageSymbolSink for VecPageSink {
        fn write_symbol(&mut self, esi: u32, data: &[u8]) -> Result<()> {
            self.symbols.insert(esi, data.to_vec());
            Ok(())
        }

        fn flush(&mut self) -> Result<()> {
            self.flushed = true;
            Ok(())
        }

        fn written_count(&self) -> u32 {
            self.symbols.len() as u32
        }
    }

    struct VecPageSource {
        symbols: BTreeMap<u32, Vec<u8>>,
    }

    impl VecPageSource {
        fn from_sink(sink: &VecPageSink) -> Self {
            Self {
                symbols: sink.symbols.clone(),
            }
        }

        fn from_map(symbols: BTreeMap<u32, Vec<u8>>) -> Self {
            Self { symbols }
        }
    }

    impl PageSymbolSource for VecPageSource {
        fn read_symbol(&mut self, esi: u32) -> Result<Option<Vec<u8>>> {
            Ok(self.symbols.get(&esi).cloned())
        }

        fn available_esis(&self) -> Vec<u32> {
            self.symbols.keys().copied().collect()
        }

        fn available_count(&self) -> u32 {
            self.symbols.len() as u32
        }
    }

    // -----------------------------------------------------------------------
    // Asupersync-backed SymbolCodec implementation
    // -----------------------------------------------------------------------

    #[derive(Debug)]
    struct VecTransportSink {
        symbols: Vec<Symbol>,
    }

    impl VecTransportSink {
        fn new() -> Self {
            Self {
                symbols: Vec::new(),
            }
        }
    }

    impl SymbolSink for VecTransportSink {
        fn poll_send(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            symbol: AuthenticatedSymbol,
        ) -> Poll<std::result::Result<(), SinkError>> {
            self.symbols.push(symbol.into_symbol());
            Poll::Ready(Ok(()))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), SinkError>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), SinkError>> {
            Poll::Ready(Ok(()))
        }

        fn poll_ready(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), SinkError>> {
            Poll::Ready(Ok(()))
        }
    }

    #[derive(Debug)]
    struct VecTransportStream {
        q: VecDeque<AuthenticatedSymbol>,
    }

    impl VecTransportStream {
        fn new(symbols: Vec<Symbol>) -> Self {
            let q = symbols
                .into_iter()
                .map(|s| AuthenticatedSymbol::new_verified(s, AuthenticationTag::zero()))
                .collect();
            Self { q }
        }
    }

    impl SymbolStream for VecTransportStream {
        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<std::result::Result<AuthenticatedSymbol, StreamError>>> {
            match self.q.pop_front() {
                Some(s) => Poll::Ready(Some(Ok(s))),
                None => Poll::Ready(None),
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.q.len(), Some(self.q.len()))
        }

        fn is_exhausted(&self) -> bool {
            self.q.is_empty()
        }
    }

    /// SymbolCodec backed by asupersync.
    struct AsupersyncCodec;

    impl SymbolCodec for AsupersyncCodec {
        fn encode(
            &self,
            source_data: &[u8],
            symbol_size: u32,
            repair_overhead: f64,
        ) -> Result<CodecEncodeResult> {
            let mut config = RaptorQConfig::default();
            config.encoding.symbol_size = symbol_size as u16;
            config.encoding.max_block_size = 64 * 1024;
            config.encoding.repair_overhead = repair_overhead;

            let cx = AsCx::for_testing();
            let object_id = AsObjectId::new_for_test(0xBD_1A15);
            let mut sender = RaptorQSenderBuilder::new()
                .config(config)
                .transport(VecTransportSink::new())
                .build()
                .map_err(|e| FrankenError::Internal(format!("sender build: {e}")))?;

            let outcome = sender
                .send_object(&cx, object_id, source_data)
                .map_err(|e| FrankenError::Internal(format!("send_object: {e}")))?;

            let symbols = std::mem::take(&mut sender.transport_mut().symbols);
            let k = outcome.source_symbols as u32;

            let mut source_symbols = Vec::new();
            let mut repair_symbols = Vec::new();
            for s in &symbols {
                let esi = s.esi();
                if esi < k {
                    source_symbols.push((esi, s.data().to_vec()));
                } else {
                    repair_symbols.push((esi, s.data().to_vec()));
                }
            }

            Ok(CodecEncodeResult {
                source_symbols,
                repair_symbols,
                k_source: k,
            })
        }

        fn decode(
            &self,
            symbols: &[(u32, Vec<u8>)],
            k_source: u32,
            symbol_size: u32,
        ) -> Result<CodecDecodeResult> {
            if symbols.is_empty() {
                return Ok(CodecDecodeResult::Failure {
                    reason: DecodeFailureReason::InsufficientSymbols,
                    symbols_received: 0,
                    k_required: k_source,
                });
            }

            // Use low-level decoder for detailed stats.
            let seed = 0xBD_1A15_u64;
            let decoder = InactivationDecoder::new(k_source as usize, symbol_size as usize, seed);
            let params = decoder.params();
            let base_rows = params.s + params.h;
            let constraints = ConstraintMatrix::build(params, seed);

            let mut received = decoder.constraint_symbols();
            for (esi, data) in symbols {
                let esi_usize = *esi as usize;
                if esi_usize < k_source as usize {
                    // Source symbol: construct LT equation from constraint matrix.
                    let row = base_rows + esi_usize;
                    let mut columns = Vec::new();
                    let mut coefficients = Vec::new();
                    for col in 0..constraints.cols {
                        let coeff = constraints.get(row, col);
                        if !coeff.is_zero() {
                            columns.push(col);
                            coefficients.push(coeff);
                        }
                    }
                    received.push(asupersync::raptorq::decoder::ReceivedSymbol {
                        esi: *esi,
                        is_source: true,
                        columns,
                        coefficients,
                        data: data.clone(),
                    });
                } else {
                    // Repair symbol: use the decoder's repair equation.
                    let (columns, coefficients) = decoder.repair_equation(*esi);
                    received.push(asupersync::raptorq::decoder::ReceivedSymbol::repair(
                        *esi,
                        columns,
                        coefficients,
                        data.clone(),
                    ));
                }
            }

            match decoder.decode(&received) {
                Ok(result) => {
                    let flat: Vec<u8> = result
                        .source
                        .into_iter()
                        .flat_map(|v| v.into_iter())
                        .collect();
                    Ok(CodecDecodeResult::Success {
                        data: flat,
                        symbols_used: symbols.len() as u32,
                        peeled_count: result.stats.peeled as u32,
                        inactivated_count: result.stats.inactivated as u32,
                    })
                }
                Err(_) => Ok(CodecDecodeResult::Failure {
                    reason: if (symbols.len() as u32) < k_source {
                        DecodeFailureReason::InsufficientSymbols
                    } else {
                        DecodeFailureReason::SingularMatrix
                    },
                    symbols_received: symbols.len() as u32,
                    k_required: k_source,
                }),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn deterministic_page_data(k: usize, symbol_size: usize, seed: u64) -> Vec<u8> {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let total = k * symbol_size;
        let mut out = Vec::with_capacity(total);
        for idx in 0..total {
            state ^= state << 7;
            state ^= state >> 9;
            state = state.wrapping_mul(0xA24B_AED4_963E_E407);
            let idx_byte = (idx % 251) as u8;
            out.push((state & 0xFF) as u8 ^ idx_byte);
        }
        out
    }

    fn test_cx() -> fsqlite_types::cx::Cx {
        fsqlite_types::cx::Cx::new()
    }

    fn default_codec() -> AsupersyncCodec {
        AsupersyncCodec
    }

    fn default_config() -> PipelineConfig {
        PipelineConfig::for_page_size(512)
    }

    // -----------------------------------------------------------------------
    // §3.3 Test 12: Pipeline encode (test_pipeline_encode_async)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_encode_produces_source_and_repair() {
        let config = default_config();
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
        let cx = test_cx();
        let k = 10_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0x1234);

        let mut sink = VecPageSink::new();
        let outcome = encoder
            .encode_pages(&cx, &data, &mut sink)
            .expect("encode must succeed");

        assert_eq!(
            outcome.source_count as usize, k,
            "bead_id={BEAD_ID} case=encode_source_count"
        );
        assert!(
            outcome.repair_count > 0,
            "bead_id={BEAD_ID} case=encode_repair_present"
        );
        assert_eq!(
            outcome.symbol_size, config.symbol_size,
            "bead_id={BEAD_ID} case=encode_symbol_size"
        );
        assert!(sink.flushed, "bead_id={BEAD_ID} case=encode_sink_flushed");

        // Verify source symbols contain original page data.
        let sym_size = config.symbol_size as usize;
        for i in 0..k {
            let esi = i as u32;
            let expected = &data[i * sym_size..(i + 1) * sym_size];
            let actual = sink
                .symbols
                .get(&esi)
                .unwrap_or_else(|| panic!("source symbol ESI {esi} missing"));
            assert_eq!(
                actual, expected,
                "bead_id={BEAD_ID} case=encode_source_symbol_matches esi={esi}"
            );
        }

        info!(
            bead_id = BEAD_ID,
            source_count = outcome.source_count,
            repair_count = outcome.repair_count,
            total_written = sink.written_count(),
            "test_pipeline_encode complete"
        );
    }

    // -----------------------------------------------------------------------
    // §3.3 Test 13: Pipeline decode (test_pipeline_decode_async)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_decode_with_extra_symbols() {
        let config = default_config();
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
        let decoder =
            RaptorQPageDecoder::new(config.clone(), default_codec()).expect("decoder build");
        let cx = test_cx();
        let k = 10_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0x5678);

        // Encode.
        let mut sink = VecPageSink::new();
        let outcome = encoder
            .encode_pages(&cx, &data, &mut sink)
            .expect("encode must succeed");

        // Decode from all symbols (K + repair).
        let mut source = VecPageSource::from_sink(&sink);
        let decode_outcome = decoder
            .decode_pages(&cx, &mut source, outcome.source_count)
            .expect("decode must succeed");

        match decode_outcome {
            DecodeOutcome::Success(success) => {
                assert_eq!(
                    success.data, data,
                    "bead_id={BEAD_ID} case=decode_roundtrip_bytes"
                );
                assert!(
                    success.symbols_used >= outcome.source_count,
                    "bead_id={BEAD_ID} case=decode_symbols_used"
                );
                info!(
                    bead_id = BEAD_ID,
                    symbols_used = success.symbols_used,
                    peeled = success.peeled_count,
                    inactivated = success.inactivated_count,
                    "test_pipeline_decode complete"
                );
            }
            DecodeOutcome::Failure(failure) => {
                panic!(
                    "bead_id={BEAD_ID} case=decode_unexpected_failure reason={:?}",
                    failure.reason
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // §3.3 Test 14: Cancel-safety (test_pipeline_cancel_safe)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_cancel_safe_encode() {
        let config = PipelineConfig {
            checkpoint_interval: 2, // checkpoint every 2 symbols
            ..default_config()
        };
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");

        // Create a Cx that is already cancelled.
        let cx = fsqlite_types::cx::Cx::new();
        cx.cancel_with_reason(fsqlite_types::cx::CancelReason::UserInterrupt);

        let k = 10_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0xABCD);
        let mut sink = VecPageSink::new();

        let result = encoder.encode_pages(&cx, &data, &mut sink);
        assert!(
            result.is_err(),
            "bead_id={BEAD_ID} case=cancel_safe_encode_aborts"
        );
        assert!(
            matches!(result.unwrap_err(), FrankenError::Abort),
            "bead_id={BEAD_ID} case=cancel_safe_encode_error_type"
        );
        // Sink should not have been flushed.
        assert!(!sink.flushed, "bead_id={BEAD_ID} case=cancel_safe_no_flush");
    }

    #[test]
    fn test_pipeline_cancel_safe_decode() {
        let config = PipelineConfig {
            checkpoint_interval: 2,
            ..default_config()
        };
        let decoder =
            RaptorQPageDecoder::new(config.clone(), default_codec()).expect("decoder build");

        // Create a Cx that is already cancelled.
        let cx = fsqlite_types::cx::Cx::new();
        cx.cancel_with_reason(fsqlite_types::cx::CancelReason::UserInterrupt);

        // Feed some symbols.
        let mut symbols = BTreeMap::new();
        for esi in 0..10_u32 {
            symbols.insert(esi, vec![0xAA; config.symbol_size as usize]);
        }
        let mut source = VecPageSource::from_map(symbols);

        let result = decoder.decode_pages(&cx, &mut source, 10);
        assert!(
            result.is_err(),
            "bead_id={BEAD_ID} case=cancel_safe_decode_aborts"
        );
        assert!(
            matches!(result.unwrap_err(), FrankenError::Abort),
            "bead_id={BEAD_ID} case=cancel_safe_decode_error_type"
        );
    }

    // -----------------------------------------------------------------------
    // §3.3 Test 15: Backpressure (test_pipeline_backpressure)
    // -----------------------------------------------------------------------

    /// Sink that fails after N writes, simulating a full output buffer.
    struct BackpressureSink {
        limit: u32,
        count: u32,
    }

    impl BackpressureSink {
        fn new(limit: u32) -> Self {
            Self { limit, count: 0 }
        }
    }

    impl PageSymbolSink for BackpressureSink {
        fn write_symbol(&mut self, _esi: u32, _data: &[u8]) -> Result<()> {
            if self.count >= self.limit {
                return Err(FrankenError::Busy);
            }
            self.count += 1;
            Ok(())
        }

        fn flush(&mut self) -> Result<()> {
            Ok(())
        }

        fn written_count(&self) -> u32 {
            self.count
        }
    }

    #[test]
    fn test_pipeline_backpressure_sink_full() {
        let config = default_config();
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
        let cx = test_cx();
        let k = 10_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0xEEFF);

        // Sink that only accepts 3 symbols then returns Busy.
        let mut sink = BackpressureSink::new(3);
        let result = encoder.encode_pages(&cx, &data, &mut sink);

        assert!(
            result.is_err(),
            "bead_id={BEAD_ID} case=backpressure_propagated"
        );
        assert!(
            matches!(result.unwrap_err(), FrankenError::Busy),
            "bead_id={BEAD_ID} case=backpressure_error_type"
        );
        assert_eq!(
            sink.written_count(),
            3,
            "bead_id={BEAD_ID} case=backpressure_partial_write"
        );
    }

    // -----------------------------------------------------------------------
    // Config Validation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validation_zero_symbol_size() {
        let config = PipelineConfig {
            symbol_size: 0,
            ..default_config()
        };
        assert!(
            config.validate().is_err(),
            "bead_id={BEAD_ID} case=config_reject_zero_symbol_size"
        );
    }

    #[test]
    fn test_config_validation_non_power_of_two() {
        let config = PipelineConfig {
            symbol_size: 1000,
            ..default_config()
        };
        assert!(
            config.validate().is_err(),
            "bead_id={BEAD_ID} case=config_reject_non_power_of_two"
        );
    }

    #[test]
    fn test_config_validation_below_min() {
        let config = PipelineConfig {
            symbol_size: 256,
            ..default_config()
        };
        assert!(
            config.validate().is_err(),
            "bead_id={BEAD_ID} case=config_reject_below_min"
        );
    }

    #[test]
    fn test_config_validation_above_max() {
        let config = PipelineConfig {
            symbol_size: 128 * 1024,
            ..default_config()
        };
        assert!(
            config.validate().is_err(),
            "bead_id={BEAD_ID} case=config_reject_above_max"
        );
    }

    #[test]
    fn test_config_validation_zero_max_block_size() {
        let config = PipelineConfig {
            max_block_size: 0,
            ..default_config()
        };
        assert!(
            config.validate().is_err(),
            "bead_id={BEAD_ID} case=config_reject_zero_max_block"
        );
    }

    #[test]
    fn test_config_validation_repair_overhead_below_one() {
        let config = PipelineConfig {
            repair_overhead: 0.5,
            ..default_config()
        };
        assert!(
            config.validate().is_err(),
            "bead_id={BEAD_ID} case=config_reject_repair_overhead_below_one"
        );
    }

    #[test]
    fn test_config_validation_zero_checkpoint_interval() {
        let config = PipelineConfig {
            checkpoint_interval: 0,
            ..default_config()
        };
        assert!(
            config.validate().is_err(),
            "bead_id={BEAD_ID} case=config_reject_zero_checkpoint_interval"
        );
    }

    #[test]
    fn test_config_validation_valid_configs() {
        for symbol_size in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] {
            let config = PipelineConfig::for_page_size(symbol_size);
            assert!(
                config.validate().is_ok(),
                "bead_id={BEAD_ID} case=config_valid symbol_size={symbol_size}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Decode Proof on Failure
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_failure_insufficient_symbols() {
        let config = default_config();
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
        let decoder =
            RaptorQPageDecoder::new(config.clone(), default_codec()).expect("decoder build");
        let cx = test_cx();
        let k = 10_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0xDEAD);

        // Encode.
        let mut sink = VecPageSink::new();
        let outcome = encoder
            .encode_pages(&cx, &data, &mut sink)
            .expect("encode must succeed");

        // Keep only K-3 source symbols (insufficient).
        let mut partial = BTreeMap::new();
        for esi in 0..((k - 3) as u32) {
            if let Some(sym) = sink.symbols.get(&esi) {
                partial.insert(esi, sym.clone());
            }
        }
        let mut source = VecPageSource::from_map(partial);

        let decode_outcome = decoder
            .decode_pages(&cx, &mut source, outcome.source_count)
            .expect("decode call itself should not error");

        match decode_outcome {
            DecodeOutcome::Failure(failure) => {
                assert_eq!(
                    failure.reason,
                    DecodeFailureReason::InsufficientSymbols,
                    "bead_id={BEAD_ID} case=decode_failure_reason"
                );
                assert!(
                    failure.symbols_received < outcome.source_count,
                    "bead_id={BEAD_ID} case=decode_failure_symbol_count"
                );
                assert_eq!(
                    failure.k_required, outcome.source_count,
                    "bead_id={BEAD_ID} case=decode_failure_k_required"
                );
            }
            DecodeOutcome::Success(_) => {
                panic!("bead_id={BEAD_ID} case=decode_should_have_failed");
            }
        }
    }

    // -----------------------------------------------------------------------
    // E2E Round-trip: encode → store → read → decode → verify
    // -----------------------------------------------------------------------

    #[test]
    fn test_e2e_roundtrip_multiple_page_sizes() {
        for &symbol_size in &[512_u32, 1024, 4096] {
            let config = PipelineConfig::for_page_size(symbol_size);
            let encoder =
                RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
            let decoder =
                RaptorQPageDecoder::new(config.clone(), default_codec()).expect("decoder build");
            let cx = test_cx();

            let k = 8_usize;
            let data = deterministic_page_data(k, symbol_size as usize, u64::from(symbol_size));

            // Encode → store.
            let mut sink = VecPageSink::new();
            let outcome = encoder
                .encode_pages(&cx, &data, &mut sink)
                .expect("encode must succeed");

            // Read → decode.
            let mut source = VecPageSource::from_sink(&sink);
            let decode_result = decoder
                .decode_pages(&cx, &mut source, outcome.source_count)
                .expect("decode must succeed");

            match decode_result {
                DecodeOutcome::Success(success) => {
                    assert_eq!(
                        success.data, data,
                        "bead_id={BEAD_ID} case=e2e_roundtrip symbol_size={symbol_size}"
                    );
                }
                DecodeOutcome::Failure(f) => {
                    panic!(
                        "bead_id={BEAD_ID} case=e2e_roundtrip_failure symbol_size={symbol_size} reason={:?}",
                        f.reason
                    );
                }
            }
        }
    }

    #[test]
    fn test_e2e_roundtrip_64_pages() {
        let config = PipelineConfig::for_page_size(4096);
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
        let decoder =
            RaptorQPageDecoder::new(config.clone(), default_codec()).expect("decoder build");
        let cx = test_cx();

        let k = 64_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0xE2E6_4000);

        let mut sink = VecPageSink::new();
        let outcome = encoder
            .encode_pages(&cx, &data, &mut sink)
            .expect("encode must succeed");

        assert_eq!(
            outcome.source_count as usize, k,
            "bead_id={BEAD_ID} case=e2e_64_source_count"
        );

        let mut source = VecPageSource::from_sink(&sink);
        let decode_result = decoder
            .decode_pages(&cx, &mut source, outcome.source_count)
            .expect("decode must succeed");

        match decode_result {
            DecodeOutcome::Success(success) => {
                assert_eq!(
                    success.data, data,
                    "bead_id={BEAD_ID} case=e2e_64_roundtrip_bytes"
                );
                info!(
                    bead_id = BEAD_ID,
                    k,
                    peeled = success.peeled_count,
                    inactivated = success.inactivated_count,
                    "E2E 64-page roundtrip complete"
                );
            }
            DecodeOutcome::Failure(f) => {
                panic!(
                    "bead_id={BEAD_ID} case=e2e_64_failure reason={:?}",
                    f.reason
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // E2E: Retry after failure
    // -----------------------------------------------------------------------

    #[test]
    fn test_e2e_retry_after_failure() {
        let config = default_config();
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
        let decoder =
            RaptorQPageDecoder::new(config.clone(), default_codec()).expect("decoder build");
        let cx = test_cx();
        let k = 10_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0xAE_7121);

        // Encode.
        let mut sink = VecPageSink::new();
        let outcome = encoder
            .encode_pages(&cx, &data, &mut sink)
            .expect("encode must succeed");

        // First attempt: K-2 source symbols only → should fail.
        let mut partial = BTreeMap::new();
        for esi in 0..((k - 2) as u32) {
            if let Some(sym) = sink.symbols.get(&esi) {
                partial.insert(esi, sym.clone());
            }
        }
        let mut source_attempt1 = VecPageSource::from_map(partial.clone());
        let result1 = decoder
            .decode_pages(&cx, &mut source_attempt1, outcome.source_count)
            .expect("decode call should not error");
        assert!(
            matches!(result1, DecodeOutcome::Failure(_)),
            "bead_id={BEAD_ID} case=retry_first_attempt_fails"
        );

        // Second attempt: add all remaining symbols → should succeed.
        let full = sink.symbols.clone();
        let mut source_attempt2 = VecPageSource::from_map(full);
        let result2 = decoder
            .decode_pages(&cx, &mut source_attempt2, outcome.source_count)
            .expect("decode call should not error");
        match result2 {
            DecodeOutcome::Success(success) => {
                assert_eq!(
                    success.data, data,
                    "bead_id={BEAD_ID} case=retry_second_attempt_succeeds"
                );
            }
            DecodeOutcome::Failure(f) => {
                panic!(
                    "bead_id={BEAD_ID} case=retry_second_should_succeed reason={:?}",
                    f.reason
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Decode with exact K symbols (fragile recovery)
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_source_only_exact_k() {
        let config = default_config();
        let encoder =
            RaptorQPageEncoder::new(config.clone(), default_codec()).expect("encoder build");
        let decoder =
            RaptorQPageDecoder::new(config.clone(), default_codec()).expect("decoder build");
        let cx = test_cx();
        let k = 8_usize;
        let data = deterministic_page_data(k, config.symbol_size as usize, 0xE4AC7);

        let mut sink = VecPageSink::new();
        let outcome = encoder
            .encode_pages(&cx, &data, &mut sink)
            .expect("encode must succeed");

        // Keep only K source symbols (no repair).
        let mut source_only = BTreeMap::new();
        for esi in 0..(k as u32) {
            if let Some(sym) = sink.symbols.get(&esi) {
                source_only.insert(esi, sym.clone());
            }
        }

        let mut source = VecPageSource::from_map(source_only);
        let decode_result = decoder
            .decode_pages(&cx, &mut source, outcome.source_count)
            .expect("decode must not error");

        match decode_result {
            DecodeOutcome::Success(success) => {
                assert_eq!(
                    success.data, data,
                    "bead_id={BEAD_ID} case=exact_k_roundtrip"
                );
                assert_eq!(
                    success.symbols_used, k as u32,
                    "bead_id={BEAD_ID} case=exact_k_symbols_used"
                );
            }
            DecodeOutcome::Failure(f) => {
                panic!(
                    "bead_id={BEAD_ID} case=exact_k_should_succeed reason={:?}",
                    f.reason
                );
            }
        }
    }
}
