//! Report rendering: JSONL run records -> human-readable markdown summary.
//!
//! Bead: bd-1w6k.6.3
//!
//! Reads JSONL files containing [`RunRecordV1`] or [`BenchmarkSummary`]
//! records and renders them into a self-contained Markdown document with
//! comparison tables and key statistics.
//!
//! ## Usage
//!
//! ```text
//! let records = parse_run_records_jsonl(&jsonl_content)?;
//! let markdown = render_run_records_markdown(&records);
//! std::fs::write("reports/summary.md", markdown)?;
//! ```

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

use crate::benchmark::BenchmarkSummary;
use crate::report::RunRecordV1;

// ── Parsing ────────────────────────────────────────────────────────────

/// Parse a JSONL string into a list of [`RunRecordV1`] records.
///
/// Blank lines and lines that fail to parse are silently skipped.
#[must_use]
pub fn parse_run_records_jsonl(jsonl: &str) -> Vec<RunRecordV1> {
    jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// Parse a JSONL string into a list of [`BenchmarkSummary`] records.
///
/// Blank lines and lines that fail to parse are silently skipped.
#[must_use]
pub fn parse_benchmark_summaries_jsonl(jsonl: &str) -> Vec<BenchmarkSummary> {
    jsonl
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

// ── Markdown rendering from RunRecordV1 ────────────────────────────────

/// Render a list of [`RunRecordV1`] records into a Markdown summary.
///
/// Groups records by `(fixture_id, workload, concurrency)` and produces
/// a comparison table showing each engine's performance side by side.
#[must_use]
pub fn render_run_records_markdown(records: &[RunRecordV1]) -> String {
    type GroupKey = (String, String, u16);
    let mut out = String::with_capacity(4096);

    let _ = writeln!(out, "# E2E Run Report\n");

    if records.is_empty() {
        let _ = writeln!(out, "_No records to report._");
        return out;
    }

    // Capture methodology from the first record.
    let meth = &records[0].methodology;
    let _ = writeln!(out, "## Methodology\n");
    let _ = writeln!(out, "- **Version:** {}", meth.version);
    let _ = writeln!(out, "- **Warmup iterations:** {}", meth.warmup_iterations);
    let _ = writeln!(
        out,
        "- **Measurement iterations:** ≥ {}",
        meth.min_measurement_iterations
    );
    let _ = writeln!(
        out,
        "- **Measurement time:** ≥ {} s",
        meth.measurement_time_secs
    );
    let _ = writeln!(out, "- **Primary statistic:** {}", meth.primary_statistic);
    let _ = writeln!(out, "- **Tail statistic:** {}", meth.tail_statistic);
    let _ = writeln!(
        out,
        "- **Fresh DB per iteration:** {}",
        meth.fresh_db_per_iteration
    );
    let _ = writeln!(
        out,
        "- **Identical PRAGMAs enforced:** {}\n",
        meth.identical_pragmas_enforced
    );

    // Group by (fixture_id, workload, concurrency).
    let mut groups: BTreeMap<GroupKey, Vec<&RunRecordV1>> = BTreeMap::new();
    for record in records {
        let key = (
            record.fixture_id.clone(),
            record.workload.clone(),
            record.concurrency,
        );
        groups.entry(key).or_default().push(record);
    }

    let _ = writeln!(out, "## Results\n");

    for ((fixture_id, workload, concurrency), group) in &groups {
        let _ = writeln!(out, "### {fixture_id} / {workload} (c={concurrency})\n");

        let _ = writeln!(
            out,
            "| Engine | Wall (ms) | Ops | Ops/sec | Retries | Aborts | Integrity | Error |"
        );
        let _ = writeln!(
            out,
            "|--------|-----------|-----|---------|---------|--------|-----------|-------|"
        );

        for record in group {
            let r = &record.report;
            let integrity = r.correctness.integrity_check_ok.map_or_else(
                || "-".to_owned(),
                |ok| {
                    if ok {
                        "ok".to_owned()
                    } else {
                        "FAIL".to_owned()
                    }
                },
            );
            let error = r.error.as_deref().unwrap_or("-");
            let error_display = if error.len() > 40 {
                format!("{}...", &error[..37])
            } else {
                error.to_owned()
            };

            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.1} | {} | {} | {} | {} |",
                record.engine.name,
                r.wall_time_ms,
                r.ops_total,
                r.ops_per_sec,
                r.retries,
                r.aborts,
                integrity,
                error_display,
            );
        }

        let _ = writeln!(out);
    }

    out
}

// ── Markdown rendering from BenchmarkSummary ───────────────────────────

/// Render a list of [`BenchmarkSummary`] records into a Markdown summary.
///
/// Each summary gets its own section with latency and throughput tables.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn render_benchmark_summaries_markdown(summaries: &[BenchmarkSummary]) -> String {
    let mut out = String::with_capacity(4096);

    let _ = writeln!(out, "# Benchmark Report\n");

    if summaries.is_empty() {
        let _ = writeln!(out, "_No benchmarks to report._");
        return out;
    }

    // Methodology from first summary.
    let meth = &summaries[0].methodology;
    let _ = writeln!(out, "## Methodology\n");
    let _ = writeln!(out, "- **Version:** {}", meth.version);
    let _ = writeln!(out, "- **Primary statistic:** {}", meth.primary_statistic);
    let _ = writeln!(out, "- **Tail statistic:** {}\n", meth.tail_statistic);

    // Environment from first summary.
    let env = &summaries[0].environment;
    let _ = writeln!(out, "## Environment\n");
    let _ = writeln!(out, "- **OS:** {}", env.os);
    let _ = writeln!(out, "- **Arch:** {}", env.arch);
    let _ = writeln!(out, "- **CPUs:** {}", env.cpu_count);
    if let Some(ref model) = env.cpu_model {
        let _ = writeln!(out, "- **CPU model:** {model}");
    }
    if let Some(ram) = env.ram_bytes {
        let _ = writeln!(out, "- **RAM:** {:.1} GiB", ram as f64 / 1_073_741_824.0);
    }
    let _ = writeln!(out, "- **rustc:** {}", env.rustc_version);
    let _ = writeln!(out, "- **Profile:** {}\n", env.cargo_profile);

    // Summary table.
    let _ = writeln!(out, "## Summary\n");
    let _ = writeln!(
        out,
        "| Benchmark | Engine | Iters | Median (ms) | p95 (ms) | p99 (ms) | Stddev (ms) | Median Ops/s | Peak Ops/s |"
    );
    let _ = writeln!(
        out,
        "|-----------|--------|-------|-------------|----------|----------|-------------|--------------|------------|"
    );

    for s in summaries {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} |",
            s.benchmark_id,
            s.engine,
            s.measurement_count,
            s.latency.median_ms,
            s.latency.p95_ms,
            s.latency.p99_ms,
            s.latency.stddev_ms,
            s.throughput.median_ops_per_sec,
            s.throughput.peak_ops_per_sec,
        );
    }

    let _ = writeln!(out);

    // Detailed per-benchmark sections.
    for s in summaries {
        let _ = writeln!(out, "### {}\n", s.benchmark_id);
        let _ = writeln!(out, "- **Fixture:** {}", s.fixture_id);
        let _ = writeln!(out, "- **Workload:** {}", s.workload);
        let _ = writeln!(out, "- **Concurrency:** {}", s.concurrency);
        let _ = writeln!(out, "- **Warmup iterations:** {}", s.warmup_count);
        let _ = writeln!(out, "- **Measurement iterations:** {}", s.measurement_count);
        let _ = writeln!(
            out,
            "- **Total measurement time:** {} ms\n",
            s.total_measurement_ms
        );

        let _ = writeln!(out, "**Latency (ms):**\n");
        let _ = writeln!(out, "| Min | Max | Mean | Median | p95 | p99 | Stddev |");
        let _ = writeln!(out, "|-----|-----|------|--------|-----|-----|--------|");
        let _ = writeln!(
            out,
            "| {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
            s.latency.min_ms,
            s.latency.max_ms,
            s.latency.mean_ms,
            s.latency.median_ms,
            s.latency.p95_ms,
            s.latency.p99_ms,
            s.latency.stddev_ms,
        );

        let _ = writeln!(out, "**Throughput (ops/sec):**\n");
        let _ = writeln!(out, "| Mean | Median | Peak |");
        let _ = writeln!(out, "|------|--------|------|");
        let _ = writeln!(
            out,
            "| {:.0} | {:.0} | {:.0} |\n",
            s.throughput.mean_ops_per_sec,
            s.throughput.median_ops_per_sec,
            s.throughput.peak_ops_per_sec,
        );

        // Per-iteration table (truncated if large).
        let max_rows = 20;
        let show_count = s.iterations.len().min(max_rows);
        if !s.iterations.is_empty() {
            let _ = writeln!(
                out,
                "<details>\n<summary>Iteration details ({} iterations, showing first {show_count})</summary>\n",
                s.iterations.len()
            );
            let _ = writeln!(
                out,
                "| # | Wall (ms) | Ops/sec | Ops | Retries | Aborts | Error |"
            );
            let _ = writeln!(
                out,
                "|---|-----------|---------|-----|---------|--------|-------|"
            );

            for iter in s.iterations.iter().take(max_rows) {
                let error = iter.error.as_deref().unwrap_or("-");
                let _ = writeln!(
                    out,
                    "| {} | {} | {:.1} | {} | {} | {} | {} |",
                    iter.iteration,
                    iter.wall_time_ms,
                    iter.ops_per_sec,
                    iter.ops_total,
                    iter.retries,
                    iter.aborts,
                    error,
                );
            }

            if s.iterations.len() > max_rows {
                let _ = writeln!(
                    out,
                    "\n_... {} more iterations not shown._",
                    s.iterations.len() - max_rows
                );
            }

            let _ = writeln!(out, "\n</details>\n");
        }
    }

    out
}

// ── File-based convenience functions ───────────────────────────────────

/// Read a JSONL file and render run records to Markdown.
///
/// # Errors
///
/// Returns an I/O error if the file cannot be read.
pub fn render_run_records_from_file(path: &Path) -> std::io::Result<String> {
    let content = std::fs::read_to_string(path)?;
    let records = parse_run_records_jsonl(&content);
    Ok(render_run_records_markdown(&records))
}

/// Read a JSONL file and render benchmark summaries to Markdown.
///
/// # Errors
///
/// Returns an I/O error if the file cannot be read.
pub fn render_benchmark_summaries_from_file(path: &Path) -> std::io::Result<String> {
    let content = std::fs::read_to_string(path)?;
    let summaries = parse_benchmark_summaries_jsonl(&content);
    Ok(render_benchmark_summaries_markdown(&summaries))
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{BenchmarkConfig, BenchmarkMeta, run_benchmark};
    use crate::methodology::EnvironmentMeta;
    use crate::report::{CorrectnessReport, EngineInfo, EngineRunReport, RunRecordV1Args};

    fn dummy_engine_report(wall_ms: u64, ops: u64) -> EngineRunReport {
        #[allow(clippy::cast_precision_loss)]
        let ops_per_sec = if wall_ms > 0 {
            ops as f64 / (wall_ms as f64 / 1000.0)
        } else {
            0.0
        };
        EngineRunReport {
            wall_time_ms: wall_ms,
            ops_total: ops,
            ops_per_sec,
            retries: 2,
            aborts: 0,
            correctness: CorrectnessReport {
                raw_sha256_match: None,
                dump_match: None,
                canonical_sha256_match: None,
                integrity_check_ok: Some(true),
                raw_sha256: None,
                canonical_sha256: None,
                logical_sha256: None,
                notes: None,
            },
            latency_ms: None,
            error: None,
        }
    }

    fn make_run_record(engine: &str, fixture: &str, workload: &str, c: u16) -> RunRecordV1 {
        RunRecordV1::new(RunRecordV1Args {
            recorded_unix_ms: 1_700_000_000_000,
            environment: EnvironmentMeta::capture("test"),
            engine: EngineInfo {
                name: engine.to_owned(),
                sqlite_version: if engine == "sqlite3" {
                    Some("3.45.0".to_owned())
                } else {
                    None
                },
                fsqlite_git: None,
            },
            fixture_id: fixture.to_owned(),
            golden_path: None,
            golden_sha256: None,
            workload: workload.to_owned(),
            concurrency: c,
            ops_count: 100,
            report: dummy_engine_report(500, 100),
        })
    }

    #[test]
    fn parse_run_records_from_jsonl() {
        let r1 = make_run_record("sqlite3", "db-a", "inserts", 1);
        let r2 = make_run_record("fsqlite", "db-a", "inserts", 1);

        let line1 = serde_json::to_string(&r1).unwrap();
        let line2 = serde_json::to_string(&r2).unwrap();
        let jsonl = format!("{line1}\n{line2}\n\n");

        let parsed = parse_run_records_jsonl(&jsonl);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].engine.name, "sqlite3");
        assert_eq!(parsed[1].engine.name, "fsqlite");
    }

    #[test]
    fn render_run_records_empty() {
        let md = render_run_records_markdown(&[]);
        assert!(md.contains("No records to report"));
    }

    #[test]
    fn render_run_records_produces_table() {
        let records = vec![
            make_run_record("sqlite3", "db-a", "inserts", 4),
            make_run_record("fsqlite", "db-a", "inserts", 4),
        ];

        let md = render_run_records_markdown(&records);
        assert!(md.contains("# E2E Run Report"));
        assert!(md.contains("## Methodology"));
        assert!(md.contains("## Results"));
        assert!(md.contains("db-a / inserts (c=4)"));
        assert!(md.contains("| sqlite3 |"));
        assert!(md.contains("| fsqlite |"));
        assert!(md.contains("| ok |"));
    }

    #[test]
    fn render_run_records_groups_by_fixture_workload_concurrency() {
        let records = vec![
            make_run_record("sqlite3", "db-a", "inserts", 1),
            make_run_record("fsqlite", "db-a", "inserts", 1),
            make_run_record("sqlite3", "db-b", "updates", 4),
            make_run_record("fsqlite", "db-b", "updates", 4),
        ];

        let md = render_run_records_markdown(&records);
        assert!(md.contains("db-a / inserts (c=1)"));
        assert!(md.contains("db-b / updates (c=4)"));
    }

    #[test]
    fn render_benchmark_summaries_empty() {
        let md = render_benchmark_summaries_markdown(&[]);
        assert!(md.contains("No benchmarks to report"));
    }

    #[test]
    fn render_benchmark_summaries_produces_tables() {
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            min_iterations: 3,
            measurement_time_secs: 0,
        };
        let meta = BenchmarkMeta {
            engine: "sqlite3".to_owned(),
            workload: "inserts".to_owned(),
            fixture_id: "db-a".to_owned(),
            concurrency: 4,
            cargo_profile: "test".to_owned(),
        };

        let summary = run_benchmark(&config, &meta, |_| {
            Ok::<_, String>(dummy_engine_report(100, 1000))
        });

        let md = render_benchmark_summaries_markdown(&[summary]);
        assert!(md.contains("# Benchmark Report"));
        assert!(md.contains("## Methodology"));
        assert!(md.contains("## Environment"));
        assert!(md.contains("## Summary"));
        assert!(md.contains("sqlite3:inserts:db-a:c4"));
        assert!(md.contains("Latency (ms)"));
        assert!(md.contains("Throughput (ops/sec)"));
        assert!(md.contains("Iteration details"));
    }

    #[test]
    fn parse_benchmark_summaries_from_jsonl() {
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            min_iterations: 2,
            measurement_time_secs: 0,
        };
        let meta = BenchmarkMeta {
            engine: "test".to_owned(),
            workload: "w".to_owned(),
            fixture_id: "f".to_owned(),
            concurrency: 1,
            cargo_profile: "test".to_owned(),
        };

        let summary = run_benchmark(&config, &meta, |_| {
            Ok::<_, String>(dummy_engine_report(50, 500))
        });

        let line = summary.to_jsonl().unwrap();
        let jsonl = format!("{line}\n");

        let parsed = parse_benchmark_summaries_jsonl(&jsonl);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].benchmark_id, "test:w:f:c1");
    }

    #[test]
    fn render_benchmark_summary_truncates_long_iteration_list() {
        let config = BenchmarkConfig {
            warmup_iterations: 0,
            min_iterations: 30,
            measurement_time_secs: 0,
        };
        let meta = BenchmarkMeta {
            engine: "test".to_owned(),
            workload: "w".to_owned(),
            fixture_id: "f".to_owned(),
            concurrency: 1,
            cargo_profile: "test".to_owned(),
        };

        let summary = run_benchmark(&config, &meta, |_| {
            Ok::<_, String>(dummy_engine_report(10, 100))
        });

        let md = render_benchmark_summaries_markdown(&[summary]);
        assert!(md.contains("30 iterations, showing first 20"));
        assert!(md.contains("10 more iterations not shown"));
    }

    #[test]
    fn error_in_run_record_shown_in_markdown() {
        let mut record = make_run_record("sqlite3", "db-err", "inserts", 1);
        record.report.error = Some("database locked".to_owned());
        record.report.correctness.integrity_check_ok = Some(false);

        let md = render_run_records_markdown(&[record]);
        assert!(md.contains("database locked"));
        assert!(md.contains("FAIL"));
    }

    #[test]
    fn long_error_message_truncated() {
        let mut record = make_run_record("sqlite3", "db-err", "inserts", 1);
        record.report.error = Some("a".repeat(100));

        let md = render_run_records_markdown(&[record]);
        // Error should be truncated to ~40 chars
        assert!(md.contains("..."));
    }
}
