# RaptorQ Permeation Map Audit Checklist

This checklist is the living audit artifact for the §0.4 "RaptorQ Everywhere" doctrine and the §3.5.7 permeation map requirement.

Rule: every subsystem that persists or synchronizes bytes must define an ECS representation and a decode/repair path. The only allowed exception is compatibility-mode SQLite WAL files.

| subsystem_key | bytes_format | ecs_object_type | repair_mechanism | replication_path | status | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| object_identity | canonical object header bytes plus payload hash bytes | ObjectId | decode canonical bytes and re-derive ObjectId, hash mismatch is corruption | Object IDs are transported with symbol metadata, not file offsets | verified | test_new_durable_structure_requires_ecs |
| symbol_records | FSEC envelope bytes, OTI bytes, symbol payload bytes | SymbolRecord | collect K' or K'+2 symbols, decode, verify frame_xxh3 and optional auth tag | symbol-native transport, no raw file copy | verified | test_replication_uses_symbols_not_files |
| commit_capsules | canonical commit capsule bytes keyed by ObjectId | CommitCapsule, SymbolRecord | RaptorQ decode from symbol set, integrity verification before apply | symbol-native stream keyed by ObjectId | verified | test_commit_capsules_are_ecs_objects |
| commit_markers | commit marker chain record bytes | CommitMarker, SymbolRecord | decode marker object, validate commit chain links and integrity hash | marker symbols replicated by ObjectId, not as mutable file snapshots | verified | test_commit_markers_are_ecs_objects |
| commit_proofs | commit proof object bytes for durable commit evidence | CommitProof, SymbolRecord | decode proof object, fail closed if decode/integrity fails | proof symbols replicated as ECS objects | verified | test_commit_proofs_are_ecs_objects |
| root_manifest | bootstrap manifest bytes linking logical db root state | RootManifest, SymbolRecord | decode manifest object from symbols, then verify referenced roots | manifest symbols replicated before root switch | verified | test_root_manifest_is_ecs |
| checkpoints | checkpoint index and materialization metadata bytes | ManifestSegment, PageVersionIndexSegment, SymbolRecord | decode checkpoint index segments, then rebuild view from decoded pointers | checkpoint data ships as segment symbols | specified | test_checkpoints_are_ecs_objects |
| index_segments_page_versions | sorted page->version pointer bytes plus bloom bytes | PageVersionIndexSegment | decode segment object, then use bloom+lookup on decoded bytes | segment symbols replicated and cache-rebuilt from ObjectId | verified | test_index_segments_are_ecs_objects |
| index_segments_object_locator | sorted object->symbol-log-offset mapping bytes | ObjectLocatorSegment | decode locator segment, rebuild from symbol-log scan on damage | locator segment symbols replicated by ObjectId | verified | test_index_segments_are_ecs_objects |
| index_segments_manifest | commit-range->segment mapping bytes | ManifestSegment | decode manifest segment and verify range lookup invariants | manifest segment symbols replicated by ObjectId | verified | test_index_segments_are_ecs_objects |
| patch_chains_history | page history/patch pointer bytes and intent history bytes | PageHistory, VersionPointer, SymbolRecord | decode patch objects and re-materialize page history; reject on decode failure | patch/history symbols replicated as ECS objects | verified | test_patch_chains_are_ecs_objects |
| replication_symbol_transport | transport payload is SymbolRecord bytes only | SymbolRecord | receiver decodes symbol sets; missing symbols trigger decode path, never unwind path | symbol packets over replication stream, no raw db/wal file copy | verified | test_replication_uses_symbols_not_files |
| repair_decode_pipeline | corruption and loss handling workflow bytes | SymbolRecord, DecodeProof | always attempt decode+verification first; return typed error on failure, never abort | repaired objects re-enter replication by symbol stream | verified | test_repair_uses_decode_not_panic |
| compat_mode_wal_exemption | sqlite compatibility wal frame bytes in `*.db-wal` files | EXEMPT | sqlite native wal checksum+replay path in compatibility mode | local file I/O only, explicitly excluded from ECS replication | exempt | §0.4 explicit exception and README compatibility mode notes |

## Maintenance Rules

1. When adding a new durable or replicated structure, add a row before merging code.
2. Every non-exempt row must specify decode-based repair and symbol-based replication.
3. Keep the evidence column pointing to a concrete harness test.
4. Update `test_new_durable_structure_requires_ecs` if a new durable type is introduced in `fsqlite-types`.
