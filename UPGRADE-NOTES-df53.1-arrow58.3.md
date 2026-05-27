# Upgrade notes — DataFusion 50.1→53.1 / arrow 56.2→58.3

InfluxData-internal notes for the `upgrade-df-ver5310-a` branch. Records decisions and
behavioral deltas that are **not** obvious from the code or commit messages. Keep this file
updated for any further changes; **delete it before any PR to apache** (influx-internal scratch).

Versions: DataFusion **53.1.0**, arrow/parquet **58.3.0** (skip-pinned ahead of 53.1's declared
58.0, by request), sqlparser 0.61, object_store 0.13.x, MSRV 1.88, edition 2024.

---

## 1. `to_timestamp` execution-timezone behavior — UPSTREAM KEPT (revert deliberately skipped)

**Decision (2026-05-27, Paul Dix): keep upstream 53.1 behavior. Do NOT re-apply the revert.**
Documented here because it may matter later.

### What this is
The prior 52.5 patch set carried influx commit `bb43b6890`,
*Revert "Respect execution timezone in to_timestamp and related functions (#19078)"*.
Upstream #19078 (landed 2026-01-05, in 53.1.0) changed the `to_timestamp*()` family so that
**naïve (timezone-free) timestamp strings are interpreted in the configured
`datafusion.execution.time_zone`** (UTC only when that option is `None`), and results carry the
execution timezone. Before #19078, naïve strings were always interpreted as **UTC**.

The 52.5 revert restored the always-UTC behavior. During the 53.1 upgrade the revert was **skipped**
(it conflicts in ~14 regions against 53.1's reworked `to_timestamp.rs` / `common.rs` / `mod.rs` /
`to_date.rs` / `to_unixtime.rs` / `macros.rs`, and 53.1 also added Float16/Float32/Decimal numeric
input support via #19663 that must not be lost).

### Current state on this branch
- Runs **upstream 53.1 behavior**: naïve `to_timestamp('...')` is interpreted in
  `datafusion.execution.time_zone`.
- `datafusion/sqllogictest/test_files/to_timestamp_timezone.slt` is present and **passing**.
- The upstream implementation bug in #19078 (#20223, scalar-float broadcast) was fixed by #20224
  on 2026-02-12, which is in 53.1 — so there is no outstanding correctness bug to work around.

### Why this is safe to keep (and the risk if assumptions change)
This is a no-op for InfluxDB **as long as IOx does not set a non-UTC
`datafusion.execution.time_zone`**. If any future IOx code path sets a non-UTC execution timezone,
this upstream behavior will **silently shift parsed naïve timestamps** away from UTC — a correctness
change for InfluxDB queries. **If that ever happens, re-derive the revert against 53.1** (force
naïve input → UTC while preserving 53.1's numeric-type support); do not replay the 52.5-era diff.
Touch points if re-deriving: `to_timestamp.rs`, `common.rs`, `mod.rs`, `to_date.rs`,
`to_unixtime.rs`, `macros.rs`; reconcile `to_timestamp_timezone.slt` and
`datetime/timestamps.slt`.

---

## 2. `encrypted_parquet.slt` — NOT an arrow 58.3 regression (false alarm in the original handoff)

**Resolution (2026-05-27): no code change, no test edit, no version pin-back needed.**

The original handoff flagged this test as failing due to a "genuine upstream change between parquet
58.0 and 58.3." That diagnosis was **incorrect**. Root cause: the test was run **without** the
`parquet_encryption` feature.

### What actually happens
Parquet encryption (read *and* write) is gated behind `#[cfg(feature = "parquet_encryption")]`.
With the feature **off**, the `format.crypto.*` table options are silently ignored, so the
"encrypted" files are written as **plaintext** (footer magic `PAR1`, plaintext schema/column names
on disk). The final step of the test then reads those plaintext files without keys and **succeeds**,
so the `query error … decryption properties were not provided` assertion fails with
"expected to fail, but actually succeed."

This is independent of the arrow/parquet version — the feature gate behaves identically under 58.0
and 58.3. (Verified by reasoning about the `cfg` gate; the on-disk files confirm plaintext output
when the feature is off.)

### How to run it correctly
```bash
cargo test -p datafusion-sqllogictest --test sqllogictests --features parquet_encryption -- encrypted_parquet
```
With the feature **on**: files are written with an encrypted footer (magic `PARE`, no plaintext
column names on disk), the keyless read errors as expected, and the test **passes**. Verified on
this branch under arrow/parquet 58.3.

**Operational note:** the bare `cargo test -p datafusion-sqllogictest --test sqllogictests` command
fails on `encrypted_parquet.slt` on *any* build without the feature (including pristine 53.1 base).
Always pass `--features parquet_encryption` when running the slt suite, or run the full
`cargo test --workspace` (where feature unification across workspace members enables it).

---

## 3. `nth_value` / `first_value` per-kind signatures — DO NOT UNDO

The "improve OneOf signature diagnostics" commit backports apache #21032 (not in 53.1). It gives
`nth_value`/`first_value` **per-kind** signatures (`Signature::any(1)` / `Signature::any(2)`) rather
than 53.1's `one_of([Nullary, Any(1), Any(2)])`. Reverting to `one_of` re-breaks 5 `errors.slt`
cases (e.g. `one_of` wrongly lets `first_value(c5, 2)` succeed). **Keep per-kind.**

---

## 4. Pre-existing upstream test (not our regression)

`datafusion-expr` → `expr_rewriter::order_by::test::rewrite_sort_cols_by_agg_alias` is flagged in the
handoff as failing on the pristine 53.1.0 tag (feature/env-dependent), unrelated to our patches or
the arrow bump. It did **not** surface in the full `cargo test --workspace` run on this branch
(2026-05-27, exit 0, 0 failures across all suites). Leave it; flag upstream if it recurs.

---

## 5. Verification status (2026-05-27)

- `cargo test --workspace` — **PASS** (exit 0, 0 failures across all suites).
- `encrypted_parquet.slt` with `--features parquet_encryption` — **PASS**.
- `cargo clippy --workspace --all-targets -- -D warnings` — **PASS** (exit 0) after the fix below.
- **Fixed:** 8 `object_store::path::Path::child` deprecation warnings (object_store 0.13 bump)
  that would fail CI's `-D warnings`. Replaced with the maintainer-sanctioned `.join()` /
  `.clone().join()` (exact behavioral equivalent — single percent-encoded segment; `join` takes
  `self` by value so a `.clone()` is added where the receiver was borrowed). Sites:
  `datasource/src/write/demux.rs` (3), `datasource/src/url.rs` (1 line, 2 calls),
  `core/tests/sql/path_partition.rs` (1), `core/src/datasource/physical_plan/parquet.rs` (1).
  Affected tests re-run and green (datasource lib, `test_prefix_path`, `path_partition`,
  `parquet_exec_with_error`).

## 6. Finishing checklist
- [x] §1 to_timestamp revert — resolved: keep upstream, documented above.
- [x] §2 encrypted_parquet — resolved: false alarm, run with `--features parquet_encryption`.
- [x] Full `cargo test --workspace` — green.
- [x] `cargo clippy --workspace --all-targets -- -D warnings` clean (fixed 8 `Path::child` deprecations).
- [ ] Delete `HANDOFF-df-upgrade-ver5310.md` and this `UPGRADE-NOTES-*.md` before any PR to apache.
