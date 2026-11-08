# Changelog

## 0.1.0 (2026-10-03)

First version: cev in-process for Python.

### Added
- `Cev.load()`: a Qwen3 checkpoint from a directory or the Hugging Face cache,
  run by a native module over the cev Rust crates. Metal and Apple's Accelerate
  BLAS on macOS, portable multi-threaded kernels elsewhere, CUDA as a build
  feature. Every call releases the GIL.
- `decide()`, `feedback()`, `examples()`, `tasks()`, `reset_task()`: typed
  decisions, labels, and the online adapters that learn from them.
- `decisions()`, `decision()`, `export()`, `export_ndjson()`, `stats()`: the
  decision log, in memory or in a SQLite file (`db`).
- `Cev.mock()`: the whole API without weights, for tests.
- `threads` and `concurrency` options, `info()`, `TypedDict`s for the wire
  format, and `scripts/bench.py`.
