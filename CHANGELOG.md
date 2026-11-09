# Changelog

## 0.2.0 (2026-11-08)

Takes cev 0.2.1, now from crates.io: building no longer needs a rust-cev
checkout next to this repository. First version on PyPI, as `cev-py`, with
wheels for macOS arm64 and Linux x86_64/aarch64.

### Added
- `Cev.load(model)` runs the open decision models natively: Bespoke Nimble,
  Kev, d3, d1 and d1-omni, plus Qwen3.5 checkpoints. Short names work
  (`"kev:0.8b"`, `"d1-omni"`, `"nimble-v3"`, ...); the family is detected from
  the repository's files.
- `images=` and `audio=` on `decide()`, and in `examples()`, as paths, bytes
  or base64, for models that take them.
- `cev.model["family"]` and `cev.model["modalities"]`.
- Namespaces: `namespace=` on `decide()` and in `examples()` keeps feedback and
  adapters apart. `tasks(namespace)`, `reset_task(task, namespace)`,
  `decisions(namespace=)` and `export(namespace=)` read one namespace.
- `Cev.mock(media=True)`, which accepts images and audio.

### Changed
- `debias` defaults to `"auto"`: `"full"` for zero-shot code readouts as
  before, `"none"` for the trained decision models.

### Fixed
- The macOS module loads when built with Xcode 27: release builds are no
  longer stripped, which made dyld reject the library ("mis-aligned LINKEDIT
  string pool").

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
