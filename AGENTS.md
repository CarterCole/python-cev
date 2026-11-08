# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A Python binding for [cev](https://github.com/CarterCole/rust-cev). The
engine, the decision log and the online learner are Rust crates in that
repository (`cev-core`, `cev-model`, `cev-runtime`) and are published on their
own. This repository holds only the glue: a PyO3 module and a typed Python
package. Nothing from the engine is copied here. A change to prompts, model
code, storage or learning belongs in rust-cev. The Node.js sibling,
[node-cev](https://github.com/neopunisher/node-cev), has the same shape; keep
the two APIs in step.

## Layout

- `src/lib.rs` — the native module `cev._native`: wraps `cev_runtime::Cev`
  behind a class whose methods take and return JSON text, release the GIL, and
  bound forward passes with `concurrency` permits. It also sizes rayon's global
  pool (`threads`).
- `python/cev/__init__.py` — the public API: `Cev`, `CevError`, `info()`.
- `python/cev/_types.py` — `TypedDict`s for the wire format (cev's REST bodies).
- `python/cev/_native.pyi` — stub for the compiled module.
- `tests/test_cev.py` — pytest, against the mock backend.
- `scripts/bench.py` — latency and throughput for one configuration.
- `Cargo.toml` / `pyproject.toml` — the crate and the maturin build.

## Commands

```bash
uv sync                              # build the module (release) into .venv; needs ../rust-cev
uv run pytest                        # no weights needed
uv run maturin develop --release     # rebuild after changing src/lib.rs
CEV_TEST_MODEL=Qwen/Qwen3-0.6B uv run pytest   # also run the real-model test
uv run python scripts/bench.py --device cpu --threads 8
```

Run `uv run maturin develop --release && uv run pytest` before calling a
change done.

## Rules

- No runtime dependencies beyond the standard library.
- Python >= 3.11; the module is built against the stable ABI (`abi3-py311`).
- Requests and responses keep cev's wire format so they match the REST API and
  the exported training rows. A new `Cev.load` option needs a field in
  `Options` (`src/lib.rs`), a keyword in `load` (and `mock` if it applies to
  the runtime), a docstring line and a README options-table row.
- Every native method that runs the model or touches SQLite goes through
  `Cev::run`, which releases the GIL. Forward passes go through `Shared::pass`.
- Native errors are `RuntimeError("<code>: <message>")`; `__init__.py` turns
  the code into `CevError.code`. Keep the two lists of codes in step.
- Tests must not need weights or the network: use `Cev.mock()`. Tests that
  need a model are skipped unless `CEV_TEST_MODEL` is set.
- Benchmark numbers in the README come from `scripts/bench.py` on the machine
  named there. Re-measure rather than extrapolate when changing them.
- User-visible changes get a `CHANGELOG.md` entry.

## Releasing

Not automated yet: there are no wheels on PyPI, so the package is built from
source. Publishing needs the cev crates on crates.io first (then drop the
`path` keys in `Cargo.toml`) and a CI job that builds wheels per platform
(`maturin build --release`; one abi3 wheel per platform covers every Python).
The PyPI names `cev` and `pycev` are taken; this distribution is `cev-py`.
