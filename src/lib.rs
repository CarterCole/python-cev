//! Python bindings for cev: the engine and the decision runtime behind one
//! class whose methods take and return the REST bodies as JSON text. The
//! package in `../python/cev` adds the Python API on top.
//!
//! Every method releases the GIL while it works, so Python threads can call
//! in parallel. Forward passes are bounded by `concurrency` permits and run on
//! rayon's pool (`threads`), which is shared by the whole process.

use cev_core::{Backend, Debias, ExamplesRequest, FeedbackRequest, MockBackend, SystemOneRequest};
use cev_model::{Engine, EngineOptions};
use cev_runtime::{Cev as Runtime, CevError, CevResult, ExportFilter, LearnConfig, RuntimeConfig, Store};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};

/// `Cev.load` / `Cev.mock` options, as sent by the package.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct Options {
    mock: bool,
    model: Option<String>,
    name: Option<String>,
    device: Option<String>,
    dtype: Option<String>,
    max_context: Option<usize>,
    prefix_cache_tokens: Option<usize>,
    db: Option<String>,
    online_learning: Option<bool>,
    store_features: Option<bool>,
    min_examples: Option<u64>,
    debias: Option<Debias>,
    max_permutations: Option<usize>,
    threads: Option<usize>,
    concurrency: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct Page {
    task: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

/// Size of rayon's global pool once this module has set it.
static THREADS: OnceLock<usize> = OnceLock::new();

/// Performance cores on Apple Silicon. The efficiency cores slow the matmul
/// kernels down: work is split evenly, so every pass waits for the slow ones.
#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    let mut n: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let rc = unsafe { libc::sysctlbyname(c"hw.perflevel0.physicalcpu".as_ptr(), (&raw mut n).cast(), &mut len, std::ptr::null_mut(), 0) };
    (rc == 0 && n > 0).then_some(n as usize)
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}

fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// Size rayon's global pool: the `threads` option, else `RAYON_NUM_THREADS`,
/// else the performance cores. The pool is process-wide and built once, so
/// only the first instance's `threads` counts.
fn init_threads(requested: Option<usize>) -> usize {
    *THREADS.get_or_init(|| {
        let env = std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok());
        let n = requested.or(env).filter(|n| *n > 0).unwrap_or_else(|| performance_cores().unwrap_or_else(cores));
        // Fails when something else in the process built the pool first; the
        // size it has is then the answer.
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("cev-cpu-{i}")).build_global();
        rayon::current_num_threads()
    })
}

/// `CEV_LOG=debug` prints the engine's log (per-pass timings) to stderr.
fn init_log() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        if let Ok(filter) = std::env::var("CEV_LOG") {
            let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init();
        }
    });
}

/// Bounds the forward passes that run at once.
struct Permits {
    free: Mutex<usize>,
    returned: Condvar,
}

struct Permit<'a>(&'a Permits);

impl Permits {
    fn acquire(&self) -> Permit<'_> {
        let mut free = self.free.lock().expect("permit lock");
        while *free == 0 {
            free = self.returned.wait(free).expect("permit lock");
        }
        *free -= 1;
        Permit(self)
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().expect("permit lock") += 1;
        self.0.returned.notify_one();
    }
}

struct Shared {
    runtime: Runtime,
    permits: Permits,
    concurrency: usize,
    threads: usize,
}

impl Shared {
    fn open(o: Options) -> anyhow::Result<Arc<Self>> {
        init_log();
        let threads = init_threads(o.threads);
        let backend: Arc<dyn Backend> = if o.mock {
            Arc::new(MockBackend::default())
        } else {
            let d = EngineOptions::default();
            Arc::new(Engine::load(EngineOptions {
                model: o.model.unwrap_or(d.model),
                device: o.device.unwrap_or(d.device),
                dtype: o.dtype.unwrap_or(d.dtype),
                max_context: o.max_context.unwrap_or(d.max_context),
                prefix_cache_tokens: o.prefix_cache_tokens.unwrap_or(d.prefix_cache_tokens),
                name: o.name,
                ..d
            })?)
        };
        let d = RuntimeConfig::default();
        let cfg = RuntimeConfig {
            online_learning: o.online_learning.unwrap_or(d.online_learning),
            store_features: o.store_features.unwrap_or(d.store_features),
            learn: LearnConfig { min_examples: o.min_examples.unwrap_or(d.learn.min_examples), ..d.learn },
            debias: o.debias.unwrap_or(d.debias),
            max_permutations: o.max_permutations.unwrap_or(d.max_permutations),
        };
        let store = match o.db.as_deref() {
            None | Some(":memory:") => Store::in_memory()?,
            Some(path) => Store::open(path)?,
        };
        let concurrency = o.concurrency.unwrap_or(1).max(1);
        let permits = Permits { free: Mutex::new(concurrency), returned: Condvar::new() };
        Ok(Arc::new(Self { runtime: Runtime::new(backend, store, cfg)?, permits, concurrency, threads }))
    }

    /// One forward pass at a time per permit, on a thread of the CPU pool.
    /// candle's CPU kernels fork into that pool many times per layer: from
    /// inside it a fork is a work-stealing join, from outside a handoff to
    /// another thread and a sleep.
    fn pass<T: Send>(&self, pass: impl FnOnce(&Runtime) -> T + Send) -> T {
        let _permit = self.permits.acquire();
        let mut out = None;
        rayon::scope(|_| out = Some(pass(&self.runtime)));
        out.expect("scope ran the pass")
    }
}

/// Errors cross to Python as `<code>: <message>`; the package turns the code
/// into `CevError.code`.
fn fail(code: &str, message: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("{code}: {message}"))
}

fn error(e: CevError) -> PyErr {
    let code = match &e {
        CevError::BadRequest(_) => "bad_request",
        CevError::NotFound(_) => "not_found",
        CevError::Internal(_) => "internal",
    };
    fail(code, format_args!("{e:#}"))
}

fn parse<T: DeserializeOwned>(text: &str) -> CevResult<T> {
    serde_json::from_str(text).map_err(|e| CevError::BadRequest(e.to_string()))
}

fn text(v: &impl serde::Serialize) -> CevResult<String> {
    serde_json::to_string(v).map_err(|e| CevError::Internal(e.into()))
}

/// Library version, the tensor backends compiled in, and the CPU pool.
#[pyfunction]
fn info() -> String {
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "metal": cfg!(any(feature = "metal", target_os = "macos")),
        "accelerate": cfg!(any(feature = "accelerate", target_os = "macos")),
        "cuda": cfg!(feature = "cuda"),
        "cores": cores(),
        "performance_cores": performance_cores(),
        "threads": THREADS.get(),
    })
    .to_string()
}

#[pyclass(frozen, module = "cev._native")]
struct Cev {
    /// `None` once closed.
    shared: RwLock<Option<Arc<Shared>>>,
}

impl Cev {
    fn shared(&self) -> PyResult<Arc<Shared>> {
        self.shared.read().expect("handle lock").clone().ok_or_else(|| fail("closed", "this Cev instance is closed"))
    }

    /// Run `work` with the GIL released.
    fn run(&self, py: Python<'_>, work: impl FnOnce(&Shared) -> CevResult<String> + Send) -> PyResult<String> {
        let shared = self.shared()?;
        py.detach(move || work(&shared)).map_err(error)
    }
}

#[pymethods]
impl Cev {
    /// Load a model (or the mock).
    #[new]
    fn new(py: Python<'_>, options: &str) -> PyResult<Self> {
        let options: Options = serde_json::from_str(options).map_err(|e| fail("bad_request", e))?;
        let shared = py.detach(|| Shared::open(options)).map_err(|e| fail("internal", format_args!("{e:#}")))?;
        Ok(Self { shared: RwLock::new(Some(shared)) })
    }

    /// `GET /v1/models`, plus how this instance runs.
    fn model(&self) -> PyResult<String> {
        let shared = self.shared()?;
        let mut info = serde_json::to_value(shared.runtime.model_info()).map_err(|e| fail("internal", e))?;
        info["threads"] = json!(shared.threads);
        info["concurrency"] = json!(shared.concurrency);
        Ok(info.to_string())
    }

    /// `POST /v1/systemone`
    fn decide(&self, py: Python<'_>, request: &str) -> PyResult<String> {
        self.run(py, |s| {
            let request = parse::<SystemOneRequest>(request)?;
            text(&s.pass(|cev| cev.decide(&request))?)
        })
    }

    /// `POST /v1/examples`
    fn examples(&self, py: Python<'_>, request: &str) -> PyResult<String> {
        self.run(py, |s| {
            let request = parse::<ExamplesRequest>(request)?;
            let out = s.pass(|cev| cev.examples(&request))?;
            text(&out.into_iter().map(|(response, feedback)| json!({"response": response, "feedback": feedback})).collect::<Vec<_>>())
        })
    }

    /// `POST /v1/feedback`
    fn feedback(&self, py: Python<'_>, request: &str) -> PyResult<String> {
        self.run(py, |s| text(&s.runtime.feedback(&parse::<FeedbackRequest>(request)?)?))
    }

    /// `GET /v1/tasks`
    fn tasks(&self, py: Python<'_>) -> PyResult<String> {
        self.run(py, |s| text(&s.runtime.tasks()))
    }

    /// `DELETE /v1/tasks/{task}`
    fn reset_task(&self, py: Python<'_>, task: &str) -> PyResult<String> {
        self.run(py, |s| text(&s.runtime.reset_task(task)?))
    }

    /// `GET /v1/decisions`
    fn decisions(&self, py: Python<'_>, page: &str) -> PyResult<String> {
        self.run(py, |s| {
            let p: Page = parse(page)?;
            text(&s.runtime.store().recent_decisions(p.task.as_deref(), p.limit.unwrap_or(50).min(1000), p.offset.unwrap_or(0))?)
        })
    }

    /// `GET /v1/decisions/{id}`; `null` when there is no such decision.
    fn decision(&self, py: Python<'_>, id: &str) -> PyResult<String> {
        self.run(py, |s| {
            let Some((decision, feedback)) = s.runtime.decision(id)? else { return Ok("null".into()) };
            let request = s.runtime.store().request(&decision.request_id)?;
            let prompt = request.as_ref().map(|r| format!("{}{}", r.prefix, decision.suffix));
            text(&json!({"decision": decision, "state": request.map(|r| r.state), "prompt": prompt, "feedback": feedback}))
        })
    }

    /// `GET /v1/export`: NDJSON training rows.
    fn export(&self, py: Python<'_>, filter: &str) -> PyResult<String> {
        self.run(py, |s| {
            let mut out = String::new();
            s.runtime.export(&parse::<ExportFilter>(filter)?, |row| {
                out.push_str(&serde_json::to_string(&row)?);
                out.push('\n');
                Ok(())
            })?;
            Ok(out)
        })
    }

    /// `GET /v1/stats`
    fn stats(&self, py: Python<'_>) -> PyResult<String> {
        self.run(py, |s| text(&json!({"store": s.runtime.stats()?, "tasks": s.runtime.tasks().len(), "model": s.runtime.model_info()})))
    }

    /// Free the model once the calls in flight have returned.
    fn close(&self) {
        self.shared.write().expect("handle lock").take();
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Cev>()?;
    m.add_function(wrap_pyfunction!(info, m)?)?;
    Ok(())
}
