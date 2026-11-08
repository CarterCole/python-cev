"""cev: typed, calibrated decisions in one forward pass, in-process.

    from cev import Cev

    cev = Cev.load("Qwen/Qwen3-0.6B")      # or "kev:0.8b", "d1-omni", a directory, ...
    res = cev.decide(
        {"ticket": "The app crashes when I tap Log in"},
        {"team": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "Payments", "tech": "Bugs and crashes"}}},
    )
    res["answers"]["team"]["choice"]        # "tech"
    cev.feedback(decision_id=res["answers"]["team"]["x_cev"]["decision_id"], label="tech")
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable, Mapping, Sequence, TypeVar

from . import _native
from ._types import (
    Answer,
    AnswerMeta,
    ChoiceAnswer,
    Criterion,
    Debias,
    DecideResponse,
    DecisionDetail,
    DecisionRecord,
    Device,
    DType,
    Example,
    ExampleResult,
    ExportRow,
    FeedbackRecord,
    Family,
    FeedbackResponse,
    Info,
    Label,
    Media,
    Modality,
    ModelInfo,
    NoulAnswer,
    OptionSpec,
    Question,
    ScoreAnswer,
    Stats,
    StoreStats,
    TaskInfo,
    Usage,
)

__all__ = [
    "Answer",
    "AnswerMeta",
    "Cev",
    "CevError",
    "ChoiceAnswer",
    "Criterion",
    "DType",
    "Debias",
    "DecideResponse",
    "DecisionDetail",
    "DecisionRecord",
    "Device",
    "Example",
    "ExampleResult",
    "ExportRow",
    "Family",
    "FeedbackRecord",
    "FeedbackResponse",
    "Info",
    "Label",
    "Media",
    "Modality",
    "ModelInfo",
    "NoulAnswer",
    "OptionSpec",
    "Question",
    "ScoreAnswer",
    "Stats",
    "StoreStats",
    "TaskInfo",
    "Usage",
    "info",
]

_CODES = ("bad_request", "not_found", "closed", "internal")


class CevError(Exception):
    """Every failure of this package. ``code`` says whose fault it was:

    - ``bad_request``: the call was malformed (unknown option, no questions, ...)
    - ``not_found``: no such decision
    - ``closed``: the instance was closed
    - ``internal``: everything else (missing weights, I/O, ...)
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_T = TypeVar("_T")


def _call(fn: Callable[..., _T], *args: str | None) -> _T:
    """The native module reports errors as ``<code>: <message>``."""
    try:
        return fn(*args)
    except RuntimeError as e:
        code, sep, message = str(e).partition(": ")
        if sep and code in _CODES:
            raise CevError(code, message) from None
        raise CevError("internal", str(e)) from None


def _json(fn: Callable[..., str], *args: Any) -> Any:
    return json.loads(_call(fn, *(a if a is None or isinstance(a, str) else json.dumps(a) for a in args)))


def _encode(media: Media) -> str:
    """Base64 for a path or bytes; a ``str`` is already in the wire format."""
    if isinstance(media, str):
        return media
    if isinstance(media, os.PathLike):
        with open(media, "rb") as f:
            media = f.read()
    if isinstance(media, (bytes, bytearray, memoryview)):
        return base64.b64encode(media).decode("ascii")
    raise TypeError(f"images and audio are paths, bytes or base64 strings, not {type(media).__name__}")


def _media(request: Mapping[str, Any]) -> dict[str, Any]:
    """A request body with its images and audio as base64 strings."""
    out = dict(request)
    if out.get("images") is not None:
        if isinstance(out["images"], (str, bytes, os.PathLike)):
            raise TypeError("images is a list; pass [image] for one")
        out["images"] = [_encode(m) for m in out["images"]]
    if out.get("audio") is not None:
        out["audio"] = _encode(out["audio"])
    return out


def _given(**fields: Any) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


def info() -> Info:
    """What was compiled into the native module, and the machine it runs on."""
    return json.loads(_native.info())


class Cev:
    """A loaded model with its decision log.

    Every method releases the GIL while it works, so calls from several
    threads overlap; ``concurrency`` of them run a forward pass at a time and
    the rest wait their turn. In async code use ``asyncio.to_thread``.
    """

    def __init__(self, native: _native.Cev) -> None:
        """Use :meth:`load` or :meth:`mock`."""
        self._native = native
        self.model: ModelInfo = json.loads(native.model())
        """The loaded model and how it runs."""

    @classmethod
    def load(
        cls,
        model: str | None = None,
        *,
        name: str | None = None,
        device: Device | None = None,
        dtype: DType | None = None,
        max_context: int | None = None,
        prefix_cache_tokens: int | None = None,
        threads: int | None = None,
        concurrency: int | None = None,
        db: str | None = None,
        online_learning: bool | None = None,
        store_features: bool | None = None,
        min_examples: int | None = None,
        debias: Debias | None = None,
        max_permutations: int | None = None,
    ) -> Cev:
        """Load a model from a directory or the Hugging Face cache,
        downloading it into ``~/.cache/cev/models`` on first use.

        :param model: What to run (default ``Qwen/Qwen3-1.7B``): a Hugging Face
            repo id, a local directory, or a short name (``kev``, ``kev:0.8b``,
            ``kev:4b``, ``nimble-v3``, ``d3``, ``d3:2b``, ``d1``, ``d1-omni``).
            The family is detected from the files: Qwen3 and Qwen3.5
            checkpoints, or the Nimble, Kev, d3, d1 and d1-omni decision models.
            ``model.modalities`` says whether it takes images or audio.
        :param name: Name reported as ``model`` in responses (default ``cev-<model name>``).
        :param device: Default ``auto``: Metal, then CUDA, then CPU, of those compiled in.
        :param dtype: Default ``auto``: f32 on CPU and for checkpoints up to 4 GB on Metal, else bf16.
        :param max_context: Longest prompt in tokens (default 32768, capped by the model).
        :param prefix_cache_tokens: Tokens of state prefixes kept for reuse across requests (default 65536).
        :param threads: CPU threads for the tensor kernels. Defaults to
            ``RAYON_NUM_THREADS``, else the performance cores. Process-wide:
            only the first load's value counts.
        :param concurrency: Forward passes that may run at once (default 1; keep 1 on a GPU).
        :param db: SQLite file for the decision log, feedback and adapters.
            Defaults to ``:memory:``, which is gone when the process exits.
        :param online_learning: Learn from labels as they arrive (default True).
            When False, labels are only stored.
        :param store_features: Keep hidden states with decisions (default True).
            Online learning from later feedback needs them.
        :param min_examples: Labels a task needs before its adapter may be served (default 8).
        :param debias: Default position-bias correction (default ``auto``).
        :param max_permutations: Most option rotations per question when permuting (default 4).
        """
        return cls._open(
            model=model,
            name=name,
            device=device,
            dtype=dtype,
            max_context=max_context,
            prefix_cache_tokens=prefix_cache_tokens,
            threads=threads,
            concurrency=concurrency,
            db=db,
            online_learning=online_learning,
            store_features=store_features,
            min_examples=min_examples,
            debias=debias,
            max_permutations=max_permutations,
        )

    @classmethod
    def mock(
        cls,
        *,
        db: str | None = None,
        online_learning: bool | None = None,
        store_features: bool | None = None,
        min_examples: int | None = None,
        debias: Debias | None = None,
        max_permutations: int | None = None,
        media: bool | None = None,
    ) -> Cev:
        """A keyword-overlap stand-in for the model: no weights, instant, and
        the whole API works, including learning. For tests and wiring.

        :param media: Accept ``images`` and ``audio`` (default False: text only, like Qwen3).
        """
        return cls._open(
            mock=True,
            media=media,
            db=db,
            online_learning=online_learning,
            store_features=store_features,
            min_examples=min_examples,
            debias=debias,
            max_permutations=max_permutations,
        )

    @classmethod
    def _open(cls, **options: Any) -> Cev:
        return cls(_call(_native.Cev, json.dumps(_given(**options))))

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        no_store: bool = False,
        debias: Debias | None = None,
        namespace: str | None = None,
        images: Sequence[Media] | None = None,
        audio: Media | None = None,
    ) -> DecideResponse:
        """Answer every question about ``state`` in one forward pass.

        :param state: The evidence: free text, or anything ``json.dumps`` takes (program state).
        :param questions: Asked together and answered under the same ids.
        :param no_store: Do not log this request. Its answers then cannot take feedback.
        :param debias: Overrides the instance's ``debias`` for this request.
        :param namespace: Feedback bucket. Online adapters are kept per namespace
            and task, so labels on decisions made in one namespace only change
            answers served in that namespace (one per customer, say). Omit for
            the default bucket.
        :param images: Images every question sees, before the state: PNG, JPEG,
            WebP or GIF. Needs a model with ``image`` in ``model["modalities"]``
            (d1, d1-omni, d3).
        :param audio: One sound clip every question hears: WAV (16 kHz mono is
            best; other rates are resampled). Needs a model with ``audio`` in
            ``model["modalities"]`` (d1-omni).
        """
        request = {
            "state": state,
            "questions": questions,
            "no_store": no_store,
            **_given(debias=debias, namespace=namespace, images=images, audio=audio),
        }
        return _json(self._native.decide, _media(request))

    def feedback(
        self,
        *,
        decision_id: str | None = None,
        request_id: str | None = None,
        question_id: str | None = None,
        label: Label | None = None,
        weight: float | None = None,
        comment: str | None = None,
        metadata: Any = None,
    ) -> FeedbackResponse:
        """Attach a label or a comment to a past answer, and learn from the label.

        Name the answer by ``decision_id``, or by ``request_id`` and ``question_id``.

        :param label: The right answer; omit to leave only a comment.
        :param weight: Importance of this label (default 1).
        :param metadata: Anything worth keeping with the label: who gave it, where it came from.
        """
        request = _given(
            decision_id=decision_id,
            request_id=request_id,
            question_id=question_id,
            label=label,
            weight=weight,
            comment=comment,
            metadata=metadata,
        )
        return _json(self._native.feedback, request)

    def examples(self, examples: Sequence[Example]) -> list[ExampleResult]:
        """Decide each example, then apply its labels as feedback."""
        return _json(self._native.examples, {"examples": [_media(e) for e in examples]})

    def tasks(self, namespace: str | None = None) -> list[TaskInfo]:
        """The online adapter of every task that has labels: in every
        namespace, or only in ``namespace`` (``""`` for the default one)."""
        return _json(self._native.tasks, namespace)

    def reset_task(self, task: str, namespace: str | None = None) -> bool:
        """Drop a task's adapter in ``namespace`` (default: the default one).
        Its labels stay in the log. Returns whether there was one."""
        return _json(self._native.reset_task, task, namespace)

    def decisions(
        self, *, task: str | None = None, namespace: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[DecisionRecord]:
        """Recent decisions, newest first (at most 1000 per call).

        :param namespace: Only this namespace (``""`` for the default one).
        """
        return _json(self._native.decisions, _given(task=task, namespace=namespace, limit=limit, offset=offset))

    def decision(self, decision_id: str) -> DecisionDetail | None:
        """One decision with its state, the exact prompt and all its feedback; ``None`` if unknown."""
        return _json(self._native.decision, decision_id)

    def export(
        self,
        *,
        task: str | None = None,
        namespace: str | None = None,
        model: str | None = None,
        labeled: bool = True,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[ExportRow]:
        """Training rows: one per labelled decision, or per decision with ``labeled=False``.

        :param namespace: Only this namespace (``""`` for the default one).
        :param model: Only decisions made by this model id.
        :param since: Only rows decided or labelled at or after this unix time in milliseconds.
        """
        text = self.export_ndjson(task=task, namespace=namespace, model=model, labeled=labeled, since=since, limit=limit)
        return [json.loads(line) for line in text.splitlines()]

    def export_ndjson(
        self,
        *,
        task: str | None = None,
        namespace: str | None = None,
        model: str | None = None,
        labeled: bool = True,
        since: int | None = None,
        limit: int | None = None,
    ) -> str:
        """The same rows as newline-delimited JSON, the file rust-cev's
        ``training/train_lora.py --source`` reads."""
        filter = _given(task=task, namespace=namespace, model=model, labeled=labeled, since=since, limit=limit)
        return _call(self._native.export, json.dumps(filter))

    def stats(self) -> Stats:
        """Counts of what is in the log."""
        return _json(self._native.stats)

    def close(self) -> None:
        """Free the model once the calls in flight have returned. Later calls
        raise :class:`CevError` with code ``closed``."""
        self._native.close()

    def __enter__(self) -> Cev:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
