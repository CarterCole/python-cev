"""Wire types: the bodies of cev's REST API (``POST /v1/systemone`` and
friends), field for field. They are plain dicts at run time; these classes
only describe them to a type checker."""

from __future__ import annotations

import os
from typing import Any, Literal, Mapping, NotRequired, Sequence, TypedDict

Debias = Literal["auto", "none", "calibrate", "permute", "full"]
"""How to correct the model's option-position bias; ``full`` does both.
``auto`` is the model's own default: ``full`` for zero-shot code readouts
(Qwen3, Qwen3.5, Nimble), ``none`` for trained decision models (Kev, d3, d1,
d1-omni)."""

Media = str | bytes | bytearray | memoryview | os.PathLike[str]
"""An image or a sound clip: a file path, the file's bytes, or base64 (with
or without a ``data:...;base64,`` header). A plain ``str`` is always base64;
pass a :class:`pathlib.Path` for a file."""

Family = Literal["cev", "nimble", "kev", "d3", "d1", "d1-omni"]
"""Decision-model family: how the prompt is built and the answer read out."""

Modality = Literal["text", "image", "audio"]

Device = Literal["auto", "cpu", "metal", "cuda"]
DType = Literal["auto", "f32", "f16", "bf16"]

Criterion = str | None | Mapping[str, Any]
"""Describes one option: text, or TypeSafe's structured form
(``{"what": ..., "not_for": ..., "examples": ...}``)."""

Label = bool | float | str | Mapping[str, float]
"""The right answer to a past question.

- noul: ``True``/``False``, ``"yes"``/``"no"``, or a probability
- choice: the option name, or ``{option: probability}``
- score: a level index (fractions split between neighbours), or ``{"0": p, ...}``
"""


class Question(TypedDict):
    type: Literal["noul", "choice", "score"]
    """noul: "is this true?"; choice: "which of these?"; score: "which level?"."""
    instructions: str
    criteria: NotRequired[str | None | Mapping[str, Criterion] | Sequence[Criterion]]
    """noul: what counts as yes, or ``{"true": ..., "false": ...}`` (optional).
    choice: option names mapped to descriptions, or just the names.
    score: level descriptions, lowest first."""
    task: NotRequired[str]
    """Stable name for the online-learning adapter. Without it the key is a hash
    of the type, instructions and option names, so rewording starts over."""


class AnswerMeta(TypedDict):
    decision_id: str
    """Send this back with ``feedback``. Empty when the request had ``no_store``."""
    task: str
    namespace: NotRequired[str]
    """The feedback bucket, when the request named one."""
    adapted: bool
    """True when an online adapter adjusted this answer."""
    base_probabilities: NotRequired[list[float]]
    """The distribution before the adapter, in the order of ``probabilities``."""


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float
    """Probability that the answer is yes."""
    x_cev: AnswerMeta


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    confidence: float
    """``(p_max - 1/K) / (1 - 1/K)``: 0 for a uniform distribution, 1 for a certain one."""
    probabilities: dict[str, float]
    x_cev: AnswerMeta


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    """Expected level index, so it can be fractional."""
    confidence: float
    """``1 - E|level - mode| / (K - 1)``."""
    legend: dict[str, str]
    """Level index to its description."""
    probabilities: dict[str, float]
    """Level index to its probability."""
    x_cev: AnswerMeta


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class DecideResponse(TypedDict):
    model: str
    answers: dict[str, Answer]
    """One answer per question, under the question's id."""
    usage: Usage
    request_id: str
    latency_ms: float


class FeedbackResponse(TypedDict):
    feedback_id: str
    decision_id: str
    task: str
    namespace: NotRequired[str]
    """The decision's feedback bucket, when it has one."""
    served_loss: NotRequired[float]
    """Log loss of the answer that was served, on this label."""
    learned: bool
    """Whether the label updated the task's adapter."""
    task_examples: int
    adapter_active: bool


class Example(TypedDict):
    """A labelled example: decide, then learn from the labels."""

    state: Any
    questions: Mapping[str, Question]
    labels: Mapping[str, Label]
    namespace: NotRequired[str]
    images: NotRequired[Sequence[Media]]
    audio: NotRequired[Media]


class ExampleResult(TypedDict):
    response: DecideResponse
    feedback: list[FeedbackResponse]


class TaskInfo(TypedDict):
    """One task's online adapter."""

    task: str
    namespace: NotRequired[str]
    """Feedback bucket; absent for the default one."""
    examples: int
    """Labels seen."""
    buffered: int
    """Labels in the replay buffer."""
    active: bool
    """Whether the adapter is being served: enough labels, and it beats the base model."""
    temperature: float
    base_loss: float
    """Running log loss of the base model on labels, each scored before it was learned from."""
    adapted_loss: float
    """The same for the adapter."""


class ModelInfo(TypedDict):
    id: str
    """Name reported in responses, e.g. ``cev-qwen3-0.6b``."""
    object: Literal["model"]
    backbone: str
    """Hugging Face repo id or directory of the weights."""
    family: Family
    modalities: list[Modality]
    """What a request may carry besides text."""
    hidden_size: int
    max_options: int
    """Most options one question can have."""
    prompt_version: str
    threads: NotRequired[int]
    """CPU threads the tensor kernels use (process-wide)."""
    concurrency: NotRequired[int]
    """Forward passes that may run at once."""


class OptionSpec(TypedDict):
    name: str
    """Key in ``probabilities``."""
    text: str
    """Text the model saw for this option."""


class DecisionRecord(TypedDict):
    """A logged decision: one question of one request."""

    decision_id: str
    request_id: str
    question_id: str
    model: str
    task: str
    namespace: NotRequired[str]
    """Feedback bucket; absent for the default one."""
    kind: Literal["noul", "choice", "score"]
    question: Question
    options: list[OptionSpec]
    codes: list[str]
    """The answer token of each option."""
    suffix: str
    """Prompt text after the request's shared prefix."""
    base_logits: list[float]
    served: list[float]
    """The distribution that was returned."""
    adapted: bool
    created_at: int
    """Unix time in milliseconds."""


class FeedbackRecord(TypedDict):
    feedback_id: str
    decision_id: str
    label: Label | None
    target: list[float] | None
    """The label as a distribution over the decision's options."""
    weight: float
    comment: str | None
    metadata: Any
    created_at: int


class DecisionDetail(TypedDict):
    decision: DecisionRecord
    state: Any
    prompt: str | None
    """The exact text the model saw."""
    feedback: list[FeedbackRecord]


class ExportRow(TypedDict):
    """One training row: exactly what the model saw, and what it should say."""

    decision_id: str
    request_id: str
    question_id: str
    model: str
    task: str
    namespace: NotRequired[str]
    """Feedback bucket; absent for the default one."""
    type: Literal["noul", "choice", "score"]
    prompt: str
    """The full prompt; the answer is the next token."""
    codes: list[str]
    options: list[str]
    target: list[float] | None
    """Target distribution over ``options``; ``None`` for unlabelled rows."""
    label: Label | None
    weight: float
    served: list[float]
    base_probabilities: list[float]
    adapted: bool
    comment: str | None
    metadata: Any
    state: Any
    question: Question
    feedback_id: str | None
    decided_at: int
    labeled_at: int | None


class StoreStats(TypedDict):
    requests: int
    decisions: int
    feedback: int
    labeled_decisions: int


class Stats(TypedDict):
    store: StoreStats
    tasks: int
    """Tasks with an adapter."""
    model: ModelInfo


class Info(TypedDict):
    """What was compiled into the native module, and the machine it runs on."""

    version: str
    metal: bool
    """Apple GPU support."""
    accelerate: bool
    """Apple's BLAS for the CPU path."""
    cuda: bool
    """NVIDIA GPU support."""
    cores: int
    """Logical cores."""
    performance_cores: int | None
    """Performance cores on Apple Silicon, else ``None``."""
    threads: int | None
    """Size of the CPU pool; ``None`` until the first ``Cev.load`` sets it."""
