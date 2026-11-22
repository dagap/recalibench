"""Value types shared by every plug-in and pipeline stage."""

from dataclasses import dataclass, field
from typing import Any, Literal, cast

QuestionKind = Literal["binary", "choice"]
QUESTION_KINDS: frozenset[str] = frozenset({"binary", "choice"})


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class Question:
    kind: QuestionKind  # each model maps this to its own primitives
    text: str
    options: tuple[Option, ...] = ()  # choice only

    def __post_init__(self) -> None:
        if self.kind not in QUESTION_KINDS:
            raise ValueError(f"question.kind must be one of {sorted(QUESTION_KINDS)}")
        if not self.text.strip():
            raise ValueError("question.text must not be empty")
        if self.kind == "binary" and self.options:
            raise ValueError("binary questions take no options")
        if self.kind == "choice":
            keys = [o.key for o in self.options]
            if len(keys) < 2:
                raise ValueError("choice questions need at least 2 options")
            if len(set(keys)) != len(keys):
                raise ValueError("choice option keys must be unique")

    @property
    def keys(self) -> tuple[str, ...]:
        """Probability keys a Prediction must carry for this question."""
        if self.kind == "binary":
            return ("yes", "no")
        return tuple(o.key for o in self.options)


@dataclass(frozen=True)
class Item:
    id: str
    state: dict[str, Any]  # what the model sees
    label: int | str  # ground truth, never sent to a model


@dataclass(frozen=True)
class Prediction:
    item_id: str
    probs: dict[str, float]  # binary: {"yes": p, "no": 1-p}; choice: one per option key
    std: dict[str, float] | None  # spread across repeats, if any
    model: str  # name@version
    latency_ms: float | None
    cost_usd: float


@dataclass(frozen=True)
class TaskSpec:
    """A task config, parsed and validated; tasks/builder.py interprets state and label."""

    name: str
    dataset: str
    question: Question
    state: dict[str, Any]
    label: dict[str, Any]
    sample_size: int | None = None
    splits: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=lambda: {})

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "TaskSpec":
        known = {"dataset", "question", "state", "label", "sample_size", "splits"}
        missing = {"dataset", "question", "state", "label"} - raw.keys()
        if missing:
            raise ValueError(f"task {name!r} is missing {sorted(missing)}")
        q = cast(dict[str, Any], raw["question"])
        options = tuple(
            Option(**cast(dict[str, str], o)) for o in cast(list[Any], q.get("options", []))
        )
        question = Question(kind=q["kind"], text=q["text"], options=options)
        return cls(
            name=name,
            dataset=str(raw["dataset"]),
            question=question,
            state=cast(dict[str, Any], raw["state"]),
            label=cast(dict[str, Any], raw["label"]),
            sample_size=raw.get("sample_size"),
            splits=raw.get("splits"),
            extra={k: v for k, v in raw.items() if k not in known},
        )
