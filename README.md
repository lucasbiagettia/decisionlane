# decisionlane

One small interface for zero-shot, single-label text classification with
decision models.

You describe your cases in plain language, pick a backend, and get back a
normalized `Decision` whose `label` is always one of your cases. What you do
with the label (routing a ticket, tagging a message) stays in your application.

Supported backends in v0.1:

| Name       | Provider mechanism                                    | Credential         |
| ---------- | ----------------------------------------------------- | ------------------ |
| `jev`      | [TypeSafe](https://docs.typesafe.ai) Choice question   | `JEV_TOKEN`        |
| `emissary` | [Emissary](https://docs.withemissary.com) routing experiment | `EMISSARY_API_KEY` |
| `openai`   | [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) Choice question | `OPENAI_API_KEY` |

All three are zero-shot: you provide case descriptions, never labeled examples.
They share the same `Annotator`, `batch()` and `Decision` interface; switching
providers only requires changing the backend.

## Installation

```bash
pip install decisionlane
```

Python 3.10 or newer. The only runtime dependency is `requests`.

## Credentials

Set the variable for the backend you use:

```bash
export JEV_TOKEN=...
export EMISSARY_API_KEY=...
export OPENAI_API_KEY=...
```

decisionlane never loads `.env` files. If you keep secrets in one, load it in
your application (for example with `python-dotenv`) before the first
classification. Keys can also be passed explicitly to the adapters (see below).

## Quick start

```python
from decisionlane import Annotator

annotator = Annotator(
    backend="jev",  # or "emissary" or "openai"
    cases={
        "refund": "The customer asks for a refund",
        "card_lost": "Reports a lost or stolen card",
        "other": "Requests that match none of the cases above",
    },
)

decision = annotator("My card was stolen last night")
print(decision.label)                 # e.g. "card_lost"
print(decision.probabilities)         # {"refund": ..., "card_lost": ..., "other": ...}
print(decision.selected_probability)  # probability of decision.label

decisions = annotator.batch([
    "I want my money back",
    "How do I change my address?",
])

annotator.close()  # closes the HTTP session of the backend it created
```

`Annotator` exposes two operations:

- `annotator(text) -> Decision` classifies one text.
- `annotator.batch(texts) -> list[Decision]` classifies a sequence and returns
  one decision per text, in order.

## Configuring a backend explicitly

Pass an adapter instead of a name to set model, key or timeout:

```python
from decisionlane import Annotator
from decisionlane.backends import Emissary, Jev, OpenAI

annotator = Annotator(
    backend=Jev(model="jev-1.13.0", api_key="...", timeout=30.0),
    cases={
        "positive": "Favorable opinion",
        "negative": "Unfavorable opinion",
    },
)

emissary = Emissary(api_key="...", timeout=120.0, experiment_name="sentiment")
openai = OpenAI(model="gpt-6-luna", api_key="...", timeout=120.0)
```

| Adapter    | Options                                                        | Defaults                                         |
| ---------- | -------------------------------------------------------------- | ------------------------------------------------ |
| `Jev`      | `model`, `api_key`, `timeout`, `session`                       | `jev-1.13.0`, `JEV_TOKEN`, 30 s                  |
| `Emissary` | `api_key`, `timeout`, `experiment_name`, `session`             | `EMISSARY_API_KEY`, 120 s, `decisionlane-<random>` |
| `OpenAI`   | `model`, `api_key`, `timeout`, `session`                        | `gpt-6-luna`, `OPENAI_API_KEY`, 120 s              |

If you pass your own `requests.Session`, the adapter uses it and never closes
it. An adapter you construct yourself is yours to close (`adapter.close()`);
`Annotator.close()` only closes a backend it created from a name. Use one
adapter instance per `Annotator`.

## The `Decision` result

```python
@dataclass(frozen=True)
class Decision:
    label: str                                  # always one of your case ids
    probabilities: Mapping[str, float] | None   # over exactly your case ids, or None
    backend: str                                # "jev", "emissary", "openai", ...
    model: str | None                           # model/resource identifier reported by the provider
    request_id: str | None                      # provider request id, when available
    latency_ms: float
    provider_confidence: float | None           # provider-native score, when returned
    explanation: str | None                     # provider explanation, when returned

    @property
    def selected_probability(self) -> float | None: ...  # probabilities[label]
```

What each field means:

- **`probabilities`** is the provider's distribution, validated but never
  renormalized or invented. Values are finite numbers in [0, 1] whose sum is 1
  within an absolute tolerance of `1e-4`; anything else raises `ResponseError`.
- **`selected_probability`** is derived from `probabilities`, so the two cannot
  disagree. It is `None` when there is no distribution.
- **`provider_confidence`** is the native confidence returned by Jev or OpenAI,
  kept separately from the probability of the chosen label. Jev's score describes
  how concentrated the distribution is; native confidence has provider-specific
  meaning. Emissary does not return one (`None`).
- **`explanation`** is `None` for all three current backends: none returns an
  explanation in the classification response. decisionlane never generates one.
- **`model`**: for Jev and OpenAI, the resolved model returned by the API (which
  may differ from the requested alias). For Emissary, the `experiment_id/version` created
  for this annotator. A response reporting a different model is rejected.
- **`latency_ms`** is measured with a monotonic clock from the start of the
  backend's `predict` call until the normalized decision is built. It includes
  the network round trip and excludes backend preparation (such as creating the
  Emissary experiment).

### Ties

- **Jev and OpenAI** return a chosen label. When several labels share the maximum
  probability, the provider's choice is kept. A choice that is not a maximum is
  rejected.
- **Emissary** returns only a distribution. The label is the highest-probability
  case; on an exact tie, the first tied label in the response order wins.

## Behavior and guarantees

- **Validation first.** Cases need at least two entries with non-empty ids and
  descriptions. Texts must be non-empty strings. Everything is checked locally
  before any request, and nothing is truncated, translated or rewritten. The
  cases mapping is copied at construction, so later changes to your dictionary
  have no effect.
- **Lazy, one-time preparation.** Constructing an `Annotator` or importing the
  package makes no request and needs no credentials. The backend is prepared
  before the first classification and reused afterwards. A failed preparation is
  retried on the next call.
- **Sequential batches.** `batch()` sends one request per text, one after the
  other. It is not a remote batch API and does not improve throughput. An empty
  sequence returns `[]` without preparing or calling the provider. A single
  string is rejected.
- **Fail-fast.** The first error stops the batch and is raised; no partial list
  and no placeholder labels are returned. Earlier requests in that batch may
  already have completed (and been billed); retrying the batch repeats them.
- **No hidden behavior.** No automatic retries, no fallback between backends,
  no caching and no logging of texts. Error messages never include credentials,
  authorization headers or request/response bodies.

Errors: `decisionlane.BackendError` (transport failure, timeout or HTTP error;
has `status_code`) and `decisionlane.ResponseError` (the provider answered with
something that does not match the contract). Invalid configuration or input
raises `ValueError` or `TypeError`.

## Backend notes

### Jev

Each text is sent as the `state` of one TypeSafe Choice question whose criteria
are your case ids and descriptions. Preparation is local. Choice accepts at most
255 options; more cases are rejected before any request. For `jev-1.13.0` the
documented context limit is 32k tokens for state plus question: inputs whose
serialized request, plus a 1,024-unit framing allowance, exceed 32,000 UTF-8
bytes are rejected locally. This is a conservative estimate, not a tokenizer,
and it is checked per text, just before that text's request. Other model
versions are not checked locally; the provider's limits apply.

### Emissary

Preparation creates one remote experiment in `routing` mode from your cases.
This is configuration, not training. The experiment/version is reused for every
classification made by that annotator. Every new adapter instance (and every
new process) creates a new experiment; decisionlane does not persist or reuse
experiments across instances. If creating the experiment fails or times out,
the next call tries again, and a timed-out attempt may still have created an
experiment on the provider side. Classification uses `data_format="probs"`.
The provider's limits apply; no local limit on the number of cases is imposed.

### OpenAI

Each text is sent to `POST https://api.openai.com/v1/decisions` as `input`, with
one named `classification` question of type `choice`. Your case ids become
choice values and their descriptions become the rubric, using the same request
and response contract as `llm-classifier-bench`'s OpenAI Decisions zero-shot
classifier. Preparation is local and consumes no labeled examples.

The default model is `gpt-6-luna`. This uses the
[Decisions API](https://developers.openai.com/api/docs/guides/decisions) through
`requests`, so no OpenAI SDK dependency is needed. `OPENAI_ORG_ID` and
`OPENAI_PROJECT_ID`, when set, are sent as organization and project headers.
Model and timeout are configured on the adapter, as with Jev. Choice accepts at
most 255 cases; larger configurations are rejected before any request. The
provider's other limits apply.

The returned probabilities are validated and ordered by your configured cases.
The provider's choice is preserved, including exact ties. A `refusal` answer
raises `ResponseError` with a sanitized refusal message, without inventing a
label or probabilities; a batch stops at that text. HTTP and transport failures
raise `BackendError`, as with the other adapters. No automatic retries are made.

## Custom backends

Anything with these two methods can be passed as `backend`:

```python
from typing import Mapping
from decisionlane import Annotator, Decision

class MyBackend:
    def prepare(self, cases: Mapping[str, str]) -> None:
        self.labels = list(cases)

    def predict(self, text: str) -> Decision:
        return Decision(label=self.labels[0], probabilities=None,
                        backend="mine", latency_ms=0.0)

annotator = Annotator(backend=MyBackend(), cases={"a": "First", "b": "Second"})
```

This is the `decisionlane.DecisionModel` protocol; no subclassing is required.
`Annotator` validates every returned `Decision` against the configured cases.

## Scope of v0.1

Included: zero-shot, single-label, closed-set classification of text with Jev,
Emissary or OpenAI Decisions, one text at a time.

Not included: other providers, training or few-shot examples, evaluation or
calibration, thresholds or abstention, caching, persistence, async or
concurrent execution, a CLI, and automatic model selection or routing between
providers.

## License

MIT © 2026 Lucas Biagetti
