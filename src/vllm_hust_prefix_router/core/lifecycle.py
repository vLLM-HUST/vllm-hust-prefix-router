# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle-aware request load accounting for prefix routing."""

import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

WorkerKey = tuple[str, int | None]


@dataclass(frozen=True)
class LifecycleRoutingConfig:
    """Configuration for lifecycle-aware routing costs."""

    prefill_load_weight: float = 1.0
    active_request_weight: float = 0.0
    selection_temperature: float = 0.0
    track_output_blocks: bool = False
    request_expiry_s: float = 300.0

    def __post_init__(self) -> None:
        for name, value in (
            ("prefill_load_weight", self.prefill_load_weight),
            ("active_request_weight", self.active_request_weight),
            ("selection_temperature", self.selection_temperature),
            ("request_expiry_s", self.request_expiry_s),
        ):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.request_expiry_s == 0:
            raise ValueError("request_expiry_s must be positive")
        if not isinstance(self.track_output_blocks, bool):
            raise ValueError("track_output_blocks must be a boolean")


@dataclass(frozen=True)
class PromptLoad:
    """Prompt-level load and active-block ownership."""

    prompt_tokens: int
    matched_tokens: int
    block_keys: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= self.matched_tokens <= self.prompt_tokens:
            raise ValueError("matched_tokens must be within the prompt length")
        if not self.block_keys:
            raise ValueError("block_keys must not be empty")

    @property
    def uncached_tokens(self) -> int:
        return self.prompt_tokens - self.matched_tokens


@dataclass(frozen=True)
class RequestLoad:
    """Candidate-specific cost inputs for one incoming API request."""

    node_id: str
    data_parallel_rank: int | None
    prompt_tokens: int
    matched_tokens: int
    prompt_count: int
    block_keys: tuple[bytes, ...]
    cache_size_blocks: int = 0
    prompt_loads: tuple[PromptLoad, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if self.data_parallel_rank is not None and self.data_parallel_rank < 0:
            raise ValueError("data_parallel_rank must be non-negative")
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= self.matched_tokens <= self.prompt_tokens:
            raise ValueError("matched_tokens must be within the prompt length")
        if self.prompt_count <= 0:
            raise ValueError("prompt_count must be positive")
        if not self.block_keys:
            raise ValueError("block_keys must not be empty")
        if (
            not isinstance(self.cache_size_blocks, int)
            or isinstance(self.cache_size_blocks, bool)
            or self.cache_size_blocks < 0
        ):
            raise ValueError("cache_size_blocks must be a non-negative integer")
        if self.prompt_loads:
            if len(self.prompt_loads) != self.prompt_count:
                raise ValueError("prompt_loads must contain one entry per prompt")
            if sum(prompt.prompt_tokens for prompt in self.prompt_loads) != (
                self.prompt_tokens
            ):
                raise ValueError(
                    "prompt_loads token total does not match prompt_tokens"
                )
            if sum(prompt.matched_tokens for prompt in self.prompt_loads) != (
                self.matched_tokens
            ):
                raise ValueError(
                    "prompt_loads match total does not match matched_tokens"
                )
            flattened_keys = tuple(
                key for prompt in self.prompt_loads for key in prompt.block_keys
            )
            if flattened_keys != self.block_keys:
                raise ValueError("prompt_loads block keys do not match block_keys")

    @property
    def worker(self) -> WorkerKey:
        return self.node_id, self.data_parallel_rank

    @property
    def uncached_tokens(self) -> int:
        return self.prompt_tokens - self.matched_tokens

    @property
    def prompts(self) -> tuple[PromptLoad, ...]:
        if self.prompt_loads:
            return self.prompt_loads
        return (
            PromptLoad(
                prompt_tokens=self.prompt_tokens,
                matched_tokens=self.matched_tokens,
                block_keys=self.block_keys,
            ),
        )


@dataclass(frozen=True)
class CandidateLoadScore:
    """Cost breakdown for one eligible worker."""

    load: RequestLoad
    active_prefill_tokens: int
    incoming_prefill_tokens: int
    potential_prefill_blocks: float
    active_decode_blocks: int
    incoming_decode_blocks: int
    potential_decode_blocks: int
    active_requests: int
    active_request_cost: float
    cost: float


@dataclass(frozen=True)
class LifecycleSelection:
    """Selected worker and the complete candidate cost set."""

    selected: CandidateLoadScore
    candidates: tuple[CandidateLoadScore, ...]


@dataclass(frozen=True)
class WorkerLoadView:
    """Active load owned by requests routed through this tracker."""

    active_prefill_tokens: int
    active_decode_blocks: int
    active_requests: int
    tracked_prefill_tokens: int
    tracked_decode_blocks: int
    tracked_output_blocks: int
    tracked_requests: int
    pending_requests: int
    accepted_requests: int


@dataclass
class _TrackedRequest:
    request_id: str
    candidates: dict[WorkerKey, RequestLoad]
    worker: WorkerKey
    prefill_completed: set[int]
    released_prompts: set[int]
    output_blocks: dict[tuple[int, int], tuple[bytes, ...]]
    generated_tokens: dict[tuple[int, int], int]
    dispatched: bool
    accepted: bool
    last_touched_s: float


class LifecycleLoadTracker:
    """Track request-owned Prefill and Decode load until terminal release."""

    def __init__(
        self,
        block_size: int,
        config: LifecycleRoutingConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        random_sample: Callable[[], float] = random.random,
    ) -> None:
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise ValueError("block_size must be a positive integer")
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self.config = config or LifecycleRoutingConfig()
        self._clock = clock
        self._random_sample = random_sample
        self._requests: dict[str, _TrackedRequest] = {}
        self._worker_requests: dict[WorkerKey, set[str]] = defaultdict(set)
        self._worker_block_refs: dict[WorkerKey, Counter[bytes]] = defaultdict(Counter)
        self._lock = RLock()

    def select_and_reserve(
        self,
        request_id: str,
        candidates: Sequence[RequestLoad],
    ) -> LifecycleSelection:
        """Select the lowest-cost candidate and reserve it before returning."""
        if not request_id:
            raise ValueError("request_id must not be empty")
        candidate_map = {candidate.worker: candidate for candidate in candidates}
        if not candidate_map:
            raise ValueError("at least one candidate is required")
        if len(candidate_map) != len(candidates):
            raise ValueError("candidate workers must be unique")

        with self._lock:
            now_s = self._clock()
            self._cleanup_expired_locked(now_s)
            if request_id in self._requests:
                raise ValueError(f"request {request_id!r} is already active")

            scores = tuple(
                self._score_locked(candidate) for candidate in candidate_map.values()
            )
            selected = self._select_score(scores)
            self._requests[request_id] = _TrackedRequest(
                request_id=request_id,
                candidates=candidate_map,
                worker=selected.load.worker,
                prefill_completed=set(),
                released_prompts=set(),
                output_blocks={},
                generated_tokens={},
                dispatched=False,
                accepted=False,
                last_touched_s=now_s,
            )
            self._acquire_locked(request_id, selected.load)
            return LifecycleSelection(selected=selected, candidates=scores)

    def mark_dispatched(self, request_id: str, worker: WorkerKey) -> bool:
        """Record the worker that accepted the current dispatch attempt."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            if request.worker != worker:
                return self._reassign_locked(request, worker)
            request.dispatched = True
            request.last_touched_s = self._clock()
            return True

    def mark_accepted(self, request_id: str) -> bool:
        """Record that the selected serving endpoint accepted the request."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.accepted:
                return False
            request.dispatched = True
            request.accepted = True
            request.last_touched_s = self._clock()
            return True

    def mark_prefill_completed(
        self,
        request_id: str,
        prompt_index: int | None = None,
    ) -> bool:
        """Stop charging Prefill work for one prompt or the whole request."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            prompt_count = len(request.candidates[request.worker].prompts)
            indices = range(prompt_count) if prompt_index is None else (prompt_index,)
            changed = False
            for index in indices:
                if not 0 <= index < prompt_count:
                    continue
                if index not in request.prefill_completed:
                    request.prefill_completed.add(index)
                    changed = True
            if not changed:
                return False
            request.last_touched_s = self._clock()
            return True

    def record_output_tokens(
        self,
        request_id: str,
        prompt_index: int,
        generated_tokens: int,
        choice_index: int = 0,
    ) -> bool:
        """Reconcile exact generated-token progress into active Decode blocks."""
        for name, value in (
            ("generated_tokens", generated_tokens),
            ("choice_index", choice_index),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or prompt_index in request.released_prompts:
                return False
            load = request.candidates[request.worker]
            if not 0 <= prompt_index < len(load.prompts):
                return False
            sequence_key = prompt_index, choice_index
            previous = request.generated_tokens.get(sequence_key, 0)
            if generated_tokens <= previous:
                return False
            request.generated_tokens[sequence_key] = generated_tokens
            request.last_touched_s = self._clock()
            if not self.config.track_output_blocks:
                return True

            prompt = load.prompts[prompt_index]
            prompt_blocks = math.ceil(prompt.prompt_tokens / self.block_size)
            sequence_blocks = math.ceil(
                (prompt.prompt_tokens + generated_tokens) / self.block_size
            )
            required = sequence_blocks - prompt_blocks
            current_keys = request.output_blocks.get(sequence_key, ())
            if required <= len(current_keys):
                return True
            new_keys = tuple(
                self._output_block_key(
                    request_id,
                    prompt_index,
                    choice_index,
                    block_index,
                )
                for block_index in range(len(current_keys), required)
            )
            request.output_blocks[sequence_key] = current_keys + new_keys
            self._worker_block_refs[request.worker].update(new_keys)
            return True

    def free_prompt(self, request_id: str, prompt_index: int) -> bool:
        """Release one completed prompt while sibling prompts keep running."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            load = request.candidates[request.worker]
            if (
                not 0 <= prompt_index < len(load.prompts)
                or prompt_index in request.released_prompts
            ):
                return False
            self._release_prompt_locked(request, prompt_index)
            request.last_touched_s = self._clock()
            if len(request.released_prompts) == len(load.prompts):
                self._requests.pop(request_id, None)
                self._worker_requests[request.worker].discard(request_id)
                if not self._worker_requests[request.worker]:
                    self._worker_requests.pop(request.worker, None)
            return True

    def touch(self, request_id: str) -> bool:
        """Renew the fallback lease when response progress is observed."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            request.last_touched_s = self._clock()
            return True

    def free(self, request_id: str) -> bool:
        """Release all request-owned load; repeated releases are harmless."""
        with self._lock:
            request = self._requests.pop(request_id, None)
            if request is None:
                return False
            self._release_locked(request)
            return True

    def cleanup_expired(self) -> tuple[str, ...]:
        """Release requests whose response progress lease expired."""
        with self._lock:
            return self._cleanup_expired_locked(self._clock())

    def worker_load(self, worker: WorkerKey) -> WorkerLoadView:
        """Return the current effective and locally tracked load."""
        with self._lock:
            now_s = self._clock()
            self._cleanup_expired_locked(now_s)
            return self._worker_load_locked(worker)

    def active_request_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._requests)

    def request_prompt_count(self, request_id: str) -> int | None:
        """Return the number of independently tracked prompts."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return None
            return len(request.candidates[request.worker].prompts)

    def _select_score(
        self,
        scores: tuple[CandidateLoadScore, ...],
    ) -> CandidateLoadScore:
        temperature = self.config.selection_temperature
        if temperature == 0:
            minimum = min(score.cost for score in scores)
            tied = [score for score in scores if score.cost == minimum]
            if len(tied) == 1:
                return tied[0]
            smallest_cache = min(score.load.cache_size_blocks for score in tied)
            if any(score.load.cache_size_blocks != smallest_cache for score in tied):
                tied = [
                    score
                    for score in tied
                    if score.load.cache_size_blocks == smallest_cache
                ]
            return tied[self._sample_index(len(tied))]

        minimum = min(score.cost for score in scores)
        maximum = max(score.cost for score in scores)
        if minimum == maximum:
            probabilities = [1 / len(scores)] * len(scores)
        else:
            value_range = maximum - minimum
            scaled = [-(score.cost / value_range) / temperature for score in scores]
            max_scaled = max(scaled)
            weights = [math.exp(value - max_scaled) for value in scaled]
            total = sum(weights)
            probabilities = [weight / total for weight in weights]

        sample = self._bounded_sample()
        cumulative = 0.0
        for score, probability in zip(scores, probabilities, strict=True):
            cumulative += probability
            if sample <= cumulative:
                return score
        return scores[-1]

    def _sample_index(self, size: int) -> int:
        return min(int(self._bounded_sample() * size), size - 1)

    def _bounded_sample(self) -> float:
        sample = self._random_sample()
        if not isinstance(sample, int | float) or isinstance(sample, bool):
            raise ValueError("random_sample must return a number")
        if not math.isfinite(sample) or not 0 <= sample <= 1:
            raise ValueError("random_sample must return a value in [0, 1]")
        return float(sample)

    def _score_locked(
        self,
        candidate: RequestLoad,
    ) -> CandidateLoadScore:
        load = self._worker_load_locked(candidate.worker)
        active_keys = self._worker_block_refs.get(candidate.worker, {})
        incoming_decode_blocks = sum(
            key not in active_keys for key in set(candidate.block_keys)
        )
        potential_prefill_blocks = (
            load.active_prefill_tokens + candidate.uncached_tokens
        ) / self.block_size
        potential_decode_blocks = load.active_decode_blocks + incoming_decode_blocks
        active_request_cost = self.config.active_request_weight * (
            load.active_requests + candidate.prompt_count
        )
        cost = (
            self.config.prefill_load_weight * potential_prefill_blocks
            + potential_decode_blocks
            + active_request_cost
        )
        return CandidateLoadScore(
            load=candidate,
            active_prefill_tokens=load.active_prefill_tokens,
            incoming_prefill_tokens=candidate.uncached_tokens,
            potential_prefill_blocks=potential_prefill_blocks,
            active_decode_blocks=load.active_decode_blocks,
            incoming_decode_blocks=incoming_decode_blocks,
            potential_decode_blocks=potential_decode_blocks,
            active_requests=load.active_requests,
            active_request_cost=active_request_cost,
            cost=cost,
        )

    def _worker_load_locked(
        self,
        worker: WorkerKey,
    ) -> WorkerLoadView:
        request_ids = self._worker_requests.get(worker, ())
        accepted_prefill = sum(
            prompt.uncached_tokens
            for request_id in request_ids
            if self._requests[request_id].accepted
            for prompt_index, prompt in enumerate(
                self._requests[request_id].candidates[worker].prompts
            )
            if prompt_index not in self._requests[request_id].prefill_completed
            and prompt_index not in self._requests[request_id].released_prompts
        )
        pending_prefill = sum(
            prompt.uncached_tokens
            for request_id in request_ids
            if not self._requests[request_id].accepted
            for prompt_index, prompt in enumerate(
                self._requests[request_id].candidates[worker].prompts
            )
            if prompt_index not in self._requests[request_id].prefill_completed
            and prompt_index not in self._requests[request_id].released_prompts
        )
        tracked_prefill = accepted_prefill + pending_prefill
        tracked_blocks = len(self._worker_block_refs.get(worker, ()))
        accepted_requests = sum(
            len(self._requests[request_id].candidates[worker].prompts)
            - len(self._requests[request_id].released_prompts)
            for request_id in request_ids
            if self._requests[request_id].accepted
        )
        pending_requests = sum(
            len(self._requests[request_id].candidates[worker].prompts)
            - len(self._requests[request_id].released_prompts)
            for request_id in request_ids
            if not self._requests[request_id].accepted
        )
        tracked_requests = accepted_requests + pending_requests
        return WorkerLoadView(
            active_prefill_tokens=tracked_prefill,
            active_decode_blocks=tracked_blocks,
            active_requests=tracked_requests,
            tracked_prefill_tokens=tracked_prefill,
            tracked_decode_blocks=tracked_blocks,
            tracked_output_blocks=sum(
                len(self._prompt_output_keys(self._requests[request_id], prompt_index))
                for request_id in request_ids
                for prompt_index in range(
                    len(self._requests[request_id].candidates[worker].prompts)
                )
                if prompt_index not in self._requests[request_id].released_prompts
            ),
            tracked_requests=tracked_requests,
            pending_requests=pending_requests,
            accepted_requests=accepted_requests,
        )

    def _acquire_locked(self, request_id: str, load: RequestLoad) -> None:
        self._worker_requests[load.worker].add(request_id)
        block_refs = self._worker_block_refs[load.worker]
        for prompt in load.prompts:
            block_refs.update(prompt.block_keys)

    def _release_locked(self, request: _TrackedRequest) -> None:
        worker = request.worker
        request_ids = self._worker_requests.get(worker)
        if request_ids is not None:
            request_ids.discard(request.request_id)
            if not request_ids:
                self._worker_requests.pop(worker, None)
        block_refs = self._worker_block_refs.get(worker)
        if block_refs is None:
            return
        load = request.candidates[worker]
        for prompt_index, prompt in enumerate(load.prompts):
            if prompt_index in request.released_prompts:
                continue
            block_refs.subtract(prompt.block_keys)
            block_refs.subtract(self._prompt_output_keys(request, prompt_index))
        block_refs += Counter()
        if not block_refs:
            self._worker_block_refs.pop(worker, None)

    def _release_prompt_locked(
        self,
        request: _TrackedRequest,
        prompt_index: int,
    ) -> None:
        if prompt_index in request.released_prompts:
            return
        worker = request.worker
        load = request.candidates[worker]
        block_refs = self._worker_block_refs.get(worker)
        if block_refs is not None:
            block_refs.subtract(load.prompts[prompt_index].block_keys)
            block_refs.subtract(self._prompt_output_keys(request, prompt_index))
            block_refs += Counter()
            if not block_refs:
                self._worker_block_refs.pop(worker, None)
        request.prefill_completed.add(prompt_index)
        request.released_prompts.add(prompt_index)

    @staticmethod
    def _output_block_key(
        request_id: str,
        prompt_index: int,
        choice_index: int,
        block_index: int,
    ) -> bytes:
        identity = f"{request_id}:{prompt_index}:{choice_index}:{block_index}".encode()
        return b"output:" + identity

    @staticmethod
    def _prompt_output_keys(
        request: _TrackedRequest,
        prompt_index: int,
    ) -> tuple[bytes, ...]:
        return tuple(
            key
            for (owner_prompt, _), keys in request.output_blocks.items()
            if owner_prompt == prompt_index
            for key in keys
        )

    def _reassign_locked(self, request: _TrackedRequest, worker: WorkerKey) -> bool:
        if request.prefill_completed or request.accepted:
            return False
        replacement = request.candidates.get(worker)
        if replacement is None:
            removed = self._requests.pop(request.request_id)
            self._release_locked(removed)
            return False
        self._release_locked(request)
        request.worker = worker
        request.dispatched = True
        request.accepted = False
        request.last_touched_s = self._clock()
        self._acquire_locked(request.request_id, replacement)
        return True

    def _cleanup_expired_locked(self, now_s: float) -> tuple[str, ...]:
        expired = tuple(
            request_id
            for request_id, request in self._requests.items()
            if now_s - request.last_touched_s >= self.config.request_expiry_s
        )
        for request_id in expired:
            request = self._requests.pop(request_id)
            self._release_locked(request)
        return expired


class RequestLifecycleObserver:
    """Translate one proxied HTTP response into load-state transitions."""

    def __init__(
        self, tracker: LifecycleLoadTracker, max_frame_bytes: int = 65536
    ) -> None:
        self.tracker = tracker
        self.max_frame_bytes = max_frame_bytes
        self.request_id: str | None = None
        self.status: int | None = None
        self.streaming = False
        self.response_complete = False
        self.prompt_count = 1
        self.choices_per_prompt = 1
        self._prefill_completed: set[int] = set()
        self._finished_choices: set[int] = set()
        self._choice_tokens: dict[int, int] = defaultdict(int)
        self._sse_buffer = b""

    def bind(
        self,
        request_id: str | None,
        *,
        choices_per_prompt: int = 1,
    ) -> None:
        self.request_id = request_id
        if request_id is not None:
            self.prompt_count = self.tracker.request_prompt_count(request_id) or 1
        self.choices_per_prompt = max(1, choices_per_prompt)

    def dispatch(self, worker: WorkerKey) -> bool:
        if self.request_id is None:
            return False
        return self.tracker.mark_dispatched(self.request_id, worker)

    def observe(self, message: dict[str, object]) -> None:
        if self.request_id is None:
            return
        message_type = message.get("type")
        if message_type == "http.response.start":
            status = message.get("status")
            self.status = status if isinstance(status, int) else None
            if self.status is not None and self.status < 400:
                self.tracker.mark_accepted(self.request_id)
            headers = message.get("headers", ())
            self.streaming = any(
                isinstance(key, bytes)
                and isinstance(value, bytes)
                and key.lower() == b"content-type"
                and b"text/event-stream" in value
                for key, value in headers  # type: ignore[union-attr]
            )
            return
        if message_type != "http.response.body":
            return

        body = message.get("body", b"")
        body = body if isinstance(body, bytes) else b""
        if body:
            self.tracker.touch(self.request_id)
        if self.status is not None and self.status < 400:
            if self.streaming:
                self._observe_stream(body)
            elif body:
                self.tracker.mark_prefill_completed(self.request_id)
        if not bool(message.get("more_body", False)):
            self.response_complete = True

    def _observe_stream(self, chunk: bytes) -> None:
        self._sse_buffer += chunk
        self._sse_buffer = self._sse_buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in self._sse_buffer:
            frame, self._sse_buffer = self._sse_buffer.split(b"\n\n", 1)
            if len(frame) > self.max_frame_bytes:
                self._sse_buffer = b""
                return
            data = b"\n".join(
                line[5:].lstrip()
                for line in frame.split(b"\n")
                if line.startswith(b"data:")
            )
            if not data or data == b"[DONE]":
                continue
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                continue
            self._observe_payload(payload)
        if len(self._sse_buffer) > self.max_frame_bytes:
            self._sse_buffer = b""

    def _observe_payload(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_index = choice.get("index")
            if (
                choice_index is None
                and self.prompt_count == self.choices_per_prompt == 1
            ):
                choice_index = 0
            if not isinstance(choice_index, int) or isinstance(choice_index, bool):
                continue
            prompt_index = min(
                choice_index // self.choices_per_prompt,
                self.prompt_count - 1,
            )
            if self._choice_has_content(choice):
                self._mark_prefill_completed(prompt_index)
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list) and all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in token_ids
            ):
                if token_ids:
                    self._mark_prefill_completed(prompt_index)
                self._choice_tokens[choice_index] += len(token_ids)
                self.tracker.record_output_tokens(
                    self.request_id,
                    prompt_index,
                    self._choice_tokens[choice_index],
                    choice_index % self.choices_per_prompt,
                )
            if choice.get("finish_reason") is not None:
                self._finished_choices.add(choice_index)
                first_choice = prompt_index * self.choices_per_prompt
                if all(
                    index in self._finished_choices
                    for index in range(
                        first_choice,
                        first_choice + self.choices_per_prompt,
                    )
                ):
                    self.tracker.free_prompt(self.request_id, prompt_index)

    @staticmethod
    def _choice_has_content(choice: dict[str, object]) -> bool:
        delta = choice.get("delta")
        delta = delta if isinstance(delta, dict) else {}
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        values = [choice.get("text")]
        values.extend(
            container.get(key)
            for container in (delta, message)
            for key in ("content", "reasoning_content", "reasoning")
        )
        return any(isinstance(value, str) and value for value in values)

    def _mark_prefill_completed(self, prompt_index: int) -> None:
        if prompt_index in self._prefill_completed:
            return
        if self.tracker.mark_prefill_completed(self.request_id, prompt_index):
            self._prefill_completed.add(prompt_index)

    def finish(self) -> bool:
        if self.request_id is None:
            return False
        request_id = self.request_id
        self.request_id = None
        return self.tracker.free(request_id)
