# Multiscale runtime/protocol final fix report

Date: 2026-09-15

Branch: `feature/fly-crossy-v1`

Base reviewed: `3bbf6b4` (`fix: preserve websocket proxy authority`)

## Scope and outcome

This single TDD wave closes the five final-review blockers before biomechanics:

1. Browser resume now becomes locally usable only after successful resume admission and preserves acting state when an observation is still pending.
2. Every server frame is validated and serialized through one UTF-8 byte-bound path. Large neural messages split deterministically by both the 20,000-update limit and the 1,048,576-byte wire limit; oversized non-neural messages fail before socket write or sequence advance.
3. Browser message publication re-checks the captured connection generation and session after callback-producing state updates, preventing an old-session terminal result from leaking through a reentrant reconnect.
4. Neural keyframes/deltas now carry explicit revision and chunk metadata. Deltas name `baseRevision` and `revision`; client activity becomes visible only after complete atomic reconstruction. Sparse keyframes contain nonzero body-ID entries over a zero baseline, and an empty one-chunk keyframe clears activity. The server skeleton answers `request_keyframe` with an honest empty revision-0 keyframe and remains connected.
5. Session terminal results, including authoritative completion time, are fully built and validated before pending/completion state or outbound sequence changes.

## Root-cause verification

- `SimulationClient.resume()` only enqueued a frame; the server resumed silently, leaving the browser phase at `paused` and blocking `observe()`.
- `SessionSender.send()` used `WebSocket.send_json()` directly, outside `_load_frame()` and its 1 MiB check. A valid maximum-count neural payload can exceed 1 MiB because JSON numeric widths vary.
- `receive()` called `setState()` and then unconditionally published the captured message. A state subscriber could synchronously replace the connection between those operations.
- Neural frames had only `updates`; no revision/base/chunk identity existed, incomplete chunks were exposed as ordinary messages, and `_apply_message()` rejected every keyframe request as unsupported.
- `finish_action()` called `_complete_pending()` and reserved a sequence before Pydantic validated `completion_time`, so invalid terminal construction corrupted session state.

## RED evidence

### Resume usability

Command:

```text
node --experimental-strip-types --test --test-name-pattern='observe immediately after a successful local resume' tests/simulation-client.test.mjs
```

Observed expected failure:

```text
not ok 1 - client can observe immediately after a successful local resume
Expected values to be strictly equal: 'paused' !== 'ready'
tests 1; pass 0; fail 1
```

### Revision/chunk wire contract

Commands:

```text
cd python && .venv/bin/python -m pytest tests/test_protocol.py -q
node --experimental-strip-types --test --test-name-pattern='reconstructible revision|server variants' tests/simulation-protocol.test.mjs
```

Observed expected failures:

```text
Python: 2 failed, 9 passed
- revision/chunkIndex/chunkCount were rejected as extra fields
- unversioned neural_keyframe did not raise ValidationError
Node: 2 failed
- unexpected key revision
- missing expected exception for unversioned neural metadata
```

### Sparse nonzero keyframe entries

Commands:

```text
cd python && .venv/bin/python -m pytest tests/test_protocol.py -q -k 'reconstructible_revision'
node --experimental-strip-types --test --test-name-pattern='reconstructible revision' tests/simulation-protocol.test.mjs
```

Observed expected failures:

```text
Python: Failed: DID NOT RAISE ValidationError
Node: Missing expected exception
```

### Actual byte-bound splitting and non-neural fail-closed sender

Commands:

```text
cd python && .venv/bin/python -m pytest tests/test_protocol.py -q -k 'actual_byte_limit or split_deterministically'
cd python && .venv/bin/python -m pytest tests/test_server.py -q -k 'oversized_non_neural'
cd python && .venv/bin/python -m pytest tests/test_server.py -q -k 'chunks_a_large_neural_frame'
```

Observed expected failures:

```text
Protocol: 1 failed, 1 passed
- parse_server_message already rejected the actual oversized UTF-8 frame
- chunk_neural_frames was absent
Sender non-neural: 1 failed
- sender attempted send_json and advanced sequence instead of using a bounded serialization path
Sender neural: 1 failed
- SessionSender.send_neural was absent
```

The test fixture independently asserts that the compact serialized 20,000-update frame is greater than 1,048,576 bytes before checking rejection.

### Atomic keyframe/delta reconstruction and server recovery

Commands:

```text
node --experimental-strip-types --test --test-name-pattern='publishes a keyframe|chunked delta atomically|empty one-chunk' tests/simulation-client.test.mjs
cd python && .venv/bin/python -m pytest tests/test_server.py -q -k 'answers_keyframe_request'
```

Observed expected failures:

```text
Node: 3 failed; client.subscribeActivity is not a function
Python: 1 failed; WebSocketDisconnect code 4402 after request_keyframe
```

### Reentrant terminal publication

Command:

```text
node --experimental-strip-types --test --test-name-pattern='does not publish an old action result' tests/simulation-client.test.mjs
```

Observed expected failure:

```text
not ok 1 - client does not publish an old action result after a state subscriber reconnects
Expected []; received the old s-current1 action_result
tests 1; pass 0; fail 1
```

### Session terminal atomicity

Command:

```text
cd python && .venv/bin/python -m pytest tests/test_session.py -q -k 'invalid_terminal_completion'
```

Observed expected failure:

```text
assert session.pending is not None
actual: None
1 failed, 13 deselected
```

## GREEN evidence

Individual behavior checks after minimal implementations:

```text
resume: 1 passed
revision/chunk Python protocol: 11 passed
revision/chunk Node protocol: 2 passed
sparse nonzero keyframe Python: 1 passed
sparse nonzero keyframe Node: 1 passed
actual-byte parser and splitter: 2 passed
oversized non-neural sender: 1 passed
large neural sender chunking: 1 passed
atomic client reconstruction: 3 passed
server keyframe recovery: 1 passed
reentrant terminal publication: 1 passed
invalid terminal completion atomicity: 1 passed
```

Focused integration command:

```text
cd python && .venv/bin/python -m pytest tests/test_protocol.py tests/test_server.py tests/test_session.py -q
cd .. && node --experimental-strip-types --test tests/simulation-protocol.test.mjs tests/simulation-client.test.mjs
```

Output:

```text
Python: 47 passed, 28 warnings in 3.37s
Node: 32 tests, 32 passed, 0 failed
```

The first full chained verification caught a TypeScript-only narrowing issue after all 102 Node tests passed:

```text
src/simulation/client.ts: TS18047: neuralAssembly/completed is possibly null
```

The fix retained the already-established runtime guard and captured the selected assembly in a local non-null reference. Fresh final verification was then run from the beginning.

Final verification command:

```text
npm test && npm run build && (cd python && .venv/bin/python -m pytest -q)
```

Final output:

```text
Node: 102 tests, 102 passed, 0 failed
TypeScript: tsc --noEmit passed
Vite 8.3.0: production build completed; 37 modules transformed
Python: 198 passed, 28 warnings in 8.78s
Exit status: 0
```

## Files changed

- `python/fly_crossy/protocol.py`
- `python/fly_crossy/server.py`
- `python/fly_crossy/session.py`
- `python/tests/test_protocol.py`
- `python/tests/test_server.py`
- `python/tests/test_session.py`
- `src/simulation/client.ts`
- `src/simulation/protocol.ts`
- `tests/simulation-client.test.mjs`
- `tests/simulation-protocol.test.mjs`
- `.superpowers/sdd/2026-09-14-multiscale-runtime-protocol/final-fix-report.md`

## Self-review

- Protocol symmetry: Python and TypeScript require the same `revision`, `chunkIndex`, `chunkCount`, and delta `baseRevision` fields, the same 256 chunk ceiling, and the same nonzero-keyframe rule.
- Byte safety: `serialize_server_message()` is the single server serialization gate; it validates again, emits compact JSON, measures UTF-8 bytes, and rejects anything above 1,048,576. `SessionSender` no longer calls `send_json()`.
- Sequencing: sender sequence advances only after the socket accepts the bounded payload. Neural chunks receive contiguous sequences. Invalid terminal result construction does not reserve a sequence.
- Chunking: initial partitions respect 20,000 updates, oversized partitions split deterministically in halves until every real serialized frame fits, and empty logical keyframes produce exactly one empty chunk.
- Reconstruction: chunks are keyed by message type/base/revision, require consistent count/time and unique indexes/IDs, and publish one sorted sparse activity snapshot only after all indexes arrive. Deltas copy/apply atomically and zero-valued delta entries clear IDs.
- Reentrancy: both ready and general message publication paths verify the captured generation/session after state callbacks.
- Session recovery: `request_keyframe` passes session/episode/sequence authority and the skeleton returns only a zero-baseline empty revision, then continues handling later messages.
- Mutation check: removing the resume phase update, serializer call, byte split, generation guard, chunk metadata, full-assembly gate, empty clear, or prevalidation order causes a named regression test to fail.
- Scope preservation: existing origin, artifact provenance, backpressure, reset-reentrancy, and security paths remain covered by the full suites.
- `git diff --check` passed before final verification.

## Concerns and deferred items

- The production build still emits the pre-existing Vite warning for a 794.58 kB minified application chunk; release hardening already tracks code splitting or a measured threshold.
- Host Python 3.14 emits 28 FastAPI 0.116.1 `asyncio.iscoroutinefunction` deprecation warnings. The pinned runtime target remains Python 3.12; this was already deferred in the final review.
- CUDA runtime behavior was not exercised on this host and no new GPU claim is made.
- The pre-existing untracked `scripts/__pycache__/` directory was not modified, removed, or staged.
