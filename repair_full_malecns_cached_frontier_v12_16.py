from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def find_repo_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "python" / "fly_crossy").is_dir() and (candidate / "runs").exists():
            return candidate
    raise SystemExit("ERROR: run this from inside the fly-crossy-connectome repository.")


ROOT = find_repo_root()
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from fly_crossy.env import (  # noqa: E402
    CONTENT_ROWS_PER_GROUP,
    GROUP_ROWS,
    OPENING_ROWS,
    GameState,
    step_game,
)
from fly_crossy.schema import ACTION_ORDER, Action  # noqa: E402
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame  # noqa: E402
from fly_crossy.v2.preference_distill import planner_action_preferences  # noqa: E402
from fly_crossy.v4.exact_seed_branch_curriculum import load_decoder_checkpoint  # noqa: E402
from fly_crossy.v4.train_expo_specialist import (  # noqa: E402
    default_flyhard_root,
    resize_rgb,
    variant_state,
)
from fly_crossy.v5.sensory_core_adaptation import (  # noqa: E402
    SensoryPlasticPolicy,
    _load_base_policy,
    load_adaptation_checkpoint,
    save_checkpoint,
)

VERSION = "crossy-full-malecns-readout-cached-frontier-12.16"
PREVIOUS_VERSION = "crossy-full-malecns-readout-cached-frontier-12.15"
ACTION_TO_INDEX = {action.value: i for i, action in enumerate(ACTION_ORDER)}
OPPOSITE = {Action.LEFT: Action.RIGHT, Action.RIGHT: Action.LEFT}
SEARCH_ACTIONS = (
    Action.FORWARD,
    Action.LEFT,
    Action.RIGHT,
    Action.WAIT,
    Action.BACKWARD,
)


@dataclass
class TraceStep:
    index: int
    state: GameState
    feature: np.ndarray
    logits: np.ndarray
    student_action: int
    next_terminal: str | None
    score_before: float
    score_after: float
    blocked: bool


def next_recovery_row(row: int) -> int:
    if row < OPENING_ROWS:
        return OPENING_ROWS + CONTENT_ROWS_PER_GROUP
    group_index = math.floor((row - OPENING_ROWS) / GROUP_ROWS)
    recovery = OPENING_ROWS + group_index * GROUP_ROWS + CONTENT_ROWS_PER_GROUP
    return recovery if row < recovery else recovery + GROUP_ROWS


def state_key(state: GameState) -> tuple[int, int, float, str]:
    return (
        int(state.step),
        int(state.fly.row),
        round(float(state.fly.column), 6),
        state.previous_action.value,
    )


def astar_teacher_action(
    state: GameState,
    *,
    search_limits: list[int],
    max_backtrack_rows: int,
    max_expansions: int,
    target_row_override: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return the first A* action plus the complete discovered action corridor.

    V12.8 keeps the original first-action interface so all repair-frontier logic
    remains compatible, but it also reconstructs the winning A* path.  The path
    is used by the optional corridor bundle to constrain several consecutive
    readout decisions in one transaction instead of hoping closed loop follows
    the remainder of the route after a one-step surgery.
    """
    local_target_row = next_recovery_row(int(state.fly.row))
    target_row = (
        max(local_target_row, int(target_row_override))
        if target_row_override is not None
        else local_target_row
    )
    minimum_row = int(state.fly.row) - max_backtrack_rows

    for search_limit in search_limits:
        counter = 0
        heap: list[
            tuple[int, int, int, int, float, int, GameState, Action | None]
        ] = []
        h0 = max(0, target_row - int(state.fly.row))
        heapq.heappush(
            heap,
            (h0, 0, 0, 0, abs(float(state.fly.column)), counter, state, None),
        )
        # Node IDs are the heap tie-break counters.  Keeping parent pointers is
        # much cheaper than copying a growing action tuple into every A* node.
        parents: dict[int, tuple[int | None, Action | None]] = {0: (None, None)}
        best: dict[tuple[int, int, float, str], tuple[int, int]] = {
            state_key(state): (0, 0)
        }
        expansions = 0

        while heap:
            _, g, reversals, waits, _, node_id, current, first_action = heapq.heappop(heap)

            if int(current.fly.row) >= target_row and first_action is not None:
                actions_rev: list[Action] = []
                cursor = int(node_id)
                while cursor != 0:
                    parent_id, action = parents[cursor]
                    if parent_id is None or action is None:
                        break
                    actions_rev.append(action)
                    cursor = int(parent_id)
                plan = list(reversed(actions_rev))
                return ACTION_TO_INDEX[first_action.value], {
                    "method": "astar",
                    "targetRecoveryRow": target_row,
                    "localRecoveryRow": local_target_row,
                    "targetRowOverride": (
                        int(target_row_override) if target_row_override is not None else None
                    ),
                    "searchLimit": search_limit,
                    "expansions": expansions,
                    "planLength": len(plan),
                    "planActions": [action.value for action in plan],
                }

            if g >= search_limit:
                continue

            expansions += 1
            if expansions > max_expansions:
                break

            for action in SEARCH_ACTIONS:
                result = step_game(current, action)
                candidate = result.state
                if candidate.terminal is not None:
                    continue
                if any(event.get("type") == "blocked" for event in result.events):
                    continue
                if int(candidate.fly.row) < minimum_row:
                    continue

                next_g = g + 1
                reversal = int(
                    current.previous_action in OPPOSITE
                    and OPPOSITE[current.previous_action] == action
                )
                next_reversals = reversals + reversal
                next_waits = waits + int(action == Action.WAIT)
                key = state_key(candidate)
                quality = (next_reversals, next_waits)
                old = best.get(key)
                if old is not None and old <= quality:
                    continue
                best[key] = quality

                first = action if first_action is None else first_action
                h = max(0, target_row - int(candidate.fly.row))
                counter += 1
                parents[counter] = (int(node_id), action)
                heapq.heappush(
                    heap,
                    (
                        next_g + h,
                        next_g,
                        next_reversals,
                        next_waits,
                        abs(float(candidate.fly.column)),
                        counter,
                        candidate,
                        first,
                    ),
                )

    values, best_action, safe = planner_action_preferences(state, depth=8)
    return int(best_action), {
        "method": "planner-depth8-fallback",
        "targetRecoveryRow": target_row,
        "localRecoveryRow": local_target_row,
        "targetRowOverride": (
            int(target_row_override) if target_row_override is not None else None
        ),
        "searchLimit": None,
        "expansions": 0,
        "planLength": 1,
        "planActions": [ACTION_ORDER[int(best_action)].value],
        "immediateSafe": safe.tolist(),
        "values": values.tolist(),
    }

def readout_feature(decoder: torch.nn.Module, motor: torch.Tensor) -> torch.Tensor:
    normalized = (motor - decoder.mean) / decoder.std
    pieces = [normalized]
    if decoder.hidden is not None:
        pieces.append(F.gelu(decoder.hidden(normalized)))
    pieces.append(torch.ones((len(motor), 1), dtype=motor.dtype, device=motor.device))
    return torch.cat(pieces, dim=1)


def decoder_virtual_weights(decoder: torch.nn.Module) -> torch.Tensor:
    pieces = [decoder.linear.weight.detach()]
    if decoder.residual is not None:
        pieces.append(decoder.residual.weight.detach())
    bias = decoder.linear.bias.detach()
    if decoder.residual is not None:
        bias = bias + decoder.residual.bias.detach()
    pieces.append(bias[:, None])
    return torch.cat(pieces, dim=1)


def apply_virtual_delta(
    decoder: torch.nn.Module,
    delta: torch.Tensor,
    *,
    alpha: float,
) -> None:
    # `minimal_readout_delta` solves the regularized linear system on CPU in
    # float64 for numerical stability. The decoder itself normally lives on
    # CUDA, so explicitly move/cast the solved update before touching weights.
    device = decoder.linear.weight.device
    dtype = decoder.linear.weight.dtype
    delta = delta.to(device=device, dtype=dtype)

    offset = 0
    motor_dim = decoder.motor_dim
    with torch.no_grad():
        decoder.linear.weight.add_(
            float(alpha) * delta[:, offset : offset + motor_dim]
        )
        offset += motor_dim
        if decoder.residual is not None:
            width = decoder.residual.weight.shape[1]
            decoder.residual.weight.add_(
                float(alpha) * delta[:, offset : offset + width]
            )
            offset += width
        decoder.linear.bias.add_(float(alpha) * delta[:, offset])


def snapshot_decoder(decoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in decoder.state_dict().items()
    }


def restore_decoder(
    decoder: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    decoder.load_state_dict(state, strict=True)


def _cpu_neural(neural: torch.Tensor) -> torch.Tensor:
    """Store an exact float32 recurrent-state snapshot in host RAM."""
    return neural.detach().to(device="cpu", dtype=torch.float32).clone()


def rollout_trace(
    *,
    policy: SensoryPlasticPolicy,
    decoder: torch.nn.Module,
    seed: str,
    device: torch.device,
    max_steps: int,
    stall_abort: int,
    quality_anchor_steps: int | None = None,
    quality_anchor_blocked: int | None = None,
    max_marginal_blocked_rate: float | None = None,
    marginal_block_warmup_steps: int = 0,
    cache_stride: int = 16,
) -> tuple[dict[str, Any], list[TraceStep], dict[int, torch.Tensor]]:
    """Full reference rollout plus sparse recurrent-state checkpoints.

    Cache entries are the MaleCNS recurrent state *before* the observation at
    the indexed step. V12 pays for this full rollout only once per fresh
    committed source (or not at all when --resume restores run_state.pt).
    """
    state = variant_state(seed, phase_offset=0.0, initial_column=0)
    neural = policy.zero_state(1, device=device)

    trace: list[TraceStep] = []
    neural_cache: dict[int, torch.Tensor] = {0: _cpu_neural(neural)}
    actions: Counter[str] = Counter()
    no_progress = 0
    max_no_progress = 0
    blocked = 0
    blocked_streak = 0
    max_blocked_streak = 0
    ping_pong = 0
    previous_actions: list[Action] = []
    stalled = False
    stall_tail_length = 0
    quality_aborted = False
    quality_abort_step: int | None = None
    quality_allowed_marginal_blocked: int | None = None

    policy.eval()
    decoder.eval()
    cache_stride = max(1, int(cache_stride))

    with torch.inference_mode():
        for index in range(max_steps):
            if state.terminal is not None:
                break

            if index not in neural_cache and index % cache_stride == 0:
                neural_cache[index] = _cpu_neural(neural)

            image_np = resize_rgb(render_crossy_neural_frame(state))
            image = torch.as_tensor(
                image_np[None],
                dtype=torch.float32,
                device=device,
            )
            neural = policy.step_state(neural, image)
            motor = policy.motor_state(neural)
            logits_t = decoder(motor)
            feature_t = readout_feature(decoder, motor)
            action_index = int(logits_t[0].argmax().item())
            action = ACTION_ORDER[action_index]

            before = state
            result = step_game(state, action)
            next_state = result.state
            is_blocked = any(event.get("type") == "blocked" for event in result.events)

            trace.append(
                TraceStep(
                    index=index,
                    state=state,
                    feature=feature_t[0].detach().cpu().numpy().astype(np.float64),
                    logits=logits_t[0].detach().cpu().numpy().astype(np.float64),
                    student_action=action_index,
                    next_terminal=next_state.terminal,
                    score_before=float(before.score),
                    score_after=float(next_state.score),
                    blocked=is_blocked,
                )
            )

            actions[action.value] += 1
            if is_blocked:
                blocked += 1
                blocked_streak += 1
                max_blocked_streak = max(max_blocked_streak, blocked_streak)
            else:
                blocked_streak = 0

            if float(next_state.score) > float(before.score):
                no_progress = 0
            else:
                no_progress += 1
                max_no_progress = max(max_no_progress, no_progress)

            if len(previous_actions) >= 2:
                a, b = previous_actions[-2], previous_actions[-1]
                if a in OPPOSITE and OPPOSITE[a] == b and action == a:
                    ping_pong += 1
            previous_actions.append(action)
            state = next_state

            # The recurrent state after observation `index` is exactly the state
            # before observation `index + 1` because readout/actions do not feed
            # directly into the MaleCNS; they affect it only through the next game
            # observation. Cache sparse anchors for exact frontier reconstruction.
            next_index = index + 1
            if next_index % cache_stride == 0:
                neural_cache[next_index] = _cpu_neural(neural)

            if (
                quality_anchor_steps is not None
                and quality_anchor_blocked is not None
                and max_marginal_blocked_rate is not None
                and int(state.step) > int(quality_anchor_steps)
            ):
                extension_steps = int(state.step) - int(quality_anchor_steps)
                extension_blocked = max(0, blocked - int(quality_anchor_blocked))
                quality_allowed_marginal_blocked = allowed_rollout_marginal_blocked_actions(
                    float(max_marginal_blocked_rate),
                    extension_steps,
                    marginal_block_warmup_steps,
                )
                if extension_blocked > quality_allowed_marginal_blocked:
                    quality_aborted = True
                    quality_abort_step = index
                    break

            if state.terminal is None and no_progress >= stall_abort:
                stalled = True
                stall_tail_length = no_progress
                break

    steps = int(state.step)
    # Keep the final recurrent state too. It is useful if a future branch starts
    # immediately after the last executed action.
    neural_cache.setdefault(steps, _cpu_neural(neural))

    if quality_aborted:
        reason = "blocked-budget"
    elif stalled:
        reason = "stagnation"
    else:
        reason = state.terminal

    if quality_anchor_steps is not None and quality_anchor_blocked is not None:
        marginal_steps = max(0, steps - int(quality_anchor_steps))
        marginal_blocked = max(0, blocked - int(quality_anchor_blocked))
        marginal_blocked_rate = (
            marginal_blocked / marginal_steps if marginal_steps > 0 else 0.0
        )
    else:
        marginal_steps = 0
        marginal_blocked = 0
        marginal_blocked_rate = 0.0
    metrics = {
        "steps": steps,
        "score": float(state.score),
        "row": int(state.fly.row),
        "terminalReason": reason,
        "reachedStepLimit": bool(
            not stalled
            and not quality_aborted
            and state.terminal is None
            and steps >= max_steps
        ),
        "progressRate": float(state.score) / max(1, steps),
        "maxNoProgressStreak": max_no_progress,
        "blockedActions": blocked,
        "blockedRate": blocked / max(1, steps),
        "maxBlockedStreak": max_blocked_streak,
        "pingPongReturns": ping_pong,
        "actions": dict(sorted(actions.items())),
        "stalled": stalled,
        "stallTailLength": stall_tail_length,
        "qualityAborted": quality_aborted,
        "qualityAbortStep": quality_abort_step,
        "qualityAnchorSteps": quality_anchor_steps,
        "qualityAnchorBlocked": quality_anchor_blocked,
        "marginalSteps": marginal_steps,
        "marginalBlockedActions": marginal_blocked,
        "marginalBlockedRate": marginal_blocked_rate,
        "allowedMarginalBlockedActions": (
            allowed_marginal_blocked_actions(float(max_marginal_blocked_rate), marginal_steps)
            if max_marginal_blocked_rate is not None else quality_allowed_marginal_blocked
        ),
        "rolloutAllowedMarginalBlockedActions": quality_allowed_marginal_blocked,
        "marginalBlockWarmupSteps": int(marginal_block_warmup_steps),
    }
    return metrics, trace, neural_cache


def _decoder_logits_from_feature_batch(
    decoder: torch.nn.Module,
    features: np.ndarray,
) -> torch.Tensor:
    """Evaluate only the frozen readout on cached features.

    This exactly mirrors RecoveryDecoderV2.forward's two output branches but
    skips the 165k-neuron MaleCNS. Cached features already contain normalized
    motor activity and the GELU hidden activation; V12 never changes the
    normalizer or hidden layer.
    """
    device = decoder.linear.weight.device
    dtype = decoder.linear.weight.dtype
    feature_t = torch.as_tensor(features, dtype=dtype, device=device)
    motor_dim = int(decoder.motor_dim)
    normalized = feature_t[:, :motor_dim]
    logits = F.linear(normalized, decoder.linear.weight, decoder.linear.bias)
    if decoder.residual is not None:
        width = int(decoder.residual.weight.shape[1])
        hidden = feature_t[:, motor_dim : motor_dim + width]
        logits = logits + F.linear(hidden, decoder.residual.weight, decoder.residual.bias)
    return logits


def analytic_prefix_validation(
    *,
    decoder: torch.nn.Module,
    trace: list[TraceStep],
    frontier_index: int,
    target_action: int,
) -> tuple[int | None, int, np.ndarray]:
    """Find the first changed action with readout-only math, no MaleCNS replay."""
    if frontier_index < 0 or frontier_index >= len(trace):
        raise IndexError(frontier_index)
    features = np.stack(
        [step.feature for step in trace[: frontier_index + 1]], axis=0
    ).astype(np.float32, copy=False)
    with torch.inference_mode():
        logits_t = _decoder_logits_from_feature_batch(decoder, features)
        # Batch GEMM can differ from the original batch=1 decoder path by a few
        # last bits. Re-run numerically fragile rows one-at-a-time so an almost-tie
        # is never accepted merely because of batched accumulation order.
        top2 = torch.topk(logits_t, k=2, dim=1).values
        near_tie = (top2[:, 0] - top2[:, 1]).abs() < 1e-4
        for index in torch.nonzero(near_tie, as_tuple=False).flatten().tolist():
            logits_t[index : index + 1] = _decoder_logits_from_feature_batch(
                decoder, features[index : index + 1]
            )
    logits = logits_t.detach().cpu().numpy().astype(np.float64)
    actions = logits.argmax(axis=1)
    for index in range(frontier_index):
        if int(actions[index]) != int(trace[index].student_action):
            return index, int(actions[frontier_index]), logits
    frontier_action = int(actions[frontier_index])
    if frontier_action == int(trace[frontier_index].student_action):
        return None, frontier_action, logits
    if frontier_action != int(target_action):
        # It changed at the requested frontier, but to the wrong action.
        return frontier_index, frontier_action, logits
    return frontier_index, frontier_action, logits


def reconstruct_neural_before(
    *,
    policy: SensoryPlasticPolicy,
    trace: list[TraceStep],
    neural_cache: dict[int, torch.Tensor],
    frontier_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Restore exact recurrent state before frontier observation from nearest cache."""
    eligible = [index for index in neural_cache if index <= frontier_index]
    if not eligible:
        raise RuntimeError("neural cache has no state at or before frontier")
    anchor = max(eligible)
    neural = neural_cache[anchor].to(device=device, dtype=torch.float32).clone()
    replayed = 0
    policy.eval()
    with torch.inference_mode():
        for index in range(anchor, frontier_index):
            image_np = resize_rgb(render_crossy_neural_frame(trace[index].state))
            image = torch.as_tensor(image_np[None], dtype=torch.float32, device=device)
            neural = policy.step_state(neural, image)
            replayed += 1
    return neural, replayed


def _prefix_runtime_stats(trace: list[TraceStep], end_exclusive: int) -> dict[str, Any]:
    actions: Counter[str] = Counter()
    blocked = 0
    blocked_streak = 0
    max_blocked_streak = 0
    no_progress = 0
    max_no_progress = 0
    ping_pong = 0
    previous_actions: list[Action] = []
    for step in trace[:end_exclusive]:
        action = ACTION_ORDER[int(step.student_action)]
        actions[action.value] += 1
        if step.blocked:
            blocked += 1
            blocked_streak += 1
            max_blocked_streak = max(max_blocked_streak, blocked_streak)
        else:
            blocked_streak = 0
        if float(step.score_after) > float(step.score_before):
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)
        if len(previous_actions) >= 2:
            a, b = previous_actions[-2], previous_actions[-1]
            if a in OPPOSITE and OPPOSITE[a] == b and action == a:
                ping_pong += 1
        previous_actions.append(action)
    return {
        "actions": actions,
        "blocked": blocked,
        "blockedStreak": blocked_streak,
        "maxBlockedStreak": max_blocked_streak,
        "noProgress": no_progress,
        "maxNoProgress": max_no_progress,
        "pingPong": ping_pong,
        "previousActions": previous_actions[-2:],
    }


def rollout_from_frontier(
    *,
    policy: SensoryPlasticPolicy,
    decoder: torch.nn.Module,
    current_trace: list[TraceStep],
    current_cache: dict[int, torch.Tensor],
    frontier_index: int,
    expected_action: int,
    prefix_logits: np.ndarray,
    device: torch.device,
    max_steps: int,
    stall_abort: int,
    quality_anchor_steps: int,
    quality_anchor_blocked: int,
    max_marginal_blocked_rate: float,
    marginal_block_warmup_steps: int,
    cache_stride: int,
) -> tuple[dict[str, Any], list[TraceStep], dict[int, torch.Tensor], dict[str, Any]]:
    """Continue closed loop at the first changed action instead of replaying step 0.

    Prefix actions were already proven identical analytically. Therefore the game
    states and MaleCNS recurrent state up to the frontier are unchanged. We
    restore the recurrent state from a sparse exact cache and execute only the
    genuinely new suffix.
    """
    cache_stride = max(1, int(cache_stride))
    neural_before, recurrent_replay_steps = reconstruct_neural_before(
        policy=policy,
        trace=current_trace,
        neural_cache=current_cache,
        frontier_index=frontier_index,
        device=device,
    )

    # Prefix state objects/features are immutable. Only logits need refreshing
    # because the output readout changed while preserving their argmax actions.
    trial_trace: list[TraceStep] = []
    for index in range(frontier_index):
        old = current_trace[index]
        trial_trace.append(
            TraceStep(
                index=old.index,
                state=old.state,
                feature=old.feature,
                logits=prefix_logits[index].copy(),
                student_action=old.student_action,
                next_terminal=old.next_terminal,
                score_before=old.score_before,
                score_after=old.score_after,
                blocked=old.blocked,
            )
        )

    stats = _prefix_runtime_stats(current_trace, frontier_index)
    actions: Counter[str] = stats["actions"]
    blocked = int(stats["blocked"])
    blocked_streak = int(stats["blockedStreak"])
    max_blocked_streak = int(stats["maxBlockedStreak"])
    no_progress = int(stats["noProgress"])
    max_no_progress = int(stats["maxNoProgress"])
    ping_pong = int(stats["pingPong"])
    previous_actions: list[Action] = list(stats["previousActions"])
    stalled = False
    stall_tail_length = 0
    quality_aborted = False
    quality_abort_step: int | None = None
    quality_allowed_marginal_blocked: int | None = None

    state = current_trace[frontier_index].state
    candidate_cache = {
        int(index): tensor
        for index, tensor in current_cache.items()
        if int(index) <= frontier_index
    }
    candidate_cache[frontier_index] = _cpu_neural(neural_before)

    policy.eval()
    decoder.eval()
    with torch.inference_mode():
        # Process the frontier frame exactly once. This is also an exact runtime
        # assertion that cache reconstruction agrees with the analytical readout.
        image_np = resize_rgb(render_crossy_neural_frame(state))
        image = torch.as_tensor(image_np[None], dtype=torch.float32, device=device)
        neural = policy.step_state(neural_before, image)
        motor = policy.motor_state(neural)
        logits_t = decoder(motor)
        feature_t = readout_feature(decoder, motor)
        action_index = int(logits_t[0].argmax().item())
        if action_index != int(expected_action):
            raise RuntimeError(
                "cached frontier reconstruction disagrees with analytical prefix: "
                f"expected={ACTION_ORDER[expected_action].value}, "
                f"runtime={ACTION_ORDER[action_index].value}, frontier={frontier_index}"
            )

        action = ACTION_ORDER[action_index]
        before = state
        result = step_game(state, action)
        next_state = result.state
        is_blocked = any(event.get("type") == "blocked" for event in result.events)
        trial_trace.append(
            TraceStep(
                index=frontier_index,
                state=state,
                feature=feature_t[0].detach().cpu().numpy().astype(np.float64),
                logits=logits_t[0].detach().cpu().numpy().astype(np.float64),
                student_action=action_index,
                next_terminal=next_state.terminal,
                score_before=float(before.score),
                score_after=float(next_state.score),
                blocked=is_blocked,
            )
        )
        actions[action.value] += 1
        if is_blocked:
            blocked += 1
            blocked_streak += 1
            max_blocked_streak = max(max_blocked_streak, blocked_streak)
        else:
            blocked_streak = 0
        if float(next_state.score) > float(before.score):
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)
        if len(previous_actions) >= 2:
            a, b = previous_actions[-2], previous_actions[-1]
            if a in OPPOSITE and OPPOSITE[a] == b and action == a:
                ping_pong += 1
        previous_actions.append(action)
        state = next_state
        candidate_cache[frontier_index + 1] = _cpu_neural(neural)

        index = frontier_index
        while index + 1 < max_steps and state.terminal is None:
            # Apply the same early quality/stall aborts immediately after each
            # executed action, including the changed frontier action.
            if int(state.step) > int(quality_anchor_steps):
                extension_steps = int(state.step) - int(quality_anchor_steps)
                extension_blocked = max(0, blocked - int(quality_anchor_blocked))
                quality_allowed_marginal_blocked = allowed_rollout_marginal_blocked_actions(
                    float(max_marginal_blocked_rate),
                    extension_steps,
                    marginal_block_warmup_steps,
                )
                if extension_blocked > quality_allowed_marginal_blocked:
                    quality_aborted = True
                    quality_abort_step = index
                    break
            if state.terminal is None and no_progress >= stall_abort:
                stalled = True
                stall_tail_length = no_progress
                break

            index += 1
            if index % cache_stride == 0:
                candidate_cache[index] = _cpu_neural(neural)

            image_np = resize_rgb(render_crossy_neural_frame(state))
            image = torch.as_tensor(image_np[None], dtype=torch.float32, device=device)
            neural = policy.step_state(neural, image)
            motor = policy.motor_state(neural)
            logits_t = decoder(motor)
            feature_t = readout_feature(decoder, motor)
            action_index = int(logits_t[0].argmax().item())
            action = ACTION_ORDER[action_index]
            before = state
            result = step_game(state, action)
            next_state = result.state
            is_blocked = any(event.get("type") == "blocked" for event in result.events)
            trial_trace.append(
                TraceStep(
                    index=index,
                    state=state,
                    feature=feature_t[0].detach().cpu().numpy().astype(np.float64),
                    logits=logits_t[0].detach().cpu().numpy().astype(np.float64),
                    student_action=action_index,
                    next_terminal=next_state.terminal,
                    score_before=float(before.score),
                    score_after=float(next_state.score),
                    blocked=is_blocked,
                )
            )
            actions[action.value] += 1
            if is_blocked:
                blocked += 1
                blocked_streak += 1
                max_blocked_streak = max(max_blocked_streak, blocked_streak)
            else:
                blocked_streak = 0
            if float(next_state.score) > float(before.score):
                no_progress = 0
            else:
                no_progress += 1
                max_no_progress = max(max_no_progress, no_progress)
            if len(previous_actions) >= 2:
                a, b = previous_actions[-2], previous_actions[-1]
                if a in OPPOSITE and OPPOSITE[a] == b and action == a:
                    ping_pong += 1
            previous_actions.append(action)
            state = next_state

        # Match rollout_trace: evaluate abort conditions after the final action
        # when the loop ended because terminal/max step rather than an earlier check.
        last_index = len(trial_trace) - 1
        if not quality_aborted and int(state.step) > int(quality_anchor_steps):
            extension_steps = int(state.step) - int(quality_anchor_steps)
            extension_blocked = max(0, blocked - int(quality_anchor_blocked))
            quality_allowed_marginal_blocked = allowed_rollout_marginal_blocked_actions(
                float(max_marginal_blocked_rate),
                extension_steps,
                marginal_block_warmup_steps,
            )
            if extension_blocked > quality_allowed_marginal_blocked:
                quality_aborted = True
                quality_abort_step = last_index
        if not quality_aborted and state.terminal is None and no_progress >= stall_abort:
            stalled = True
            stall_tail_length = no_progress

    steps = int(state.step)
    candidate_cache.setdefault(steps, _cpu_neural(neural))
    if quality_aborted:
        reason = "blocked-budget"
    elif stalled:
        reason = "stagnation"
    else:
        reason = state.terminal

    marginal_steps = max(0, steps - int(quality_anchor_steps))
    marginal_blocked = max(0, blocked - int(quality_anchor_blocked))
    marginal_blocked_rate = (
        marginal_blocked / marginal_steps if marginal_steps > 0 else 0.0
    )
    metrics = {
        "steps": steps,
        "score": float(state.score),
        "row": int(state.fly.row),
        "terminalReason": reason,
        "reachedStepLimit": bool(
            not stalled
            and not quality_aborted
            and state.terminal is None
            and steps >= max_steps
        ),
        "progressRate": float(state.score) / max(1, steps),
        "maxNoProgressStreak": max_no_progress,
        "blockedActions": blocked,
        "blockedRate": blocked / max(1, steps),
        "maxBlockedStreak": max_blocked_streak,
        "pingPongReturns": ping_pong,
        "actions": dict(sorted(actions.items())),
        "stalled": stalled,
        "stallTailLength": stall_tail_length,
        "qualityAborted": quality_aborted,
        "qualityAbortStep": quality_abort_step,
        "qualityAnchorSteps": quality_anchor_steps,
        "qualityAnchorBlocked": quality_anchor_blocked,
        "marginalSteps": marginal_steps,
        "marginalBlockedActions": marginal_blocked,
        "marginalBlockedRate": marginal_blocked_rate,
        "allowedMarginalBlockedActions": allowed_marginal_blocked_actions(
            float(max_marginal_blocked_rate), marginal_steps
        ),
        "rolloutAllowedMarginalBlockedActions": quality_allowed_marginal_blocked,
        "marginalBlockWarmupSteps": int(marginal_block_warmup_steps),
    }
    runtime = {
        "recurrentReplaySteps": recurrent_replay_steps,
        "suffixMaleCNSSteps": max(0, len(trial_trace) - frontier_index),
        "prefixMaleCNSStepsAvoided": frontier_index,
    }
    return metrics, trial_trace, candidate_cache, runtime



def build_forced_corridor_features(
    *,
    policy: SensoryPlasticPolicy,
    decoder: torch.nn.Module,
    current_trace: list[TraceStep],
    current_cache: dict[int, torch.Tensor],
    frontier_index: int,
    plan_actions: list[int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Teacher-force a short A* corridor and cache its readout features.

    Only the decoder output layers are ever changed by surgery.  Therefore the
    normalized motor/hidden readout feature at a teacher-forced game state is
    independent of those output-layer changes.  We can compute the exact
    MaleCNS features for the corridor once, then solve several readout decisions
    jointly while preserving the historical prefix.
    """
    if not plan_actions:
        return [], {"valid": False, "reason": "empty-corridor"}

    neural_before, recurrent_replay_steps = reconstruct_neural_before(
        policy=policy,
        trace=current_trace,
        neural_cache=current_cache,
        frontier_index=frontier_index,
        device=device,
    )
    neural = neural_before
    state = current_trace[frontier_index].state
    entries: list[dict[str, Any]] = []
    invalid_reason: str | None = None

    policy.eval()
    decoder.eval()
    with torch.inference_mode():
        for offset, action_index in enumerate(plan_actions):
            if state.terminal is not None:
                invalid_reason = f"terminal-before-corridor-step-{offset}"
                break

            image_np = resize_rgb(render_crossy_neural_frame(state))
            image = torch.as_tensor(image_np[None], dtype=torch.float32, device=device)
            neural = policy.step_state(neural, image)
            motor = policy.motor_state(neural)
            feature_t = readout_feature(decoder, motor)
            feature = feature_t[0].detach().cpu().numpy().astype(np.float64)

            action = ACTION_ORDER[int(action_index)]
            result = step_game(state, action)
            blocked = any(event.get("type") == "blocked" for event in result.events)
            entries.append(
                {
                    "offset": int(offset),
                    "stateStep": int(state.step),
                    "row": int(state.fly.row),
                    "column": float(state.fly.column),
                    "actionIndex": int(action_index),
                    "action": action.value,
                    "feature": feature,
                    "blocked": bool(blocked),
                    "nextTerminal": result.state.terminal,
                }
            )
            if blocked:
                invalid_reason = f"blocked-at-corridor-step-{offset}"
                break
            if result.state.terminal is not None:
                invalid_reason = f"terminal-at-corridor-step-{offset}:{result.state.terminal}"
                break
            state = result.state

    valid = invalid_reason is None and len(entries) == len(plan_actions)
    return entries, {
        "valid": bool(valid),
        "reason": invalid_reason,
        "requestedActions": len(plan_actions),
        "builtActions": len(entries),
        "recurrentReplaySteps": int(recurrent_replay_steps),
        "finalRow": int(state.fly.row),
        "finalScore": float(state.score),
    }


def apply_corridor_bundle(
    *,
    decoder: torch.nn.Module,
    corridor_entries: list[dict[str, Any]],
    base_protection: np.ndarray,
    protection_strength: float,
    margin: float,
    alpha: float,
) -> dict[str, Any]:
    """Impose several consecutive corridor actions in one readout transaction.

    Each later surgery is solved in the exact nullspace of both the protected
    historical prefix and all earlier corridor features.  Thus later edits
    cannot undo corridor decisions already installed.  The caller still
    performs full analytic prefix validation and a normal closed-loop rollout
    before any bridge or commit can be accepted.
    """
    if not corridor_entries:
        raise RuntimeError("corridor bundle has no entries")

    before_virtual = decoder_virtual_weights(decoder).detach().cpu().double()
    prior_features: list[np.ndarray] = []
    segment_reports: list[dict[str, Any]] = []

    for entry in corridor_entries:
        feature = np.asarray(entry["feature"], dtype=np.float64)
        pieces: list[np.ndarray] = []
        if len(base_protection):
            pieces.append(base_protection.astype(np.float64, copy=False))
        if prior_features:
            pieces.append(np.stack(prior_features, axis=0).astype(np.float64, copy=False))
        if pieces:
            protection = np.concatenate(pieces, axis=0)
        else:
            protection = np.zeros((0, len(feature)), dtype=np.float64)

        with torch.inference_mode():
            logits_t = _decoder_logits_from_feature_batch(
                decoder, feature[None].astype(np.float32, copy=False)
            )
        logits = logits_t[0].detach().cpu().numpy().astype(np.float64)
        delta, report = minimal_readout_delta(
            feature=feature,
            logits=logits,
            target_action=int(entry["actionIndex"]),
            protection=protection,
            protection_strength=protection_strength,
            margin=margin,
        )
        apply_virtual_delta(decoder, delta, alpha=alpha)
        segment_reports.append(
            {
                "offset": int(entry["offset"]),
                "stateStep": int(entry["stateStep"]),
                "row": int(entry["row"]),
                "targetAction": str(entry["action"]),
                "deltaNorm": float(report["deltaNorm"]),
                "protectedStates": int(report["protectedStates"]),
                "protectedRank": int(report["protectedRank"]),
                "nullspaceDimensionLowerBound": int(report["nullspaceDimensionLowerBound"]),
                "nullspaceNoveltyRatio": float(report["nullspaceNoveltyRatio"]),
            }
        )
        prior_features.append(feature)

    features = np.stack(
        [np.asarray(entry["feature"], dtype=np.float32) for entry in corridor_entries], axis=0
    )
    with torch.inference_mode():
        logits_t = _decoder_logits_from_feature_batch(decoder, features)
    predicted_indices = logits_t.argmax(dim=1).detach().cpu().tolist()
    target_indices = [int(entry["actionIndex"]) for entry in corridor_entries]
    matched = sum(int(p == t) for p, t in zip(predicted_indices, target_indices))

    after_virtual = decoder_virtual_weights(decoder).detach().cpu().double()
    aggregate_norm = float(torch.linalg.vector_norm(after_virtual - before_virtual).item())
    last = segment_reports[-1]
    return {
        "featureDim": int(features.shape[1]),
        "protectedStates": int(last["protectedStates"]),
        "protectedRank": int(last["protectedRank"]),
        "nullspaceDimensionLowerBound": int(last["nullspaceDimensionLowerBound"]),
        "margin": float(margin),
        "deltaNorm": aggregate_norm,
        "corridorLength": len(corridor_entries),
        "corridorMatchedActions": int(matched),
        "corridorTargetActions": [ACTION_ORDER[index].value for index in target_indices],
        "corridorPredictedActions": [ACTION_ORDER[int(index)].value for index in predicted_indices],
        "corridorAllMatched": bool(matched == len(target_indices)),
        "segmentReports": segment_reports,
        "legacyProtectionStrengthArgument": float(protection_strength),
    }

def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)



def _rebuild_root_sweep_progress(
    repairs: list[dict[str, Any]],
    *,
    terminal_rescue_lookback: int,
) -> tuple[list[int], int]:
    """Recover persistent root-sweep progress from a V12.15 repair history.

    V12.15 accidentally cleared root_frontier_attempts/lookback whenever it entered
    a tentative bridge or backtracked. The report still records every root rejection
    and every window expansion, so V12.16 can reconstruct the committed-root search
    progress without re-running those exhausted frontiers.
    """
    last_commit = -1
    for i, item in enumerate(repairs):
        if item.get("decision") == "commit-score-gain":
            last_commit = i

    attempts: list[int] = []
    seen: set[int] = set()
    lookback = max(1, int(terminal_rescue_lookback))
    for item in repairs[last_commit + 1 :]:
        decision = item.get("decision")
        if decision == "root-frontier-rejected":
            for raw in item.get("rootFrontierAttempts") or []:
                value = int(raw)
                if value not in seen:
                    seen.add(value)
                    attempts.append(value)
            if item.get("frontierStep") is not None:
                value = int(item["frontierStep"])
                if value not in seen:
                    seen.add(value)
                    attempts.append(value)
            lookback = max(lookback, int(item.get("rootLookback") or lookback))
        elif decision == "root-sweep-expand":
            lookback = max(lookback, int(item.get("newLookback") or lookback))

    return attempts, lookback


def save_run_state(
    *,
    path: Path,
    signature: dict[str, Any],
    decoder: torch.nn.Module,
    current_metrics: dict[str, Any],
    current_trace: list[TraceStep],
    current_cache: dict[int, torch.Tensor],
    committed_decoder: dict[str, torch.Tensor],
    committed_metrics: dict[str, Any],
    committed_trace: list[TraceStep],
    committed_cache: dict[int, torch.Tensor],
    baseline_metrics: dict[str, Any],
    quality_anchor_steps: int,
    quality_anchor_blocked: int,
    bridge_depth: int,
    bridge_frontiers: list[int],
    bridge_path: list[dict[str, Any]],
    banned_choices: dict[tuple[int, int, float, str], set[tuple[int, str, str, str]]],
    search_restarts: int,
    root_frontier_lookback_current: int,
    root_frontier_attempts: list[int],
    repairs: list[dict[str, Any]],
    next_repair_index: int,
    success: bool,
    elapsed_seconds: float,
) -> None:
    _atomic_torch_save(
        {
            "version": VERSION,
            "signature": signature,
            "decoderState": snapshot_decoder(decoder),
            "currentMetrics": current_metrics,
            "currentTrace": current_trace,
            "currentCache": current_cache,
            "committedDecoder": committed_decoder,
            "committedMetrics": committed_metrics,
            "committedTrace": committed_trace,
            "committedCache": committed_cache,
            "baselineMetrics": baseline_metrics,
            "qualityAnchorSteps": quality_anchor_steps,
            "qualityAnchorBlocked": quality_anchor_blocked,
            "bridgeDepth": bridge_depth,
            "bridgeFrontiers": bridge_frontiers,
            "bridgePath": bridge_path,
            "bannedChoices": banned_choices,
            "searchRestarts": search_restarts,
            "rootFrontierLookbackCurrent": root_frontier_lookback_current,
            "rootFrontierAttempts": root_frontier_attempts,
            "repairs": repairs,
            "nextRepairIndex": next_repair_index,
            "success": success,
            "elapsedSeconds": elapsed_seconds,
        },
        path,
    )


def load_run_state(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _choice_signature(
    action_index: int,
    strength: float,
    alpha: float,
    source: str = "single",
) -> tuple[int, str, str, str]:
    return (
        int(action_index),
        str(source),
        f"{float(strength):.12g}",
        f"{float(alpha):.12g}",
    )


def _derive_bridge_path(
    repairs: list[dict[str, Any]],
    trace: list[TraceStep],
) -> list[dict[str, Any]]:
    """Reconstruct the current tentative path when importing a V12 autosave."""
    path: list[dict[str, Any]] = []
    action_index = {action.value: i for i, action in enumerate(ACTION_ORDER)}
    for item in repairs:
        decision = item.get("decision")
        if decision in {"commit-score-gain", "rollback-transaction", "search-backtrack"}:
            path = []
            continue
        if decision != "tentative-bridge":
            continue
        selected = item.get("selected") or {}
        frontier_index = int(item.get("frontierStep", -1))
        name = selected.get("targetAction")
        if frontier_index < 0 or frontier_index >= len(trace) or name not in action_index:
            continue
        strength = float(selected.get("protectionStrength", 1.0))
        alpha = float(selected.get("alpha", 1.0))
        path.append(
            {
                "stateKey": state_key(trace[frontier_index].state),
                "frontierIndex": frontier_index,
                "choice": _choice_signature(
                    action_index[name],
                    strength,
                    alpha,
                    str(selected.get("targetActionSource", "single")),
                ),
                "targetAction": name,
                "protectionStrength": strength,
                "alpha": alpha,
            }
        )
    return path


def load_teacher_reference(path: Path, *, expected_seed: str) -> dict[str, Any]:
    """Load the fixed-seed A* trajectory and index first step for each net score.

    The saved teacher has no blocked actions, so its score is exactly the cumulative
    FORWARD minus BACKWARD count. Using score rather than wall-clock state makes this
    a local difficulty reference, not an attempt to clone the teacher's timing.
    """
    if not path.is_file():
        raise SystemExit(f"ERROR: teacher trajectory not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    seed = str(payload.get("seed", ""))
    if seed != expected_seed:
        raise SystemExit(
            f"ERROR: teacher seed mismatch: trajectory={seed!r} expected={expected_seed!r}"
        )

    actions = list(payload.get("actions") or [])
    if not actions:
        raise SystemExit("ERROR: teacher trajectory contains no actions.")

    first_step_for_score: dict[int, int] = {0: 0}
    score = 0
    for step, action_name in enumerate(actions, start=1):
        if action_name == Action.FORWARD.value:
            score += 1
        elif action_name == Action.BACKWARD.value:
            score -= 1
        first_step_for_score.setdefault(score, step)

    summary = dict(payload.get("summary") or {})
    expected_score = int(round(float(summary.get("score", score))))
    if score != expected_score:
        raise SystemExit(
            "ERROR: teacher action sequence does not reproduce summary score: "
            f"actions={score} summary={expected_score}"
        )

    if int(summary.get("blockedActions", 0)) != 0:
        raise SystemExit("ERROR: teacher reference must have zero blocked actions.")

    return {
        "path": str(path.resolve()),
        "seed": seed,
        "steps": len(actions),
        "score": score,
        "firstStepForScore": first_step_for_score,
        "summary": summary,
    }


def teacher_local_gate(
    reference: dict[str, Any],
    *,
    committed_score: float,
    candidate_score: float,
    committed_steps: int,
    candidate_steps: int,
    max_slowdown: float,
    step_slack: int,
) -> dict[str, Any]:
    """Compare candidate extension cost with the teacher over the same score interval.

    A small additive slack prevents one- or two-row gains from being judged against an
    unrealistically tiny denominator when the student reaches that row at a different
    obstacle phase. Larger extensions are governed by the slowdown multiplier.
    """
    start_score = int(round(float(committed_score)))
    end_score = int(round(float(candidate_score)))
    score_gain = end_score - start_score
    extension_steps = max(0, int(candidate_steps) - int(committed_steps))
    first = reference["firstStepForScore"]

    if score_gain <= 0:
        return {
            "eligible": False,
            "reason": "no-score-gain",
            "scoreGain": score_gain,
            "candidateExtensionSteps": extension_steps,
            "teacherSteps": None,
            "slowdown": None,
            "allowedCandidateSteps": None,
        }
    if start_score not in first or end_score not in first:
        return {
            "eligible": False,
            "reason": "teacher-score-out-of-range",
            "scoreGain": score_gain,
            "candidateExtensionSteps": extension_steps,
            "teacherSteps": None,
            "slowdown": None,
            "allowedCandidateSteps": None,
        }

    teacher_steps = int(first[end_score]) - int(first[start_score])
    if teacher_steps <= 0:
        return {
            "eligible": False,
            "reason": "invalid-teacher-window",
            "scoreGain": score_gain,
            "candidateExtensionSteps": extension_steps,
            "teacherSteps": teacher_steps,
            "slowdown": None,
            "allowedCandidateSteps": None,
        }

    allowed_candidate_steps = max(
        int(math.ceil(float(max_slowdown) * teacher_steps - 1e-12)),
        int(teacher_steps + max(0, int(step_slack))),
    )
    slowdown = extension_steps / teacher_steps
    return {
        "eligible": bool(extension_steps <= allowed_candidate_steps),
        "reason": "ok" if extension_steps <= allowed_candidate_steps else "teacher-slowdown",
        "scoreGain": score_gain,
        "candidateExtensionSteps": extension_steps,
        "teacherSteps": teacher_steps,
        "slowdown": slowdown,
        "allowedCandidateSteps": allowed_candidate_steps,
        "maxSlowdown": float(max_slowdown),
        "stepSlack": int(step_slack),
        "teacherStartStep": int(first[start_score]),
        "teacherEndStep": int(first[end_score]),
    }


def allowed_marginal_blocked_actions(rate: float, steps: int) -> int:
    """Discrete blocked-action budget corresponding to a fractional rate."""
    if steps <= 0:
        return 0
    return int(math.ceil(float(rate) * int(steps) - 1e-12))


def allowed_rollout_marginal_blocked_actions(
    rate: float,
    steps: int,
    warmup_steps: int,
) -> int:
    """Rollout-only blocked budget with a short-window observation warmup.

    Early cumulative percentages are unstable: two blocked actions in the first
    few suffix steps can kill a branch before it has enough time to dilute to the
    same discrete budget that a healthy 15-20 step extension would receive.
    The warmup only delays the early-abort decision. Final commit quality still
    uses the real `steps` count through `allowed_marginal_blocked_actions`.
    """
    if steps <= 0:
        return 0
    effective_steps = max(int(steps), max(0, int(warmup_steps)))
    return allowed_marginal_blocked_actions(float(rate), effective_steps)


def marginal_block_quality(
    metrics: dict[str, Any],
    *,
    max_marginal_blocked_rate: float,
) -> dict[str, Any]:
    """Use one consistent discrete blocked-action budget for rollout and commit.

    A fractional threshold such as 7.5% cannot always be represented exactly by
    an integer event count. The rollout already uses ceil(rate * steps), so the
    final commit gate must use the same count budget rather than a stricter floor
    on short windows. Global blockedRate remains an independent safety floor.
    """
    steps = int(metrics.get("marginalSteps", 0))
    blocked = int(metrics.get("marginalBlockedActions", 0))
    rate = blocked / steps if steps > 0 else 0.0
    allowed = allowed_marginal_blocked_actions(
        float(max_marginal_blocked_rate), steps
    )
    eligible = bool(blocked <= allowed)
    strict_rate_ok = bool(rate <= float(max_marginal_blocked_rate) + 1e-12)
    return {
        "eligible": eligible,
        "steps": steps,
        "blocked": blocked,
        "allowed": allowed,
        "rate": rate,
        "strictRateOk": strict_rate_ok,
        "discreteToleranceUsed": bool(eligible and not strict_rate_ok),
        "budgetRule": "ceil(rate * steps), identical to rollout early-abort budget",
    }


def selection_key(metrics: dict[str, Any], target: int) -> tuple[float, ...]:
    # Score/progress first. Survival alone is not allowed to win.
    return (
        float(metrics["score"]),
        float(metrics["reachedStepLimit"]),
        float(metrics["progressRate"]),
        -float(metrics["maxNoProgressStreak"]),
        -float(metrics["pingPongReturns"]),
        -float(metrics["blockedActions"]),
        float(metrics["steps"]),
    )


def quality_success(
    metrics: dict[str, Any],
    target: int,
    *,
    max_no_progress: int,
    max_blocked_rate: float,
    min_progress_rate: float,
    max_marginal_blocked_rate: float,
) -> bool:
    marginal_quality = marginal_block_quality(
        metrics, max_marginal_blocked_rate=max_marginal_blocked_rate
    )
    return bool(
        metrics["reachedStepLimit"]
        and not bool(metrics.get("qualityAborted", False))
        and float(metrics["progressRate"]) >= min_progress_rate
        and int(metrics["maxNoProgressStreak"]) <= max_no_progress
        and float(metrics["blockedRate"]) <= max_blocked_rate
        and marginal_quality["eligible"]
    )


def _decision_margin(step: TraceStep) -> float:
    logits = np.asarray(step.logits, dtype=np.float64)
    action = int(step.student_action)
    if len(logits) <= 1:
        return float("inf")
    competitor = max(
        float(logits[index])
        for index in range(len(logits))
        if index != action
    )
    return float(logits[action]) - competitor


def choose_protection_indices(
    trace: list[TraceStep],
    *,
    frontier_index: int,
    recent: int,
    anchors: int,
    low_margin: int,
) -> list[int]:
    """Seed a compact exact-protection set.

    We protect:
    - the most recent prefix states,
    - evenly-spaced historical anchors,
    - the historically most fragile (lowest logit-margin) states.

    Any earlier action that still changes is discovered analytically from the
    cached readout features and added as an exact witness constraint later.
    """
    n = max(0, int(frontier_index))
    if n == 0:
        return []

    selected: set[int] = set()

    recent = max(0, int(recent))
    if recent:
        recent_start = max(0, n - recent)
        selected.update(range(recent_start, n))
    else:
        recent_start = n

    anchors = max(0, int(anchors))
    older_end = recent_start
    if anchors and older_end > 0:
        count = min(anchors, older_end)
        for value in np.linspace(0, older_end - 1, num=count):
            selected.add(int(round(float(value))))

    low_margin = max(0, int(low_margin))
    if low_margin:
        candidates = [index for index in range(n) if index not in selected]
        candidates.sort(key=lambda index: _decision_margin(trace[index]))
        selected.update(candidates[:low_margin])

    return sorted(selected)


def protection_matrix(
    trace: list[TraceStep],
    indices: list[int] | set[int],
    *,
    feature_dim: int,
) -> np.ndarray:
    ordered = sorted({int(index) for index in indices})
    if not ordered:
        return np.zeros((0, feature_dim), dtype=np.float64)
    return np.stack(
        [trace[index].feature for index in ordered],
        axis=0,
    ).astype(np.float64, copy=False)


def minimal_readout_delta(
    *,
    feature: np.ndarray,
    logits: np.ndarray,
    target_action: int,
    protection: np.ndarray,
    protection_strength: float,
    margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct a readout update in the exact nullspace of protected features.

    V12 keeps the compact active constraint set. Cached readout features verify
    every historical prefix action without rerunning the MaleCNS. If an
    unprotected earlier action changes, that exact feature is added as a witness
    constraint and the proposal is recomputed.
    """
    x = torch.as_tensor(feature, dtype=torch.float64)
    d = int(len(feature))

    if len(protection):
        X = torch.as_tensor(protection, dtype=torch.float64)
        # Compact SVD gives an orthonormal basis for the row-space of X.
        # Use a conservative numerical rank threshold.
        _, singular, vh = torch.linalg.svd(X, full_matrices=False)
        if len(singular):
            threshold = max(X.shape) * torch.finfo(torch.float64).eps * singular[0]
            rank = int((singular > threshold).sum().item())
        else:
            rank = 0
        if rank:
            row_basis = vh[:rank]
            projection = row_basis.T @ (row_basis @ x)
        else:
            projection = torch.zeros_like(x)
        q = x - projection
    else:
        X = torch.empty((0, d), dtype=torch.float64)
        singular = torch.empty(0, dtype=torch.float64)
        rank = 0
        q = x.clone()

    q_norm = float(torch.linalg.vector_norm(q).item())
    x_norm = float(torch.linalg.vector_norm(x).item())
    novelty_ratio = q_norm / max(x_norm, 1e-12)
    denom = float(torch.dot(q, x).item())

    if not math.isfinite(denom) or denom <= 1e-14 or novelty_ratio <= 1e-10:
        raise RuntimeError(
            "Fatal frontier readout feature lies numerically inside the protected "
            f"prefix span: rank={rank}, noveltyRatio={novelty_ratio:.3e}, denom={denom:.3e}"
        )

    delta = torch.zeros((len(ACTION_ORDER), d), dtype=torch.float64)
    virtual_logits = torch.as_tensor(logits, dtype=torch.float64).clone()

    # Raise the target above every competitor. Because q is in null(X), each
    # correction is exactly invisible on the protected prefix features.
    for _ in range(3):
        changed = False
        for competitor in range(len(ACTION_ORDER)):
            if competitor == target_action:
                continue
            needed = float(
                margin
                - (
                    virtual_logits[target_action].item()
                    - virtual_logits[competitor].item()
                )
            )
            if needed <= 0:
                continue
            direction = (needed / (2.0 * denom)) * q
            delta[target_action] += direction
            delta[competitor] -= direction
            virtual_logits = torch.as_tensor(logits, dtype=torch.float64) + delta @ x
            changed = True
        if not changed:
            break

    if len(X):
        protected_drift = X @ delta.T
        max_drift = float(protected_drift.abs().max().item())
        mean_drift = float(protected_drift.abs().mean().item())
    else:
        max_drift = 0.0
        mean_drift = 0.0

    return delta.float(), {
        "featureDim": d,
        "protectedStates": int(len(protection)),
        "protectedRank": rank,
        "nullspaceDimensionLowerBound": int(d - rank),
        "margin": float(margin),
        "frontierFeatureNorm": x_norm,
        "nullspaceNoveltyNorm": q_norm,
        "nullspaceNoveltyRatio": novelty_ratio,
        "denominator": denom,
        "deltaNorm": float(delta.norm().item()),
        "predictedTargetLogit": float(virtual_logits[target_action].item()),
        "predictedWinner": ACTION_ORDER[int(virtual_logits.argmax().item())].value,
        "predictedMaxProtectedLogitDrift": max_drift,
        "predictedMeanProtectedLogitDrift": mean_drift,
        # Kept for report compatibility; this algorithm does not use ridge strength.
        "legacyProtectionStrengthArgument": float(protection_strength),
    }


def corridor_match_status(
    decoder: torch.nn.Module,
    corridor_entries: list[dict[str, Any]] | None,
) -> tuple[int, bool, list[str]]:
    """Re-evaluate corridor actions after any extra guarded readout repair."""
    if not corridor_entries:
        return 0, True, []
    features = np.stack(
        [np.asarray(entry["feature"], dtype=np.float32) for entry in corridor_entries],
        axis=0,
    )
    with torch.inference_mode():
        logits_t = _decoder_logits_from_feature_batch(decoder, features)
    predicted = logits_t.argmax(dim=1).detach().cpu().tolist()
    target = [int(entry["actionIndex"]) for entry in corridor_entries]
    matched = sum(int(p == t) for p, t in zip(predicted, target))
    return (
        int(matched),
        bool(matched == len(target)),
        [ACTION_ORDER[int(index)].value for index in predicted],
    )


def repair_protected_witness(
    *,
    decoder: torch.nn.Module,
    trace: list[TraceStep],
    active_indices: set[int],
    witness_index: int,
    mandatory_features: np.ndarray,
    protection_strength: float,
    guard_margin: float,
) -> dict[str, Any]:
    """Restore one already-protected historical decision with an explicit margin.

    Exact-nullspace preservation is mathematically zero-drift, but the installed
    decoder weights are float32.  At extremely low-margin historical states, the
    final float32 add can flip an action by a few ulps even though that feature is
    already in the protected row-space.  V12.9 treats that as a numerical guard
    problem rather than declaring the candidate impossible.

    The diverged witness is temporarily removed from the zero-drift set so its
    original action can be reinforced.  Every other active historical feature,
    plus the requested frontier/corridor features, remains in the exact-nullspace
    protection set.  Only the existing decoder output layers are modified.
    """
    witness_index = int(witness_index)
    feature = np.asarray(trace[witness_index].feature, dtype=np.float64)
    other_indices = {int(i) for i in active_indices if int(i) != witness_index}
    historical = protection_matrix(
        trace,
        other_indices,
        feature_dim=len(feature),
    )
    pieces: list[np.ndarray] = []
    if len(historical):
        pieces.append(historical.astype(np.float64, copy=False))
    if len(mandatory_features):
        pieces.append(mandatory_features.astype(np.float64, copy=False))
    protection = (
        np.concatenate(pieces, axis=0)
        if pieces
        else np.zeros((0, len(feature)), dtype=np.float64)
    )

    with torch.inference_mode():
        logits_t = _decoder_logits_from_feature_batch(
            decoder,
            feature[None].astype(np.float32, copy=False),
        )
    logits = logits_t[0].detach().cpu().numpy().astype(np.float64)
    original_action = int(trace[witness_index].student_action)
    before_action = int(np.argmax(logits))
    before_margin = float(
        logits[original_action]
        - max(logits[i] for i in range(len(ACTION_ORDER)) if i != original_action)
    )

    delta, report = minimal_readout_delta(
        feature=feature,
        logits=logits,
        target_action=original_action,
        protection=protection,
        protection_strength=protection_strength,
        margin=float(guard_margin),
    )
    apply_virtual_delta(decoder, delta, alpha=1.0)

    with torch.inference_mode():
        after_t = _decoder_logits_from_feature_batch(
            decoder,
            feature[None].astype(np.float32, copy=False),
        )
    after = after_t[0].detach().cpu().numpy().astype(np.float64)
    after_action = int(np.argmax(after))
    after_margin = float(
        after[original_action]
        - max(after[i] for i in range(len(ACTION_ORDER)) if i != original_action)
    )
    return {
        "witnessIndex": witness_index,
        "targetAction": ACTION_ORDER[original_action].value,
        "beforeAction": ACTION_ORDER[before_action].value,
        "afterAction": ACTION_ORDER[after_action].value,
        "beforeMargin": before_margin,
        "afterMargin": after_margin,
        "guardMargin": float(guard_margin),
        "mandatoryFeatures": int(len(mandatory_features)),
        "deltaNorm": float(report["deltaNorm"]),
        "protectedStates": int(report["protectedStates"]),
        "protectedRank": int(report["protectedRank"]),
        "nullspaceDimensionLowerBound": int(report["nullspaceDimensionLowerBound"]),
        "nullspaceNoveltyRatio": float(report["nullspaceNoveltyRatio"]),
        "repaired": bool(after_action == original_action),
    }



def first_action_divergence(
    base_trace: list[TraceStep],
    candidate_trace: list[TraceStep],
) -> int | None:
    shared = min(len(base_trace), len(candidate_trace))
    for index in range(shared):
        if base_trace[index].student_action != candidate_trace[index].student_action:
            return index
    if len(base_trace) != len(candidate_trace):
        return shared
    return None


def choose_repair_frontier(
    *,
    trace: list[TraceStep],
    metrics: dict[str, Any],
    search_limits: list[int],
    max_backtrack_rows: int,
    max_expansions: int,
    min_frontier_index: int = 0,
    terminal_rescue_lookback: int = 12,
    excluded_frontiers: set[int] | None = None,
) -> tuple[int, int, dict[str, Any]] | None:
    if not trace:
        return None
    excluded_frontiers = excluded_frontiers or set()

    if metrics["terminalReason"] == "blocked-budget":
        index = int(metrics.get("qualityAbortStep") or (len(trace) - 1))
        index = max(0, min(index, len(trace) - 1))
        if not trace[index].blocked:
            for candidate in range(index, -1, -1):
                if trace[candidate].blocked:
                    index = candidate
                    break
        target_action, teacher = astar_teacher_action(
            trace[index].state,
            search_limits=search_limits,
            max_backtrack_rows=max_backtrack_rows,
            max_expansions=max_expansions,
        )
        teacher = {
            **teacher,
            "qualityRepair": True,
            "blockedBudgetViolation": True,
            "selectedFrontier": index,
            "marginalBlockedActions": metrics.get("marginalBlockedActions"),
            "allowedMarginalBlockedActions": metrics.get("allowedMarginalBlockedActions"),
        }
        return index, target_action, teacher

    if metrics["terminalReason"] not in {None, "stagnation"}:
        # V12.6 rescued terminal branches by scanning this suffix from newest
        # to oldest and choosing the *latest* A* disagreement. The 642/288
        # report showed that this still lets the decoder drift for several
        # autonomous steps before intervention (e.g. floor=643, selected=648).
        # V12.7 scans oldest-to-newest and repairs the *earliest* A* disagreement
        # after the last tentative bridge. This stitches the teacher corridor
        # before the branch enters a later blocked/terminal trap, while DFS still
        # owns changes to any already-selected bridge frontier.
        last_index = len(trace) - 1
        floor = max(0, int(min_frontier_index))
        lookback = max(1, int(terminal_rescue_lookback))
        floor = max(floor, last_index - lookback + 1)

        # V12.9 fixed the float32 protected-witness flips, but the 916/400
        # report exposed a different issue: rescue rewound to a state near row
        # 393 and A* only targeted that state's local next recovery row (394).
        # The action became exact and the historical prefix stayed intact, yet
        # the rewritten suffix finished at score 395 because the teacher goal
        # was behind progress the doomed branch had already earned. V12.10
        # targets the next recovery row *beyond* the terminal branch's achieved
        # progress so a rescue must actually recover the lost suffix.
        terminal_progress_row = max(
            int(metrics.get("row", trace[last_index].state.fly.row)),
            max(int(step.state.fly.row) for step in trace),
        )
        terminal_recovery_target = next_recovery_row(terminal_progress_row)

        for index in range(floor, last_index + 1):
            if int(index) in excluded_frontiers:
                continue
            target_action, teacher = astar_teacher_action(
                trace[index].state,
                search_limits=search_limits,
                max_backtrack_rows=max_backtrack_rows,
                max_expansions=max_expansions,
                target_row_override=terminal_recovery_target,
            )
            if teacher.get("method") != "astar":
                continue
            if int(target_action) == int(trace[index].student_action):
                continue
            result = step_game(trace[index].state, ACTION_ORDER[int(target_action)])
            if result.state.terminal is not None:
                continue
            if any(event.get("type") == "blocked" for event in result.events):
                continue
            teacher = {
                **teacher,
                "terminalRescue": bool(index < last_index),
                "terminalTailIndex": int(last_index),
                "selectedFrontier": int(index),
                "terminalRescueFloor": int(floor),
                "terminalRescueLookback": int(lookback),
                "terminalRescuePolicy": "earliest-astar-divergence",
                "terminalProgressRow": int(terminal_progress_row),
                "terminalRecoveryTargetRow": int(terminal_recovery_target),
                "terminalRescueTargetPolicy": "next-recovery-beyond-terminal-progress",
            }
            return index, target_action, teacher

        # No real A* rescue path remained in the allowed suffix. Preserve the
        # old final-state fallback so the safe-action sweep/backtracking logic
        # can still decide whether this branch has any local escape, while
        # keeping the same beyond-terminal target for consistency.
        index = last_index
        if int(index) in excluded_frontiers or int(index) < int(floor):
            return None
        target_action, teacher = astar_teacher_action(
            trace[index].state,
            search_limits=search_limits,
            max_backtrack_rows=max_backtrack_rows,
            max_expansions=max_expansions,
            target_row_override=terminal_recovery_target,
        )
        teacher = {
            **teacher,
            "terminalRescue": False,
            "terminalTailIndex": int(last_index),
            "selectedFrontier": int(index),
            "terminalRescueFloor": int(floor),
            "terminalRescueLookback": int(lookback),
            "terminalRescuePolicy": "earliest-astar-divergence",
            "terminalProgressRow": int(terminal_progress_row),
            "terminalRecoveryTargetRow": int(terminal_recovery_target),
            "terminalRescueTargetPolicy": "next-recovery-beyond-terminal-progress",
        }
        return index, target_action, teacher

    if metrics["terminalReason"] == "stagnation":
        tail = max(1, int(metrics.get("stallTailLength", 1)))
        start = max(0, len(trace) - tail)
        fallback: tuple[int, int, dict[str, Any]] | None = None

        for index in range(start, len(trace)):
            target_action, teacher = astar_teacher_action(
                trace[index].state,
                search_limits=search_limits,
                max_backtrack_rows=max_backtrack_rows,
                max_expansions=max_expansions,
            )
            if fallback is None and trace[index].blocked:
                fallback = (index, target_action, teacher)
            if target_action != trace[index].student_action:
                teacher = {
                    **teacher,
                    "stallRepair": True,
                    "stallTailStart": start,
                    "selectedFrontier": index,
                }
                return index, target_action, teacher

        if fallback is not None:
            index, target_action, teacher = fallback
            teacher = {
                **teacher,
                "stallRepair": True,
                "stallTailStart": start,
                "selectedFrontier": index,
                "fallbackBlockedFrontier": True,
            }
            return index, target_action, teacher

    return None


def frontier_action_candidates(
    *,
    state: GameState,
    student_action: int,
    teacher_action: int,
) -> list[tuple[int, str]]:
    """Return teacher first, then every other immediately safe local action.

    The A* action remains the preferred candidate. If that exact first move is
    incompatible with the current closed-loop readout, other safe local actions
    are tested instead of declaring the frontier unsalvageable.
    """
    candidates: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(action_index: int, source: str) -> None:
        if action_index == student_action or action_index in seen:
            return
        action = ACTION_ORDER[action_index]
        result = step_game(state, action)
        if result.state.terminal is not None:
            return
        if any(event.get("type") == "blocked" for event in result.events):
            return
        seen.add(action_index)
        candidates.append((action_index, source))

    add(int(teacher_action), "astar-preferred")
    for index in range(len(ACTION_ORDER)):
        add(index, "safe-alternative")

    return candidates


def parse_floats(raw: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid float list: {raw}") from exc
    if not values:
        raise SystemExit("ERROR: float list cannot be empty.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V12.9 cached-frontier guarded surgical repair for the full V5 MaleCNS. "
            "The 165k-neuron recurrent policy stays frozen. Prefix validation is "
            "performed analytically on cached readout features, and candidate closed-loop "
            "rollouts resume from sparse exact recurrent-state checkpoints near the frontier."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default="crossy-v4-expo:0006")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--max-repairs", type=int, default=160)
    parser.add_argument("--max-bridge-repairs", type=int, default=16)
    parser.add_argument(
        "--max-search-restarts",
        type=int,
        default=32,
        help=(
            "Maximum DFS-style transaction restarts. When a tentative branch hits a "
            "dead end, V12.9 bans its last bridge choice, restores the committed root, "
            "and deterministically rebuilds the path until the first untried sibling."
        ),
    )
    parser.add_argument("--max-bridge-score-drop", type=float, default=2.0)
    parser.add_argument("--max-rewind-step-drop", type=int, default=48)
    parser.add_argument("--bridge-max-stall", type=int, default=12)
    parser.add_argument("--stall-abort", type=int, default=20)
    parser.add_argument("--max-blocked-rate", type=float, default=0.08)
    parser.add_argument("--max-marginal-blocked-rate", type=float, default=0.075)
    parser.add_argument(
        "--marginal-block-warmup-steps",
        type=int,
        default=14,
        help=(
            "Rollout-only minimum observation horizon for the marginal blocked "
            "early-abort budget. With the default 7.5% threshold, 14 is the first "
            "window where two discrete blocked actions are representable. Final "
            "commit quality still uses actual marginal steps with ceil(rate * steps)."
        ),
    )
    parser.add_argument(
        "--terminal-rescue-lookback",
        type=int,
        default=12,
        help=(
            "For ordinary terminal failures, inspect this many recent suffix states and "
            "repair the earliest A* disagreement after the last tentative bridge, before "
            "the decoder can drift into a later irreversible terminal/blocked trap."
        ),
    )
    parser.add_argument(
        "--root-frontier-lookback",
        type=int,
        default=48,
        help=(
            "When the committed root has no admissible commit/bridge at the first "
            "terminal-rescue frontier, keep searching other A* disagreement frontiers. "
            "The search expands from terminal-rescue-lookback up to this many suffix steps "
            "instead of terminating after one root repair."
        ),
    )
    parser.add_argument(
        "--corridor-max-actions",
        type=int,
        default=12,
        help=(
            "Maximum number of consecutive actions from the winning A* path to install "
            "as one exact-nullspace readout bundle. Set to 1 to effectively disable "
            "multi-action corridor surgery while retaining ordinary one-step repair."
        ),
    )
    parser.add_argument(
        "--min-commit-progress-rate",
        type=float,
        default=0.40,
        help=(
            "Absolute global safety floor. V12.9 does not use 0.45 as the primary "
            "incremental commit gate; local teacher-relative efficiency is primary."
        ),
    )
    parser.add_argument("--bridge-min-progress-rate", type=float, default=0.40)
    parser.add_argument(
        "--teacher-trajectory",
        type=Path,
        default=ROOT / "runs" / "full-malecns-seed-search-v5" / "trajectory-5000.json",
        help="Fixed-seed A* trajectory used only as a local efficiency reference.",
    )
    parser.add_argument(
        "--max-teacher-slowdown",
        type=float,
        default=3.0,
        help="Maximum local candidate/teacher step ratio for a score-gain commit.",
    )
    parser.add_argument(
        "--teacher-step-slack",
        type=int,
        default=12,
        help="Additive local step slack for short score-gain windows.",
    )
    parser.add_argument("--margin", type=float, default=0.50)
    parser.add_argument("--protection-recent", type=int, default=96)
    parser.add_argument("--protection-anchors", type=int, default=64)
    parser.add_argument("--protection-low-margin", type=int, default=64)
    parser.add_argument("--max-active-rounds", type=int, default=96)
    parser.add_argument(
        "--max-protected-guard-repairs",
        type=int,
        default=16,
        help=(
            "Maximum number of explicit float32 guard repairs attempted inside one "
            "active-set round when a witness already present in the protected set "
            "still changes action after the main nullspace surgery."
        ),
    )
    parser.add_argument(
        "--protected-guard-margin",
        type=float,
        default=0.05,
        help=(
            "Decision margin used when explicitly re-asserting an already-protected "
            "historical action. Frontier/corridor features remain exact-nullspace protected."
        ),
    )
    parser.add_argument("--protection-strengths", default="1")
    parser.add_argument("--alphas", default="0.90,1.0,1.10")
    parser.add_argument("--search-limits", default="80,160")
    parser.add_argument("--max-backtrack-rows", type=int, default=6)
    parser.add_argument("--max-expansions", type=int, default=250000)
    parser.add_argument(
        "--cache-stride",
        type=int,
        default=16,
        help=(
            "Store an exact MaleCNS recurrent state every N steps. Candidate rollouts "
            "reconstruct at most N-1 old recurrent steps before the changed frontier."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume current/committed traces, recurrent caches, decoder, and tentative bridges from run_state.pt.",
    )
    parser.add_argument(
        "--upgrade-v12-15-state",
        type=Path,
        default=None,
        help=(
            "Import a completed safe autosave from V12.15 into a new V12.16 output directory. "
            "V12.16 reconstructs persistent root-sweep progress from the old repair history, "
            "keeps DFS bans/current tentative branch, and does not carry the huge old repair log."
        ),
    )
    parser.add_argument(
        "--run-state",
        type=Path,
        default=None,
        help="Optional explicit autosave state path. Defaults to OUT/run_state.pt.",
    )
    parser.add_argument(
        "--no-autosave-state",
        action="store_true",
        help="Disable transaction autosave after baseline/bridge/commit. Not recommended.",
    )

    parser.add_argument(
        "--v5-checkpoint",
        type=Path,
        default=ROOT / "runs" / "crossy-full-readout-cached-frontier-v12-9-0006-1000" / "best.pt",
    )
    parser.add_argument(
        "--v4-checkpoint",
        type=Path,
        default=ROOT / "runs" / "crossy-v4-expo-specialist" / "best.pt",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=ROOT / "runs" / "crossy-v4-exact-seed-branch-curriculum" / "best_decoder.pt",
    )
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "crossy-full-readout-cached-frontier-v12-10-0006-1000",
    )
    args = parser.parse_args()

    if args.resume and args.upgrade_v12_15_state is not None:
        raise SystemExit("ERROR: use either --resume or --upgrade-v12-15-state, not both.")
    if args.target <= 0 or args.max_repairs <= 0:
        raise SystemExit("ERROR: target and max-repairs must be positive.")
    if args.cache_stride < 1:
        raise SystemExit("ERROR: --cache-stride must be >= 1.")
    if args.max_search_restarts < 0:
        raise SystemExit("ERROR: --max-search-restarts must be non-negative.")
    if args.max_teacher_slowdown <= 0.0:
        raise SystemExit("ERROR: --max-teacher-slowdown must be positive.")
    if args.teacher_step_slack < 0:
        raise SystemExit("ERROR: --teacher-step-slack must be non-negative.")
    if (
        args.max_bridge_repairs < 0
        or args.bridge_max_stall < 1
        or args.max_bridge_score_drop < 0.0
        or args.max_rewind_step_drop < 0
    ):
        raise SystemExit("ERROR: bridge controls are invalid.")
    if args.stall_abort < 2:
        raise SystemExit("ERROR: --stall-abort must be >= 2.")
    if not (0.0 <= args.max_blocked_rate <= 1.0):
        raise SystemExit("ERROR: --max-blocked-rate must be in [0, 1].")
    if not (0.0 <= args.max_marginal_blocked_rate <= 1.0):
        raise SystemExit("ERROR: --max-marginal-blocked-rate must be in [0, 1].")
    if args.marginal_block_warmup_steps < 0:
        raise SystemExit("ERROR: --marginal-block-warmup-steps must be non-negative.")
    if args.root_frontier_lookback < 1:
        raise SystemExit("ERROR: --root-frontier-lookback must be >= 1.")
    if args.corridor_max_actions < 1:
        raise SystemExit("ERROR: --corridor-max-actions must be >= 1.")
    if args.max_protected_guard_repairs < 0:
        raise SystemExit("ERROR: --max-protected-guard-repairs must be non-negative.")
    if args.protected_guard_margin <= 0.0:
        raise SystemExit("ERROR: --protected-guard-margin must be positive.")
    if not (0.0 <= args.bridge_min_progress_rate <= args.min_commit_progress_rate <= 1.0):
        raise SystemExit(
            "ERROR: require 0 <= bridge-min-progress-rate <= min-commit-progress-rate <= 1."
        )
    if (
        args.protection_recent < 0
        or args.protection_anchors < 0
        or args.protection_low_margin < 0
        or args.max_active_rounds < 0
    ):
        raise SystemExit("ERROR: active-set protection controls must be non-negative.")

    protection_strengths = parse_floats(args.protection_strengths)
    alphas = parse_floats(args.alphas)
    try:
        search_limits = sorted(
            {int(part.strip()) for part in args.search_limits.split(",") if part.strip()}
        )
    except ValueError as exc:
        raise SystemExit("ERROR: invalid --search-limits.") from exc

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA requested but unavailable.")

    for path in (args.v5_checkpoint, args.v4_checkpoint, args.decoder_checkpoint):
        if not path.is_file():
            raise SystemExit(f"ERROR: required file not found: {path}")

    teacher_reference = load_teacher_reference(
        args.teacher_trajectory, expected_seed=args.seed
    )

    args.out.mkdir(parents=True, exist_ok=True)
    run_state_path = args.run_state or (args.out / "run_state.pt")
    started = time.perf_counter()
    elapsed_prior = 0.0

    v5_payload = torch.load(args.v5_checkpoint, map_location="cpu", weights_only=False)
    rank = int(v5_payload.get("rank", 16))
    residual_scale = float(v5_payload.get("residualScale", 0.25))

    base, metadata, _, _ = _load_base_policy(
        checkpoint_path=args.v4_checkpoint,
        flyhard_root=args.flyhard_root,
        device=device,
    )
    policy = SensoryPlasticPolicy(base, rank=rank, residual_scale=residual_scale).to(device)
    decoder, decoder_metadata = load_decoder_checkpoint(args.decoder_checkpoint, device=device)
    loaded_v5 = load_adaptation_checkpoint(args.v5_checkpoint, policy=policy, decoder=decoder)

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    decoder.eval()

    print("=== FULL MALECNS / PERSISTENT ROOT-SWEEP V12.16 ===", flush=True)
    print(f"source checkpoint: {args.v5_checkpoint.resolve()}", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(
        f"MaleCNS frozen but active: {metadata['neurons']:,} neurons / "
        f"{metadata['edges']:,} measured edges / {metadata['motor']:,} vnc_motor",
        flush=True,
    )
    print(
        f"source V5 stage: {loaded_v5.get('stage')} | "
        f"readout={decoder.architecture} | target={args.target} | cacheStride={args.cache_stride}",
        flush=True,
    )
    print(
        f"teacher reference: {teacher_reference['steps']} steps / "
        f"score {teacher_reference['score']} | local slowdown<={args.max_teacher_slowdown:g}x "
        f"with +{args.teacher_step_slack} step slack | global floor>={args.min_commit_progress_rate:.3f}",
        flush=True,
    )
    print(
        f"terminal rescue: earliest-A* divergence, lookback={args.terminal_rescue_lookback} suffix steps, "
        f"root sweep expands to {args.root_frontier_lookback} suffix steps | "
        f"target=next recovery beyond terminal progress | "
        f"A* corridor bundle<={args.corridor_max_actions} actions | "
        f"protected guard<={args.max_protected_guard_repairs} repairs @ margin={args.protected_guard_margin:g} | "
        f"rollout block warmup={args.marginal_block_warmup_steps}",
        flush=True,
    )

    signature = {
        "version": VERSION,
        "seed": args.seed,
        "target": args.target,
        "maxBridgeRepairs": args.max_bridge_repairs,
        "maxBridgeScoreDrop": args.max_bridge_score_drop,
        "maxRewindStepDrop": args.max_rewind_step_drop,
        "stallAbort": args.stall_abort,
        "maxBlockedRate": args.max_blocked_rate,
        "maxMarginalBlockedRate": args.max_marginal_blocked_rate,
        "marginalBlockWarmupSteps": args.marginal_block_warmup_steps,
        "terminalRescueLookback": args.terminal_rescue_lookback,
        "rootFrontierLookback": args.root_frontier_lookback,
        "terminalRescuePolicy": "earliest-astar-divergence-with-root-frontier-sweep",
        "terminalRescueTargetPolicy": "next-recovery-beyond-terminal-progress",
        "corridorMaxActions": args.corridor_max_actions,
        "corridorBundlePolicy": "multi-action exact-nullspace bundle with previous corridor features protected",
        "minCommitProgressRate": args.min_commit_progress_rate,
        "bridgeMinProgressRate": args.bridge_min_progress_rate,
        "teacherTrajectory": str(args.teacher_trajectory.resolve()),
        "maxTeacherSlowdown": args.max_teacher_slowdown,
        "teacherStepSlack": args.teacher_step_slack,
        "margin": args.margin,
        "protectionRecent": args.protection_recent,
        "protectionAnchors": args.protection_anchors,
        "protectionLowMargin": args.protection_low_margin,
        "maxActiveRounds": args.max_active_rounds,
        "maxProtectedGuardRepairs": args.max_protected_guard_repairs,
        "protectedGuardMargin": args.protected_guard_margin,
        "protectionStrengths": protection_strengths,
        "alphas": alphas,
        "cacheStride": args.cache_stride,
        "sourceV5Checkpoint": str(args.v5_checkpoint.resolve()),
    }

    config = {
        **signature,
        "maxRepairs": args.max_repairs,
        "bridgeMaxStall": args.bridge_max_stall,
        "searchLimits": search_limits,
        "maxBacktrackRows": args.max_backtrack_rows,
        "maxExpansions": args.max_expansions,
        "fullMaleCNSActive": True,
        "fullMaleCNSRetrained": False,
        "readoutSurgery": "minimal exact-nullspace updates plus protected-witness float32 guard repairs in existing decoder output layers",
        "prefixValidation": "analytic cached readout features; no MaleCNS prefix replay",
        "frontierResume": "sparse exact recurrent-state cache plus suffix-only closed loop",
        "transactionAutosave": not args.no_autosave_state,
        "transactionBacktracking": "DFS bridge-choice restart plus persistent committed-root multi-frontier sweep; sweep state resets only after a score commit",
        "maxSearchRestarts": args.max_search_restarts,
        "commitGate": "V12.16 keeps V12.10 readout surgery and every quality gate unchanged. It fixes V12.15 root-sweep persistence: rejected root frontiers and the expanded lookback survive tentative bridges and DFS backtracks, and reset only after a real score commit.",
        "teacherReferenceSummary": {
            "steps": teacher_reference["steps"],
            "score": teacher_reference["score"],
            "seed": teacher_reference["seed"],
        },
    }

    if args.upgrade_v12_15_state is not None:
        import_state_path = args.upgrade_v12_15_state.resolve()
        if not import_state_path.is_file():
            raise SystemExit(f"ERROR: V12.15 import state does not exist: {import_state_path}")
        saved = load_run_state(import_state_path)
        saved_version = saved.get("version")
        if saved_version != PREVIOUS_VERSION:
            raise SystemExit(
                f"ERROR: --upgrade-v12-15-state requires {PREVIOUS_VERSION}, got {saved_version}."
            )
        saved_signature = copy.deepcopy(saved.get("signature") or {})
        saved_signature["version"] = VERSION
        if saved_signature != signature:
            raise SystemExit(
                "ERROR: imported V12.15 state configuration does not match this V12.16 invocation. "
                "Use the same seed/checkpoint/quality/search arguments used by V12.15."
            )

        restore_decoder(decoder, saved["decoderState"])
        current_metrics = saved["currentMetrics"]
        current_trace = saved["currentTrace"]
        current_cache = saved["currentCache"]
        committed_decoder = saved["committedDecoder"]
        committed_metrics = saved["committedMetrics"]
        committed_trace = saved["committedTrace"]
        committed_cache = saved["committedCache"]
        baseline_metrics = saved["baselineMetrics"]
        quality_anchor_steps = int(saved["qualityAnchorSteps"])
        quality_anchor_blocked = int(saved["qualityAnchorBlocked"])
        bridge_depth = int(saved["bridgeDepth"])
        bridge_frontiers = list(saved["bridgeFrontiers"])
        old_repairs = list(saved.get("repairs") or [])
        if "bridgePath" in saved:
            bridge_path = list(saved["bridgePath"])
        else:
            bridge_path = _derive_bridge_path(old_repairs, current_trace)
        raw_bans = saved.get("bannedChoices") or {}
        banned_choices: dict[tuple[int, int, float, str], set[tuple[int, str, str, str]]] = {
            tuple(key): {tuple(choice) for choice in choices}
            for key, choices in raw_bans.items()
        }
        search_restarts = int(saved.get("searchRestarts", 0))
        reconstructed_attempts, reconstructed_lookback = _rebuild_root_sweep_progress(
            old_repairs,
            terminal_rescue_lookback=args.terminal_rescue_lookback,
        )
        # Union any state that happened to survive the V12.15 reset bug.
        for raw in saved.get("rootFrontierAttempts", []) or []:
            value = int(raw)
            if value not in reconstructed_attempts:
                reconstructed_attempts.append(value)
        root_frontier_attempts = reconstructed_attempts
        root_frontier_lookback_current = max(
            reconstructed_lookback,
            int(saved.get("rootFrontierLookbackCurrent", args.terminal_rescue_lookback)),
        )
        start_repair_index = int(saved.get("nextRepairIndex", 1))
        success = bool(saved.get("success", False))
        imported_elapsed_seconds = float(saved.get("elapsedSeconds", 0.0))
        # Report V12.16 runtime separately; the old elapsed time is kept as metadata.
        elapsed_prior = 0.0
        repairs = [
            {
                "repair": 0,
                "decision": "upgrade-v12-15-state",
                "sourceState": str(import_state_path),
                "importedElapsedSeconds": imported_elapsed_seconds,
                "importedRepairCount": len(old_repairs),
                "reconstructedRootFrontierAttempts": list(root_frontier_attempts),
                "reconstructedRootLookback": int(root_frontier_lookback_current),
                "preservedSearchRestarts": int(search_restarts),
                "preservedBannedChoiceCount": sum(len(v) for v in banned_choices.values()),
                "preservedBridgeDepth": int(bridge_depth),
            }
        ]
        config["upgradedFromV1215State"] = str(import_state_path)
        config["importedV1215ElapsedSeconds"] = imported_elapsed_seconds
        print(
            f"\n[UPGRADE V12.15 -> V12.16] nextRepair={start_repair_index} "
            f"current={current_metrics['steps']}/{current_metrics['score']:.0f} "
            f"committed={committed_metrics['steps']}/{committed_metrics['score']:.0f} "
            f"bridgeDepth={bridge_depth} searchRestarts={search_restarts} "
            f"rootSweep={len(root_frontier_attempts)} rootLookback={root_frontier_lookback_current}",
            flush=True,
        )
        print(
            f"  imported old runtime={imported_elapsed_seconds/3600.0:.2f}h; "
            "V12.16 runtime counter starts at 0 and old repair payload is pruned.",
            flush=True,
        )
    elif args.resume:
        if not run_state_path.is_file():
            raise SystemExit(f"ERROR: --resume requested but state file does not exist: {run_state_path}")
        saved = load_run_state(run_state_path)
        saved_version = saved.get("version")
        if saved_version != VERSION:
            raise SystemExit(
                f"ERROR: run state version mismatch: {saved_version} != {VERSION}. "
                "Start V12.16 from the last safe best.pt in a new output directory; "
                "do not reuse an older search transaction."
            )
        if saved.get("signature") != signature:
            raise SystemExit(
                "ERROR: run-state configuration does not match this V12.16 invocation. "
                "Use the same V12.16 arguments or start a fresh output directory."
            )
        restore_decoder(decoder, saved["decoderState"])
        current_metrics = saved["currentMetrics"]
        current_trace = saved["currentTrace"]
        current_cache = saved["currentCache"]
        committed_decoder = saved["committedDecoder"]
        committed_metrics = saved["committedMetrics"]
        committed_trace = saved["committedTrace"]
        committed_cache = saved["committedCache"]
        baseline_metrics = saved["baselineMetrics"]
        quality_anchor_steps = int(saved["qualityAnchorSteps"])
        quality_anchor_blocked = int(saved["qualityAnchorBlocked"])
        bridge_depth = int(saved["bridgeDepth"])
        bridge_frontiers = list(saved["bridgeFrontiers"])
        repairs = list(saved["repairs"])
        if "bridgePath" in saved:
            bridge_path = list(saved["bridgePath"])
        else:
            bridge_path = _derive_bridge_path(repairs, current_trace)
        raw_bans = saved.get("bannedChoices") or {}
        banned_choices: dict[tuple[int, int, float, str], set[tuple[int, str, str, str]]] = {
            tuple(key): {tuple(choice) for choice in choices}
            for key, choices in raw_bans.items()
        }
        search_restarts = int(saved.get("searchRestarts", 0))
        root_frontier_lookback_current = int(
            saved.get("rootFrontierLookbackCurrent", args.terminal_rescue_lookback)
        )
        root_frontier_attempts = [int(x) for x in saved.get("rootFrontierAttempts", [])]
        start_repair_index = int(saved["nextRepairIndex"])
        success = bool(saved["success"])
        elapsed_prior = float(saved.get("elapsedSeconds", 0.0))
        print(
            f"\n[RESUME] repair={start_repair_index} "
            f"current={current_metrics['steps']}/{current_metrics['score']:.0f} "
            f"committed={committed_metrics['steps']}/{committed_metrics['score']:.0f} "
            f"bridgeDepth={bridge_depth} cacheStates={len(current_cache)} "
            f"searchRestarts={search_restarts} rootSweep={len(root_frontier_attempts)} "
            f"rootLookback={root_frontier_lookback_current}",
            flush=True,
        )
    else:
        current_metrics, current_trace, current_cache = rollout_trace(
            policy=policy,
            decoder=decoder,
            seed=args.seed,
            device=device,
            max_steps=args.target,
            stall_abort=args.stall_abort,
            cache_stride=args.cache_stride,
        )
        baseline_metrics = copy.deepcopy(current_metrics)
        quality_anchor_steps = int(baseline_metrics["steps"])
        quality_anchor_blocked = int(baseline_metrics["blockedActions"])
        print(
            f"\n[baseline] steps={current_metrics['steps']} "
            f"score={current_metrics['score']:.0f} terminal={current_metrics['terminalReason']}",
            flush=True,
        )
        print(
            f"[quality anchor] steps={quality_anchor_steps} blocked={quality_anchor_blocked} "
            f"rate={float(baseline_metrics['blockedRate']):.3f} | "
            f"new-extension blocked~={args.max_marginal_blocked_rate:.3f} discrete "
            f"(rollout warmup={args.marginal_block_warmup_steps}) | "
            f"global floor>={args.min_commit_progress_rate:.3f}",
            flush=True,
        )
        print(
            f"[cache] {len(current_cache)} recurrent states retained; future candidates "
            f"will analytically skip the prefix and replay at most {args.cache_stride - 1} "
            "old MaleCNS frames before their frontier.",
            flush=True,
        )

        save_checkpoint(
            args.out / "best.pt",
            policy=policy,
            decoder=decoder,
            source_checkpoint=args.v4_checkpoint,
            source_decoder=args.decoder_checkpoint,
            stage="readout-cached-frontier-baseline",
            metrics={"exactExpo": current_metrics},
            config=config,
        )
        repairs: list[dict[str, Any]] = []
        success = quality_success(
            current_metrics,
            args.target,
            max_no_progress=args.stall_abort,
            max_blocked_rate=args.max_blocked_rate,
            min_progress_rate=args.min_commit_progress_rate,
            max_marginal_blocked_rate=args.max_marginal_blocked_rate,
        )
        committed_decoder = snapshot_decoder(decoder)
        committed_metrics = copy.deepcopy(current_metrics)
        committed_trace = current_trace
        committed_cache = current_cache
        bridge_depth = 0
        bridge_frontiers: list[int] = []
        bridge_path: list[dict[str, Any]] = []
        banned_choices: dict[tuple[int, int, float, str], set[tuple[int, str, str, str]]] = {}
        search_restarts = 0
        root_frontier_lookback_current = max(1, int(args.terminal_rescue_lookback))
        root_frontier_attempts: list[int] = []
        start_repair_index = 1

    def elapsed_total() -> float:
        return elapsed_prior + (time.perf_counter() - started)

    def autosave(next_repair_index: int) -> None:
        if args.no_autosave_state:
            return
        save_run_state(
            path=run_state_path,
            signature=signature,
            decoder=decoder,
            current_metrics=current_metrics,
            current_trace=current_trace,
            current_cache=current_cache,
            committed_decoder=committed_decoder,
            committed_metrics=committed_metrics,
            committed_trace=committed_trace,
            committed_cache=committed_cache,
            baseline_metrics=baseline_metrics,
            quality_anchor_steps=quality_anchor_steps,
            quality_anchor_blocked=quality_anchor_blocked,
            bridge_depth=bridge_depth,
            bridge_frontiers=bridge_frontiers,
            bridge_path=bridge_path,
            banned_choices=banned_choices,
            search_restarts=search_restarts,
            root_frontier_lookback_current=root_frontier_lookback_current,
            root_frontier_attempts=root_frontier_attempts,
            repairs=repairs,
            next_repair_index=next_repair_index,
            success=success,
            elapsed_seconds=elapsed_total(),
        )

    def backtrack_transaction(reason: str, repair_index: int) -> bool:
        nonlocal current_metrics, current_trace, current_cache
        nonlocal bridge_depth, bridge_frontiers, bridge_path, search_restarts
        nonlocal root_frontier_lookback_current, root_frontier_attempts
        if not bridge_path or search_restarts >= args.max_search_restarts:
            return False

        dead_choice = bridge_path[-1]
        dead_state_key = tuple(dead_choice["stateKey"])
        dead_signature = tuple(dead_choice["choice"])
        banned_choices.setdefault(dead_state_key, set()).add(dead_signature)
        search_restarts += 1

        restore_decoder(decoder, committed_decoder)
        current_metrics = copy.deepcopy(committed_metrics)
        current_trace = committed_trace
        current_cache = committed_cache
        bridge_depth = 0
        bridge_frontiers = []
        bridge_path = []
        # V12.16: keep committed-root sweep progress across DFS backtracks.
        # Only a real score commit changes the committed root and is allowed to
        # reset root_frontier_attempts/root_frontier_lookback_current.

        repairs.append(
            {
                "repair": repair_index,
                "decision": "search-backtrack",
                "reason": reason,
                "bannedStateKey": dead_state_key,
                "bannedChoice": dead_signature,
                "searchRestart": search_restarts,
                "restoredCommittedSteps": current_metrics["steps"],
                "restoredCommittedScore": current_metrics["score"],
            }
        )
        print(
            f"  SEARCH BACKTRACK {search_restarts}/{args.max_search_restarts}: "
            f"{reason}; banning last bridge {dead_choice['targetAction']} "
            f"alpha={dead_choice['alpha']:g} at frontier {dead_choice['frontierIndex']} "
            f"and rebuilding from committed {current_metrics['steps']}/{current_metrics['score']:.0f}.",
            flush=True,
        )
        autosave(repair_index + 1)
        return True

    if not args.resume:
        autosave(start_repair_index)
        if not args.no_autosave_state:
            print(f"[autosave] {run_state_path.resolve()}", flush=True)

    stop_reason: str | None = None
    interrupted = False

    try:
        for repair_index in range(start_repair_index, args.max_repairs + 1):
            if success:
                stop_reason = "target-achieved"
                break
            if not current_trace:
                stop_reason = "empty-trace"
                break

            at_committed_root = bridge_depth == 0
            min_rescue_frontier = (
                max(bridge_frontiers) + 1
                if bridge_depth > 0 and bridge_frontiers
                else 0
            )
            effective_lookback = (
                int(root_frontier_lookback_current)
                if at_committed_root
                else int(args.terminal_rescue_lookback)
            )
            frontier_choice = choose_repair_frontier(
                trace=current_trace,
                metrics=current_metrics,
                search_limits=search_limits,
                max_backtrack_rows=args.max_backtrack_rows,
                max_expansions=args.max_expansions,
                min_frontier_index=min_rescue_frontier,
                terminal_rescue_lookback=effective_lookback,
                excluded_frontiers=(set(root_frontier_attempts) if at_committed_root else None),
            )
            if frontier_choice is None:
                if at_committed_root and root_frontier_lookback_current < args.root_frontier_lookback:
                    old_window = int(root_frontier_lookback_current)
                    root_frontier_lookback_current = min(
                        int(args.root_frontier_lookback),
                        max(old_window + 1, old_window * 2),
                    )
                    print(
                        f"\n  ROOT SWEEP EXPAND: no untried repairable frontier in last {old_window} "
                        f"steps; expanding to {root_frontier_lookback_current}. "
                        f"Already tried={root_frontier_attempts}",
                        flush=True,
                    )
                    repairs.append(
                        {
                            "repair": repair_index,
                            "decision": "root-sweep-expand",
                            "oldLookback": old_window,
                            "newLookback": int(root_frontier_lookback_current),
                            "attemptedFrontiers": list(root_frontier_attempts),
                        }
                    )
                    autosave(repair_index + 1)
                    continue
                stop_reason = (
                    "root-frontiers-exhausted" if at_committed_root and root_frontier_attempts
                    else "no-repairable-frontier"
                )
                print(
                    f"\nSTOP: {stop_reason}; tried root frontiers={root_frontier_attempts} "
                    f"lookback={root_frontier_lookback_current}.",
                    flush=True,
                )
                break

            frontier_index, target_action, teacher_report = frontier_choice
            rewound_transaction = False
            rewind_info: dict[str, Any] | None = None

            if (
                current_metrics["terminalReason"] in {"stagnation", "blocked-budget"}
                and bridge_depth > 0
                and bridge_frontiers
                and frontier_index < min(bridge_frontiers)
            ):
                stalled_metrics = copy.deepcopy(current_metrics)
                stalled_frontier = current_trace[frontier_index]
                earliest_bridge = min(bridge_frontiers)

                restore_decoder(decoder, committed_decoder)
                current_metrics = copy.deepcopy(committed_metrics)
                current_trace = committed_trace
                current_cache = committed_cache

                if frontier_index >= len(current_trace):
                    stop_reason = "rewind-frontier-outside-committed-trace"
                    print("\nSTOP: rewind frontier is outside the committed trace.", flush=True)
                    break
                committed_frontier = current_trace[frontier_index]
                if state_key(committed_frontier.state) != state_key(stalled_frontier.state):
                    stop_reason = "rewind-frontier-state-mismatch"
                    print("\nSTOP: rewind frontier state does not match committed prefix.", flush=True)
                    break

                target_action, rewind_teacher = astar_teacher_action(
                    committed_frontier.state,
                    search_limits=search_limits,
                    max_backtrack_rows=args.max_backtrack_rows,
                    max_expansions=args.max_expansions,
                )
                teacher_report = {
                    **rewind_teacher,
                    "transactionRewind": True,
                    "rewindFromBridgeDepth": bridge_depth,
                    "earliestTentativeFrontier": earliest_bridge,
                    "stalledBranchSteps": stalled_metrics["steps"],
                    "stalledBranchScore": stalled_metrics["score"],
                    "selectedEarlierFrontier": frontier_index,
                }
                rewind_info = {
                    "fromBridgeDepth": bridge_depth,
                    "earliestTentativeFrontier": earliest_bridge,
                    "stalledBranch": stalled_metrics,
                    "rewindFrontier": frontier_index,
                }
                bridge_depth = 0
                bridge_frontiers = []
                bridge_path = []
                rewound_transaction = True
                print(
                    f"\n  REWIND TRANSACTION: earlier frontier {frontier_index} < "
                    f"first tentative {earliest_bridge}; retrying from committed "
                    f"{current_metrics['steps']}/{current_metrics['score']:.0f}.",
                    flush=True,
                )

            frontier = current_trace[frontier_index]
            student_action = frontier.student_action
            print(
                f"\n[repair {repair_index}/{args.max_repairs}] "
                f"frontier step={frontier.index} score={current_metrics['score']:.0f} "
                f"reason={current_metrics['terminalReason']}",
                flush=True,
            )
            print(
                f"  student={ACTION_ORDER[student_action].value} -> "
                f"teacher={ACTION_ORDER[target_action].value} ({teacher_report['method']})",
                flush=True,
            )
            if target_action == student_action:
                # V11 stopped here. That was unnecessarily strict: the teacher can
                # repeat the fatal action while another immediately safe action is
                # still a viable tactical bridge. V12 keeps the safe-action sweep.
                print(
                    "  teacher repeated the current action; V12 will sweep other "
                    "immediately-safe alternatives instead of stopping.",
                    flush=True,
                )

            initial_protection_indices = choose_protection_indices(
                current_trace,
                frontier_index=frontier_index,
                recent=args.protection_recent,
                anchors=args.protection_anchors,
                low_margin=args.protection_low_margin,
            )
            base_decoder = snapshot_decoder(decoder)
            base_metrics = current_metrics
            committed_score = float(committed_metrics["score"])

            best_commit_state = None
            best_commit_metrics = None
            best_commit_trace = None
            best_commit_cache = None
            best_commit_key = None
            best_commit_choice = None
            best_bridge_state = None
            best_bridge_metrics = None
            best_bridge_trace = None
            best_bridge_cache = None
            best_bridge_key = None
            best_bridge_choice = None
            trials: list[dict[str, Any]] = []

            frontier_state_key = state_key(frontier.state)
            corridor_plan_actions: list[int] = []
            if teacher_report.get("method") == "astar" and args.corridor_max_actions > 1:
                for name in teacher_report.get("planActions", []):
                    if name not in ACTION_TO_INDEX:
                        corridor_plan_actions = []
                        break
                    corridor_plan_actions.append(int(ACTION_TO_INDEX[name]))
                corridor_plan_actions = corridor_plan_actions[: int(args.corridor_max_actions)]
                if (
                    len(corridor_plan_actions) < 2
                    or int(corridor_plan_actions[0]) != int(target_action)
                ):
                    corridor_plan_actions = []

            candidate_actions: list[tuple[int, str]] = []
            if corridor_plan_actions:
                candidate_actions.append((int(target_action), "astar-corridor"))
            candidate_actions.extend(
                frontier_action_candidates(
                    state=frontier.state,
                    student_action=student_action,
                    teacher_action=target_action,
                )
            )
            if not candidate_actions:
                if backtrack_transaction("no-immediately-safe-alternative", repair_index):
                    continue
                stop_reason = "no-immediately-safe-alternative"
                print("  STOP: no immediately safe alternative action exists and transaction search is exhausted.", flush=True)
                break

            print(
                "  action sweep: "
                + ", ".join(
                    f"{ACTION_ORDER[action_index].value}"
                    + ("**" if source == "astar-corridor" else "*" if source == "astar-preferred" else "")
                    for action_index, source in candidate_actions
                )
                + "  (** = bundled A* corridor, * = A* first action)",
                flush=True,
            )
            if corridor_plan_actions:
                print(
                    "  corridor plan: "
                    + " -> ".join(ACTION_ORDER[index].value for index in corridor_plan_actions)
                    + f"  (bundle={len(corridor_plan_actions)}, fullAStar={teacher_report.get('planLength')})",
                    flush=True,
                )
            print(
                f"  active protection seed: {len(initial_protection_indices)} states "
                f"(recent={args.protection_recent}, anchors={args.protection_anchors}, "
                f"lowMargin={args.protection_low_margin})",
                flush=True,
            )

            for candidate_action, candidate_source in candidate_actions:
                for strength in protection_strengths:
                    for alpha in alphas:
                        choice_signature = _choice_signature(candidate_action, strength, alpha, candidate_source)
                        if choice_signature in banned_choices.get(frontier_state_key, set()):
                            trials.append(
                                {
                                    "targetAction": ACTION_ORDER[candidate_action].value,
                                    "targetActionSource": candidate_source,
                                    "protectionStrength": strength,
                                    "alpha": alpha,
                                    "stabilized": False,
                                    "stabilizeReason": "search-backtrack-banned",
                                    "searchBanned": True,
                                    "commitEligible": False,
                                    "bridgeEligible": False,
                                }
                            )
                            print(
                                f"    action={ACTION_ORDER[candidate_action].value:<8} "
                                f"source={candidate_source:<16} alpha={alpha:g} -> "
                                "SKIP (previously led this transaction branch to a dead end)",
                                flush=True,
                            )
                            continue
                        corridor_entries: list[dict[str, Any]] | None = None
                        corridor_build_report: dict[str, Any] | None = None
                        if candidate_source == "astar-corridor":
                            restore_decoder(decoder, base_decoder)
                            try:
                                corridor_entries, corridor_build_report = build_forced_corridor_features(
                                    policy=policy,
                                    decoder=decoder,
                                    current_trace=current_trace,
                                    current_cache=current_cache,
                                    frontier_index=frontier_index,
                                    plan_actions=corridor_plan_actions,
                                    device=device,
                                )
                            except RuntimeError as exc:
                                corridor_entries = None
                                corridor_build_report = {"valid": False, "reason": str(exc)}
                            if (
                                not corridor_entries
                                or not corridor_build_report
                                or not corridor_build_report.get("valid", False)
                            ):
                                restore_decoder(decoder, base_decoder)
                                trials.append(
                                    {
                                        "targetAction": ACTION_ORDER[candidate_action].value,
                                        "targetActionSource": candidate_source,
                                        "protectionStrength": strength,
                                        "alpha": alpha,
                                        "stabilized": False,
                                        "stabilizeReason": "corridor-build-failed",
                                        "corridorBuild": corridor_build_report,
                                        "commitEligible": False,
                                        "bridgeEligible": False,
                                    }
                                )
                                print(
                                    f"    action={ACTION_ORDER[candidate_action].value:<8} "
                                    f"source={candidate_source:<16} alpha={alpha:g} -> "
                                    f"CORRIDOR BUILD REJECT: {corridor_build_report}",
                                    flush=True,
                                )
                                continue

                        active_indices: set[int] = set(initial_protection_indices)
                        witnesses: list[int] = []
                        active_rounds: list[dict[str, Any]] = []
                        protected_guard_repairs: list[dict[str, Any]] = []
                        stabilized = False
                        stabilize_reason = "unknown"
                        delta = None
                        delta_report = None
                        stabilized_prefix_logits: np.ndarray | None = None
                        stabilized_decoder_state: dict[str, torch.Tensor] | None = None

                        for active_round in range(args.max_active_rounds + 1):
                            active_protection = protection_matrix(
                                current_trace,
                                active_indices,
                                feature_dim=len(frontier.feature),
                            )
                            restore_decoder(decoder, base_decoder)
                            try:
                                if candidate_source == "astar-corridor":
                                    assert corridor_entries is not None
                                    delta_report = apply_corridor_bundle(
                                        decoder=decoder,
                                        corridor_entries=corridor_entries,
                                        base_protection=active_protection,
                                        protection_strength=strength,
                                        margin=args.margin,
                                        alpha=alpha,
                                    )
                                    delta = torch.empty(0)
                                else:
                                    delta, delta_report = minimal_readout_delta(
                                        feature=frontier.feature,
                                        logits=frontier.logits,
                                        target_action=candidate_action,
                                        protection=active_protection,
                                        protection_strength=strength,
                                        margin=args.margin,
                                    )
                                    apply_virtual_delta(decoder, delta, alpha=alpha)
                            except RuntimeError as exc:
                                stabilize_reason = f"nullspace-failed: {exc}"
                                active_rounds.append(
                                    {
                                        "round": active_round,
                                        "protectedStates": len(active_indices),
                                        "status": "nullspace-failed",
                                        "error": str(exc),
                                        "corridorBundle": candidate_source == "astar-corridor",
                                    }
                                )
                                break

                            prefix_divergence, predicted_frontier_action, prefix_logits = (
                                analytic_prefix_validation(
                                    decoder=decoder,
                                    trace=current_trace,
                                    frontier_index=frontier_index,
                                    target_action=candidate_action,
                                )
                            )
                            corridor_matched_actions, corridor_all_matched, corridor_predictions = (
                                corridor_match_status(
                                    decoder,
                                    corridor_entries if candidate_source == "astar-corridor" else None,
                                )
                            )
                            if candidate_source != "astar-corridor":
                                corridor_matched_actions = 1
                                corridor_all_matched = True
                            elif delta_report is not None:
                                # Keep the aggregate corridor report honest after any later guard repair.
                                delta_report["corridorMatchedActions"] = int(corridor_matched_actions)
                                delta_report["corridorAllMatched"] = bool(corridor_all_matched)
                                delta_report["corridorPredictedActions"] = corridor_predictions

                            round_guard_repairs = 0
                            guard_failure: str | None = None
                            if candidate_source == "astar-corridor":
                                assert corridor_entries is not None
                                mandatory_features = np.stack(
                                    [
                                        np.asarray(entry["feature"], dtype=np.float64)
                                        for entry in corridor_entries
                                    ],
                                    axis=0,
                                )
                            else:
                                mandatory_features = np.asarray(
                                    frontier.feature, dtype=np.float64
                                )[None]

                            # V12.9: if an already-protected witness still flips, the
                            # mathematical nullspace constraint has been lost only at
                            # float32 installation precision. Re-assert that historical
                            # action explicitly while every other active witness and the
                            # requested frontier/corridor features remain protected.
                            while (
                                prefix_divergence is not None
                                and prefix_divergence < frontier_index
                                and prefix_divergence in active_indices
                                and round_guard_repairs < args.max_protected_guard_repairs
                            ):
                                try:
                                    guard_report = repair_protected_witness(
                                        decoder=decoder,
                                        trace=current_trace,
                                        active_indices=active_indices,
                                        witness_index=prefix_divergence,
                                        mandatory_features=mandatory_features,
                                        protection_strength=strength,
                                        guard_margin=args.protected_guard_margin,
                                    )
                                except RuntimeError as exc:
                                    guard_failure = f"protected-guard-failed: {exc}"
                                    break
                                guard_report = {
                                    "activeRound": int(active_round),
                                    **guard_report,
                                }
                                protected_guard_repairs.append(guard_report)
                                round_guard_repairs += 1
                                if not bool(guard_report.get("repaired", False)):
                                    guard_failure = "protected-guard-action-not-restored"
                                    break

                                prefix_divergence, predicted_frontier_action, prefix_logits = (
                                    analytic_prefix_validation(
                                        decoder=decoder,
                                        trace=current_trace,
                                        frontier_index=frontier_index,
                                        target_action=candidate_action,
                                    )
                                )
                                if candidate_source == "astar-corridor":
                                    (
                                        corridor_matched_actions,
                                        corridor_all_matched,
                                        corridor_predictions,
                                    ) = corridor_match_status(decoder, corridor_entries)
                                    if delta_report is not None:
                                        delta_report["corridorMatchedActions"] = int(
                                            corridor_matched_actions
                                        )
                                        delta_report["corridorAllMatched"] = bool(
                                            corridor_all_matched
                                        )
                                        delta_report["corridorPredictedActions"] = corridor_predictions

                            active_rounds.append(
                                {
                                    "round": active_round,
                                    "protectedStates": len(active_indices),
                                    "prefixDivergence": prefix_divergence,
                                    "frontierPredictedAction": ACTION_ORDER[predicted_frontier_action].value,
                                    "validation": "analytic-readout-cache",
                                    "deltaNorm": float(delta_report["deltaNorm"]),
                                    "protectedRank": int(delta_report["protectedRank"]),
                                    "nullspaceDimensionLowerBound": int(
                                        delta_report["nullspaceDimensionLowerBound"]
                                    ),
                                    "corridorBundle": candidate_source == "astar-corridor",
                                    "corridorLength": int((delta_report or {}).get("corridorLength", 1)),
                                    "corridorMatchedActions": int(corridor_matched_actions),
                                    "corridorAllMatched": bool(corridor_all_matched),
                                    "protectedGuardRepairs": int(round_guard_repairs),
                                    "protectedGuardFailure": guard_failure,
                                }
                            )

                            if guard_failure is not None:
                                stabilize_reason = guard_failure
                                break
                            if (
                                prefix_divergence == frontier_index
                                and predicted_frontier_action == candidate_action
                                and corridor_all_matched
                            ):
                                stabilized = True
                                stabilize_reason = (
                                    "corridor-only-divergence"
                                    if candidate_source == "astar-corridor"
                                    else (
                                        "frontier-only-divergence-after-protected-guard"
                                        if round_guard_repairs
                                        else "frontier-only-divergence"
                                    )
                                )
                                stabilized_prefix_logits = prefix_logits
                                stabilized_decoder_state = snapshot_decoder(decoder)
                                break
                            if prefix_divergence is not None and prefix_divergence < frontier_index:
                                if prefix_divergence in active_indices:
                                    stabilize_reason = (
                                        "protected-guard-budget-exhausted"
                                        if round_guard_repairs >= args.max_protected_guard_repairs
                                        else "protected-witness-still-diverged"
                                    )
                                    break
                                active_indices.add(prefix_divergence)
                                witnesses.append(prefix_divergence)
                                stabilize_reason = "added-prefix-witness"
                                continue
                            if prefix_divergence is None:
                                stabilize_reason = "frontier-not-crossed"
                            elif predicted_frontier_action != candidate_action:
                                stabilize_reason = "frontier-crossed-to-wrong-action"
                            elif candidate_source == "astar-corridor" and not corridor_all_matched:
                                stabilize_reason = "corridor-later-action-not-crossed"
                            else:
                                stabilize_reason = "unexpected-prefix-state"
                            break

                        if (
                            not stabilized
                            or delta_report is None
                            or stabilized_prefix_logits is None
                            or stabilized_decoder_state is None
                        ):
                            restore_decoder(decoder, base_decoder)
                            trials.append(
                                {
                                    "targetAction": ACTION_ORDER[candidate_action].value,
                                    "targetActionSource": candidate_source,
                                    "protectionStrength": strength,
                                    "alpha": alpha,
                                    "stabilized": False,
                                    "stabilizeReason": stabilize_reason,
                                    "initialProtectionStates": len(initial_protection_indices),
                                    "finalProtectionStates": len(active_indices),
                                    "activeWitnesses": witnesses,
                                    "activeRounds": active_rounds,
                                    "protectedGuardRepairs": protected_guard_repairs,
                                    "corridorBuild": corridor_build_report,
                                    "corridorPlanActions": (
                                        [ACTION_ORDER[index].value for index in corridor_plan_actions]
                                        if candidate_source == "astar-corridor"
                                        else None
                                    ),
                                    "commitEligible": False,
                                    "bridgeEligible": False,
                                }
                            )
                            print(
                                f"    action={ACTION_ORDER[candidate_action].value:<8} "
                                f"source={candidate_source:<16} alpha={alpha:g} -> "
                                f"prefix-reject reason={stabilize_reason} "
                                f"active={len(active_indices)} +w={len(witnesses)} "
                                f"[MaleCNS prefix steps: 0]",
                                flush=True,
                            )
                            continue

                        restore_decoder(decoder, stabilized_decoder_state)
                        try:
                            metrics, trial_trace, trial_cache, runtime = rollout_from_frontier(
                                policy=policy,
                                decoder=decoder,
                                current_trace=current_trace,
                                current_cache=current_cache,
                                frontier_index=frontier_index,
                                expected_action=candidate_action,
                                prefix_logits=stabilized_prefix_logits,
                                device=device,
                                max_steps=args.target,
                                stall_abort=args.stall_abort,
                                quality_anchor_steps=quality_anchor_steps,
                                quality_anchor_blocked=quality_anchor_blocked,
                                max_marginal_blocked_rate=args.max_marginal_blocked_rate,
                                marginal_block_warmup_steps=args.marginal_block_warmup_steps,
                                cache_stride=args.cache_stride,
                            )
                        except RuntimeError as exc:
                            restore_decoder(decoder, base_decoder)
                            trials.append(
                                {
                                    "targetAction": ACTION_ORDER[candidate_action].value,
                                    "targetActionSource": candidate_source,
                                    "protectionStrength": strength,
                                    "alpha": alpha,
                                    "stabilized": False,
                                    "stabilizeReason": f"cached-runtime-mismatch: {exc}",
                                    "initialProtectionStates": len(initial_protection_indices),
                                    "finalProtectionStates": len(active_indices),
                                    "activeWitnesses": witnesses,
                                    "activeRounds": active_rounds,
                                    "protectedGuardRepairs": protected_guard_repairs,
                                    "commitEligible": False,
                                    "bridgeEligible": False,
                                }
                            )
                            print(
                                f"    action={ACTION_ORDER[candidate_action].value:<8} "
                                f"alpha={alpha:g} -> CACHE REJECT: {exc}",
                                flush=True,
                            )
                            continue

                        divergence = first_action_divergence(current_trace, trial_trace)
                        score_gain = float(metrics["score"]) > committed_score + 1e-9
                        preserved_prefix = divergence == frontier_index
                        marginal_quality = marginal_block_quality(
                            metrics,
                            max_marginal_blocked_rate=args.max_marginal_blocked_rate,
                        )
                        marginal_budget_violation = not bool(marginal_quality["eligible"])
                        global_blocked_ok = bool(
                            float(metrics["blockedRate"]) <= float(args.max_blocked_rate) + 1e-12
                        )
                        quality_violation = bool(
                            metrics.get("qualityAborted", False)
                            or marginal_budget_violation
                            or not global_blocked_ok
                        )
                        teacher_gate = teacher_local_gate(
                            teacher_reference,
                            committed_score=committed_metrics["score"],
                            candidate_score=metrics["score"],
                            committed_steps=int(committed_metrics["steps"]),
                            candidate_steps=int(metrics["steps"]),
                            max_slowdown=args.max_teacher_slowdown,
                            step_slack=args.teacher_step_slack,
                        )
                        commit_progress_ok = bool(
                            float(metrics["progressRate"]) >= args.min_commit_progress_rate
                            and teacher_gate["eligible"]
                        )
                        bridge_score_floor = committed_score - float(args.max_bridge_score_drop)
                        rewind_step_drop = max(
                            0, int(base_metrics["steps"]) - int(metrics["steps"])
                        )
                        # A terminal rescue intentionally rewrites an already-bad suffix.
                        # Requiring the repaired branch to survive at least as many raw
                        # steps as that doomed suffix defeats the rescue (V12.6 rejected an
                        # A*-preferred rescue for a one-step rewind). Like an explicit
                        # transaction rewind, allow a bounded provisional step decrease;
                        # commit quality is still governed by real score gain + teacher/
                        # blocked gates, so this never promotes a shorter branch by itself.
                        terminal_rescue_rewrite = bool(
                            teacher_report.get("terminalRescue", False)
                        )
                        if rewound_transaction or terminal_rescue_rewrite:
                            step_guard_ok = bool(
                                int(metrics["steps"]) > int(frontier_index)
                                and rewind_step_drop <= int(args.max_rewind_step_drop)
                            )
                        else:
                            step_guard_ok = bool(
                                int(metrics["steps"]) >= int(base_metrics["steps"])
                            )

                        commit_eligible = bool(
                            preserved_prefix
                            and score_gain
                            and not quality_violation
                            and commit_progress_ok
                        )
                        bridge_eligible = bool(
                            preserved_prefix
                            and not commit_eligible
                            and float(metrics["score"]) >= bridge_score_floor - 1e-9
                            and step_guard_ok
                            and int(metrics["maxNoProgressStreak"]) <= args.stall_abort
                            and float(metrics["progressRate"]) >= args.bridge_min_progress_rate
                        )

                        trial = {
                            "targetAction": ACTION_ORDER[candidate_action].value,
                            "targetActionSource": candidate_source,
                            "protectionStrength": strength,
                            "alpha": alpha,
                            "stabilized": True,
                            "stabilizeReason": stabilize_reason,
                            "initialProtectionStates": len(initial_protection_indices),
                            "finalProtectionStates": len(active_indices),
                            "activeWitnesses": witnesses,
                            "activeRounds": active_rounds,
                            "protectedGuardRepairs": protected_guard_repairs,
                            "delta": delta_report,
                            "firstActionDivergence": divergence,
                            "runtime": runtime,
                            "metrics": metrics,
                            "commitEligible": commit_eligible,
                            "bridgeEligible": bridge_eligible,
                            "scoreDeltaVsCommitted": float(metrics["score"]) - committed_score,
                            "bridgeScoreFloor": bridge_score_floor,
                            "qualityViolation": quality_violation,
                            "terminalBudgetViolation": marginal_budget_violation,
                            "commitProgressFloor": args.min_commit_progress_rate,
                            "teacherLocalGate": teacher_gate,
                            "globalBlockedOk": global_blocked_ok,
                            "marginalBlockedRate": metrics.get("marginalBlockedRate"),
                            "marginalBlockedActions": metrics.get("marginalBlockedActions"),
                            "allowedMarginalBlockedActions": marginal_quality["allowed"],
                            "stepGuardOk": step_guard_ok,
                            "rewindStepDrop": rewind_step_drop,
                            "maxRewindStepDrop": args.max_rewind_step_drop,
                            "transactionRewoundBeforeRepair": rewound_transaction,
                            "terminalRescueRewrite": terminal_rescue_rewrite,
                            "corridorBuild": corridor_build_report,
                            "corridorPlanActions": (
                                [ACTION_ORDER[index].value for index in corridor_plan_actions]
                                if candidate_source == "astar-corridor"
                                else None
                            ),
                        }
                        trials.append(trial)

                        print(
                            f"    action={ACTION_ORDER[candidate_action].value:<8} "
                            f"source={candidate_source:<16} alpha={alpha:g} -> "
                            f"steps={metrics['steps']}/{args.target} score={metrics['score']:.0f} "
                            f"terminal={metrics['terminalReason']} "
                            f"dScore={float(metrics['score']) - committed_score:+.0f} "
                            f"qAbort={int(quality_violation)} "
                            f"mBlock={int(metrics.get('marginalBlockedActions', 0))}/{int(metrics.get('marginalSteps', 0))} "
                            f"tSlow={(teacher_gate['slowdown'] if teacher_gate['slowdown'] is not None else float('inf')):.2f}x "
                            f"active={len(active_indices)} +w={len(witnesses)} "
                            f"MaleCNS(old={runtime['recurrentReplaySteps']}, "
                            f"suffix={runtime['suffixMaleCNSSteps']}, "
                            f"skipped={runtime['prefixMaleCNSStepsAvoided']}) "
                            f"commit={int(commit_eligible)} bridge={int(bridge_eligible)}",
                            flush=True,
                        )

                        choice_metadata = {
                            "targetAction": ACTION_ORDER[candidate_action].value,
                            "targetActionIndex": int(candidate_action),
                            "targetActionSource": candidate_source,
                            "protectionStrength": strength,
                            "alpha": alpha,
                            "choiceSignature": choice_signature,
                            "frontierStateKey": frontier_state_key,
                            "delta": delta_report,
                            "initialProtectionStates": len(initial_protection_indices),
                            "finalProtectionStates": len(active_indices),
                            "activeWitnesses": witnesses,
                            "activeRounds": active_rounds,
                            "protectedGuardRepairs": protected_guard_repairs,
                            "runtime": runtime,
                            "teacherLocalGate": teacher_gate,
                            "marginalBlockQuality": marginal_quality,
                            "corridorBuild": corridor_build_report,
                            "corridorPlanActions": (
                                [ACTION_ORDER[index].value for index in corridor_plan_actions]
                                if candidate_source == "astar-corridor"
                                else None
                            ),
                        }

                        if commit_eligible:
                            key = selection_key(metrics, args.target)
                            if best_commit_key is None or key > best_commit_key:
                                best_commit_key = key
                                best_commit_metrics = metrics
                                best_commit_trace = trial_trace
                                best_commit_cache = trial_cache
                                best_commit_state = snapshot_decoder(decoder)
                                best_commit_choice = choice_metadata
                        elif bridge_eligible:
                            corridor_astar_bridge = bool(
                                candidate_source == "astar-corridor"
                                and teacher_report.get("method") == "astar"
                            )
                            genuine_astar_bridge = bool(
                                candidate_source in {"astar-corridor", "astar-preferred"}
                                and teacher_report.get("method") == "astar"
                            )
                            key = (
                                float(corridor_astar_bridge),
                                float(genuine_astar_bridge),
                                float(not quality_violation),
                                float(metrics["score"]),
                                float(metrics["steps"]),
                                -float(metrics["maxNoProgressStreak"]),
                                -float(metrics["blockedActions"]),
                                -float(metrics["pingPongReturns"]),
                            )
                            if best_bridge_key is None or key > best_bridge_key:
                                best_bridge_key = key
                                best_bridge_metrics = metrics
                                best_bridge_trace = trial_trace
                                best_bridge_cache = trial_cache
                                best_bridge_state = snapshot_decoder(decoder)
                                best_bridge_choice = choice_metadata

            if best_commit_state is not None:
                restore_decoder(decoder, best_commit_state)
                current_metrics = best_commit_metrics
                current_trace = best_commit_trace
                current_cache = best_commit_cache
                assert current_metrics is not None and current_trace is not None and current_cache is not None
                assert best_commit_choice is not None
                committed_decoder = snapshot_decoder(decoder)
                committed_metrics = copy.deepcopy(current_metrics)
                committed_trace = current_trace
                committed_cache = current_cache
                # A score commit becomes the new reference for marginal blocked quality.
                # Without this reset, later candidates inherit the previous segment's
                # spent blocked-action budget and can be rejected immediately.
                quality_anchor_steps = int(committed_metrics["steps"])
                quality_anchor_blocked = int(committed_metrics["blockedActions"])
                bridge_depth = 0
                bridge_frontiers = []
                bridge_path = []
                root_frontier_lookback_current = max(1, int(args.terminal_rescue_lookback))
                root_frontier_attempts = []
                banned_choices.clear()
                search_restarts = 0
                success = quality_success(
                    current_metrics,
                    args.target,
                    max_no_progress=args.stall_abort,
                    max_blocked_rate=args.max_blocked_rate,
                    min_progress_rate=args.min_commit_progress_rate,
                    max_marginal_blocked_rate=args.max_marginal_blocked_rate,
                )
                print(
                    f"  COMMIT SCORE GAIN: action={best_commit_choice['targetAction']} "
                    f"({best_commit_choice['targetActionSource']}) alpha={best_commit_choice['alpha']:g} -> "
                    f"steps={current_metrics['steps']}/{args.target} "
                    f"score={current_metrics['score']:.0f} terminal={current_metrics['terminalReason']}",
                    flush=True,
                )
                print(
                    f"  QUALITY ANCHOR RESET: steps={quality_anchor_steps} "
                    f"blocked={quality_anchor_blocked}",
                    flush=True,
                )
                repairs.append(
                    {
                        "repair": repair_index,
                        "frontierStep": frontier.index,
                        "frontierScore": base_metrics["score"],
                        "terminalReason": frontier.next_terminal,
                        "studentAction": ACTION_ORDER[student_action].value,
                        "teacherAction": ACTION_ORDER[target_action].value,
                        "teacher": teacher_report,
                        "protectedStates": len(initial_protection_indices),
                        "trials": trials,
                        "decision": "commit-score-gain",
                        "transactionRewind": rewind_info,
                        "bridgeDepthAfter": 0,
                        "selected": {**best_commit_choice, "metrics": current_metrics},
                    }
                )
                save_checkpoint(
                    args.out / "best.pt",
                    policy=policy,
                    decoder=decoder,
                    source_checkpoint=args.v4_checkpoint,
                    source_decoder=args.decoder_checkpoint,
                    stage=f"readout-cached-frontier-commit-{repair_index}",
                    metrics={"exactExpo": current_metrics},
                    config=config,
                )
                autosave(repair_index + 1)
                continue

            if best_bridge_state is not None and bridge_depth < args.max_bridge_repairs:
                restore_decoder(decoder, best_bridge_state)
                current_metrics = best_bridge_metrics
                current_trace = best_bridge_trace
                current_cache = best_bridge_cache
                assert current_metrics is not None and current_trace is not None and current_cache is not None
                assert best_bridge_choice is not None
                bridge_depth += 1
                bridge_frontiers.append(frontier_index)
                bridge_path.append(
                    {
                        "stateKey": frontier_state_key,
                        "frontierIndex": frontier_index,
                        "choice": tuple(best_bridge_choice["choiceSignature"]),
                        "targetAction": best_bridge_choice["targetAction"],
                        "targetActionSource": best_bridge_choice["targetActionSource"],
                        "protectionStrength": float(best_bridge_choice["protectionStrength"]),
                        "alpha": float(best_bridge_choice["alpha"]),
                    }
                )
                # V12.16: entering a provisional branch must not erase which
                # committed-root frontiers were already exhausted. If this branch
                # later backtracks, the root sweep resumes exactly where it left off.
                print(
                    f"  TENTATIVE BRIDGE {bridge_depth}/{args.max_bridge_repairs}: "
                    f"action={best_bridge_choice['targetAction']} "
                    f"({best_bridge_choice['targetActionSource']}) alpha={best_bridge_choice['alpha']:g} -> "
                    f"steps={current_metrics['steps']}/{args.target} "
                    f"score={current_metrics['score']:.0f} terminal={current_metrics['terminalReason']}",
                    flush=True,
                )
                repairs.append(
                    {
                        "repair": repair_index,
                        "frontierStep": frontier.index,
                        "frontierScore": base_metrics["score"],
                        "terminalReason": frontier.next_terminal,
                        "studentAction": ACTION_ORDER[student_action].value,
                        "teacherAction": ACTION_ORDER[target_action].value,
                        "teacher": teacher_report,
                        "protectedStates": len(initial_protection_indices),
                        "trials": trials,
                        "decision": "tentative-bridge",
                        "transactionRewind": rewind_info,
                        "bridgeDepthAfter": bridge_depth,
                        "bridgeFrontiersAfter": list(bridge_frontiers),
                        "selected": {**best_bridge_choice, "metrics": current_metrics},
                    }
                )
                # Unlike V11, this provisional branch is now recoverable after a
                # shutdown; best.pt still remains the last quality-valid commit.
                autosave(repair_index + 1)
                continue

            reason = (
                "bridge-budget-exhausted" if best_bridge_state is not None else "no-admissible-candidate"
            )
            if backtrack_transaction(reason, repair_index):
                continue

            restore_decoder(decoder, committed_decoder)
            current_metrics = copy.deepcopy(committed_metrics)
            current_trace = committed_trace
            current_cache = committed_cache

            # V12.16: a dead end at the committed root is not global search exhaustion.
            # Mark only this frontier as tried and move to another A* disagreement in
            # the same rescue window; when that window is exhausted, the controller
            # expands it up to --root-frontier-lookback. Quality/bridge gates are not relaxed.
            if bridge_depth == 0:
                if int(frontier_index) not in root_frontier_attempts:
                    root_frontier_attempts.append(int(frontier_index))
                print(
                    f"  ROOT FRONTIER REJECTED: {reason} at step {frontier_index}. "
                    f"Will try another frontier (tried={len(root_frontier_attempts)}, "
                    f"lookback={root_frontier_lookback_current}).",
                    flush=True,
                )
                repairs.append(
                    {
                        "repair": repair_index,
                        "frontierStep": frontier.index,
                        "frontierScore": base_metrics["score"],
                        "terminalReason": frontier.next_terminal,
                        "studentAction": ACTION_ORDER[student_action].value,
                        "teacherAction": ACTION_ORDER[target_action].value,
                        "teacher": teacher_report,
                        "protectedStates": len(initial_protection_indices),
                        "trials": trials,
                        "decision": "root-frontier-rejected",
                        "transactionRewind": rewind_info,
                        "rollbackReason": reason,
                        "rootFrontierAttempts": list(root_frontier_attempts),
                        "rootLookback": int(root_frontier_lookback_current),
                        "selected": None,
                    }
                )
                autosave(repair_index + 1)
                continue

            print(
                f"  ROLLBACK TRANSACTION: {reason}. Search alternatives exhausted; back to "
                f"steps={current_metrics['steps']} score={current_metrics['score']:.0f}.",
                flush=True,
            )
            repairs.append(
                {
                    "repair": repair_index,
                    "frontierStep": frontier.index,
                    "frontierScore": base_metrics["score"],
                    "terminalReason": frontier.next_terminal,
                    "studentAction": ACTION_ORDER[student_action].value,
                    "teacherAction": ACTION_ORDER[target_action].value,
                    "teacher": teacher_report,
                    "protectedStates": len(initial_protection_indices),
                    "trials": trials,
                    "decision": "rollback-transaction",
                    "transactionRewind": rewind_info,
                    "rollbackReason": reason,
                    "bridgeDepthBeforeRollback": bridge_depth,
                    "selected": None,
                }
            )
            bridge_depth = 0
            bridge_frontiers = []
            bridge_path = []
            stop_reason = reason
            autosave(repair_index + 1)
            break
        else:
            stop_reason = "max-repairs-exhausted"
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "keyboard-interrupt"
        # Do not overwrite run_state.pt here: Ctrl+C can arrive while the decoder
        # contains a half-tested candidate that does not match current_trace. The
        # last autosave was written only after a complete bridge/commit decision,
        # so that file is the safe transactional resume point.
        print(
            "\nINTERRUPTED: keeping the last completed autosave unchanged. "
            "Rerun the same command with --resume.",
            flush=True,
        )

    if stop_reason is None:
        stop_reason = "target-achieved" if success else "loop-ended"

    report = {
        "version": VERSION,
        "seed": args.seed,
        "fullMaleCNS": metadata,
        "decoderMetadata": decoder_metadata,
        "sourceV5Checkpoint": str(args.v5_checkpoint.resolve()),
        "sourceV5Stage": loaded_v5.get("stage"),
        "config": config,
        "baseline": baseline_metrics,
        "repairs": repairs,
        "final": current_metrics,
        "committedFinal": committed_metrics,
        "transaction": {
            "bridgeDepth": bridge_depth,
            "bridgeFrontiers": bridge_frontiers,
            "bridgePath": bridge_path,
            "provisional": bridge_depth > 0,
            "searchRestarts": search_restarts,
            "bannedChoiceCount": sum(len(choices) for choices in banned_choices.values()),
            "rootFrontierLookbackCurrent": int(root_frontier_lookback_current),
            "rootFrontierAttempts": list(root_frontier_attempts),
        },
        "success": success,
        "stopReason": stop_reason,
        "interrupted": interrupted,
        "bestCheckpoint": str((args.out / "best.pt").resolve()),
        "runState": str(run_state_path.resolve()),
        "elapsedSeconds": elapsed_total(),
        "interpretation": (
            "The measured full MaleCNS and V5 sensory/core policy remain frozen and active. "
            "Only the existing motor readout output layers are changed. V12.16 keeps the V12.10 cached-prefix/suffix architecture and makes committed-root frontier sweeping persistent across tentative bridges and DFS backtracks. V12 removes repeated "
            "step-0-to-frontier MaleCNS execution: historical readout features validate the "
            "entire unchanged action prefix analytically, while sparse exact recurrent-state "
            "snapshots reconstruct the MaleCNS state near the intended frontier and only the "
            "new suffix is simulated closed-loop. Tentative bridge transactions are autosaved "
            "separately from best.pt, so a provisional branch can be resumed after shutdown "
            "without falsely promoting it to the committed checkpoint. Quality gates are unchanged."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== FINAL ===", flush=True)
    print(
        f"current steps={current_metrics['steps']} score={current_metrics['score']:.0f} "
        f"rate={current_metrics['progressRate']:.3f} terminal={current_metrics['terminalReason']}",
        flush=True,
    )
    print(
        f"committed steps={committed_metrics['steps']} score={committed_metrics['score']:.0f} "
        f"bridgeDepth={bridge_depth} searchRestarts={search_restarts}",
        flush=True,
    )
    print(f"success={success} stopReason={stop_reason}", flush=True)
    print(f"best: {(args.out / 'best.pt').resolve()}", flush=True)
    if not args.no_autosave_state:
        print(f"resume state: {run_state_path.resolve()}", flush=True)
        print("resume command: rerun the SAME command with --resume", flush=True)
    print(f"report: {report_path.resolve()}", flush=True)

if __name__ == "__main__":
    main()
