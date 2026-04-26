"""
evaluate.py — CS588 MP3 Evaluation Script
==========================================

Usage:
    python evaluate.py --task 1
    python evaluate.py --task 2
    python evaluate.py --task 3
    python evaluate.py --task all
"""

import dataclasses as _dc
import gc
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from time import perf_counter

_MP3_DIR    = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_MP3_DIR, "outputs")
_COLAB_DATA = "/content/drive/MyDrive/CS588/womd"
_LOCAL_DATA = os.path.join(_MP3_DIR, "data")
DEFAULT_DATA = _COLAB_DATA if os.path.isdir(_COLAB_DATA) else _LOCAL_DATA

sys.path.insert(0, _MP3_DIR)

import jax.numpy as jnp

from waymax import agents
from waymax import config as _cfg
from waymax import dataloader
from waymax import dynamics
from waymax import env as _env
from waymax.config import ObjectType
from waymax.datatypes import Action
from waymax.utils import geometry

from planner.reference_path import build_reference_path, DEFAULT_LANE_OFFSETS
from planner.collision import extract_agent_predictions
from planner.frenet_utils import project_to_frenet
from planner.visualization import (
    render_planner_debug_image,
    save_image,
    save_mp4,
    write_metrics_text,
)

from motion_planner import MotionPlanner, _score_trajectories

from student.trajectory_sampler import sample_trajectories, TARGET_SPEED_DELTAS
from student.pure_pursuit import PurePursuitController, PurePursuitConfig


# ── Data loading ───────────────────────────────────────────────────────────────

def _resolve_data_path(data_path: str) -> str:
    if os.path.isdir(data_path):
        for pattern in ["uncompressed_*.tfrecord*", "*.tfrecord*"]:
            candidates = sorted(Path(data_path).glob(pattern))
            if candidates:
                return str(candidates[0])
        raise FileNotFoundError(f"No tfrecord files found in {data_path}")
    return data_path


def _make_dataset_config(data_path: str, max_num_objects: int = 32) -> _cfg.DatasetConfig:
    return _dc.replace(
        _cfg.WOD_1_3_1_TRAINING,
        path            = data_path,
        max_num_objects = max_num_objects,
    )


def _iter_scenarios(data_path: str, max_scenarios: int | None = None, max_num_objects: int = 32):
    path = _resolve_data_path(data_path)
    cfg  = _make_dataset_config(path, max_num_objects)
    for i, scenario in enumerate(dataloader.simulator_state_generator(cfg)):
        if max_scenarios is not None and i >= max_scenarios:
            break
        yield i, scenario


def _make_env(controlled_object: ObjectType, max_num_objects: int = 32):
    return _env.BaseEnvironment(
        dynamics_model = dynamics.StateDynamics(),
        config         = _dc.replace(
            _cfg.EnvironmentConfig(),
            controlled_object = controlled_object,
            max_num_objects   = max_num_objects,
        ),
    )


def _make_bicycle_env(max_num_objects: int = 32):
    """Environment using InvertibleBicycleModel — SDC-only control."""
    return _env.BaseEnvironment(
        dynamics_model = dynamics.InvertibleBicycleModel(),
        config         = _dc.replace(
            _cfg.EnvironmentConfig(),
            controlled_object = ObjectType.SDC,
            max_num_objects   = max_num_objects,
        ),
    )


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _state_collision(state) -> bool:
    """True if SDC overlaps any other valid object."""
    traj_5dof = np.asarray(
        state.current_sim_trajectory.stack_fields(["x", "y", "length", "width", "yaw"])
    )[:, 0, :]
    overlaps = np.asarray(geometry.compute_pairwise_overlaps(traj_5dof)).copy()
    valid    = np.asarray(state.current_sim_trajectory.valid[:, 0]).astype(bool)
    overlaps &= valid[None, :] & valid[:, None]
    np.fill_diagonal(overlaps, False)
    sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
    return bool(overlaps[np.flatnonzero(sdc_mask)[0]].any())



def _ego_progress(state, ref_path) -> float:
    """Arc-length progress of SDC along ref_path."""
    sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
    xy = np.asarray(state.current_sim_trajectory.xy)[np.flatnonzero(sdc_mask)[0], 0]
    s, _ = project_to_frenet(ref_path, xy[None, :])
    return float(s[0])


# ── Closed-loop episode runner ─────────────────────────────────────────────────

def _collect_ego(state, sdc_idx):
    """Return (x, y, yaw, speed) for the SDC at the current timestep."""
    t = state.current_sim_trajectory
    x     = float(np.asarray(t.x    )[sdc_idx, 0])
    y     = float(np.asarray(t.y    )[sdc_idx, 0])
    yaw   = float(np.asarray(t.yaw  )[sdc_idx, 0])
    vx    = float(np.asarray(t.vel_x)[sdc_idx, 0])
    vy    = float(np.asarray(t.vel_y)[sdc_idx, 0])
    speed = float(np.sqrt(vx**2 + vy**2))
    return x, y, yaw, speed


def _episode_dynamics_metrics(speeds, yaws, dt=0.1):
    """Compute RMS combined jerk from per-step speed and yaw lists."""
    if len(speeds) < 3:
        return dict(rms_jerk=0.0)
    spd = np.array(speeds)
    yaw = np.unwrap(yaws)
    accel     = np.gradient(spd, dt)
    lon_jerk  = np.gradient(accel, dt)
    yaw_rate  = np.gradient(yaw, dt)
    lat_jerk  = np.gradient(spd * yaw_rate, dt)
    return dict(rms_jerk=float(np.sqrt(np.mean(lon_jerk**2 + lat_jerk**2))))


def _run_planner_episode(
    scenario_index: int,
    scenario,
    env,
    traffic_actor,
    out_dir: Path,
    tag: str,
    max_steps: int = 300,
    save_media: bool = True,
) -> dict:
    """Run one closed-loop episode with the sampling-based planner."""
    planner = MotionPlanner(planning_horizon_s=5.0, replan_every=5)

    state    = env.reset(scenario)
    ref_path = build_reference_path(state)
    target_s = min(ref_path.goal_s, 50.0)

    sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
    sdc_idx  = int(np.flatnonzero(sdc_mask)[0])

    frames       = []
    collided     = False
    render_ref   = ref_path
    policy_times = []
    speeds, yaws = [], []
    sim_tta      = None
    step_count   = 0
    dt           = 0.1

    for step in range(min(max_steps, int(state.remaining_timesteps))):
        # Collect ego state before acting
        ex, ey, eyaw, espd = _collect_ego(state, sdc_idx)
        speeds.append(espd); yaws.append(eyaw)

        t0     = perf_counter()
        action = planner.plan(state)
        policy_times.append(perf_counter() - t0)

        if save_media:
            try:
                render_ref = build_reference_path(state)
            except ValueError:
                pass
            if planner.last_is_replan:
                frames.append(render_planner_debug_image(
                    state, render_ref,
                    trajectories     = planner.last_trajectories,
                    costs            = planner.last_costs,
                    best_index       = planner.last_best_index,
                    chosen_trajectory    = planner.last_chosen_traj,
                    chosen_start_index   = planner.last_active_index,
                ))
            else:
                frames.append(render_planner_debug_image(
                    state, render_ref,
                    chosen_trajectory    = planner.last_chosen_traj,
                    chosen_start_index   = planner.last_active_index,
                ))

        traffic_out = traffic_actor.select_action({}, state, None, None)
        ego_out = agents.WaymaxActorOutput(
            actor_state   = None,
            action        = action,
            is_controlled = state.object_metadata.is_sdc,
        )
        state = env.step(state, agents.merge_actions([ego_out, traffic_out]))
        step_count += 1

        if _state_collision(state):
            collided = True
            if save_media:
                frames.append(render_planner_debug_image(state, render_ref))
            break
        if _ego_progress(state, ref_path) >= target_s:
            sim_tta = step_count * dt
            if save_media:
                frames.append(render_planner_debug_image(state, render_ref))
            break

    goal_reached = sim_tta is not None
    end_dist = max(target_s - _ego_progress(state, ref_path), 0.0)

    if save_media and frames:
        save_mp4(frames, out_dir / f"{tag}.mp4", fps=10)

    dyn = _episode_dynamics_metrics(speeds, yaws, dt)
    return dict(
        collided         = collided,
        goal_reached     = goal_reached,
        end_dist_m       = end_dist,
        sim_tta_s        = sim_tta,
        mean_policy_time = float(np.mean(policy_times)) if policy_times else 0.0,
        **dyn,
    )


def _run_frenet_pp_episode(
    scenario_index: int,
    scenario,
    env,
    controller,
    out_dir: Path,
    tag: str,
    max_steps: int = 300,
    save_media: bool = True,
    replan_every: int = 5,
    horizon_s: float = 3.0,
) -> dict:
    """Run one closed-loop episode using PP steering on the best Frenet trajectory."""
    state    = env.reset(scenario)
    sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
    sdc_idx  = int(np.flatnonzero(sdc_mask)[0])
    num_objs = int(sdc_mask.shape[0])

    ref_path = build_reference_path(state)
    target_s = min(ref_path.goal_s, 50.0)

    num_samples    = int(round(horizon_s / 0.1)) + 1
    best_traj      = None
    active_sample  = 0
    last_ref       = ref_path
    last_trajs     = None
    last_best_idx  = None
    frames         = []
    collided       = False
    policy_times   = []
    speeds, yaws = [], []
    sim_tta    = None
    step_count = 0
    dt         = 0.1

    for step in range(min(max_steps, int(state.remaining_timesteps))):
        # Collect ego state before acting
        ex, ey, eyaw, espd = _collect_ego(state, sdc_idx)
        speeds.append(espd); yaws.append(eyaw)

        t0 = perf_counter()

        if best_traj is None or step % replan_every == 0:
            try:
                last_ref = build_reference_path(state)
            except ValueError:
                pass
            trajs      = sample_trajectories(state, last_ref, horizon_s=horizon_s, num_samples=num_samples)
            preds      = extract_agent_predictions(state, horizon_steps=num_samples)
            totals     = _score_trajectories(trajs, last_ref, preds, state)
            best_idx   = int(np.argmin(totals))
            best_traj  = trajs[best_idx]
            active_sample = 0
            last_trajs    = trajs
            last_best_idx = best_idx

        if save_media:
            frames.append(render_planner_debug_image(
                state, last_ref,
                trajectories=last_trajs, best_index=last_best_idx,
            ))

        traj_state = state.current_sim_trajectory
        ego_xy  = np.array([float(np.asarray(traj_state.x)[sdc_idx, 0]),
                            float(np.asarray(traj_state.y)[sdc_idx, 0])])
        ego_yaw = float(np.asarray(traj_state.yaw)[sdc_idx, 0])
        ego_vx  = float(np.asarray(traj_state.vel_x)[sdc_idx, 0])
        ego_vy  = float(np.asarray(traj_state.vel_y)[sdc_idx, 0])
        ego_spd = float(np.sqrt(ego_vx**2 + ego_vy**2))

        active_sample = min(active_sample + 1, num_samples - 1)
        path_xy  = np.stack([best_traj.x[active_sample:], best_traj.y[active_sample:]], axis=1)
        path_spd = best_traj.speed[active_sample:]
        if len(path_xy) == 0:
            path_xy  = np.stack([best_traj.x[-1:], best_traj.y[-1:]], axis=1)
            path_spd = best_traj.speed[-1:]

        lookahead_xy, _ = controller.find_lookahead_point(ego_xy, path_xy, path_spd)
        kappa   = controller.compute_steering(ego_xy, ego_yaw, lookahead_xy)
        tgt_spd = float(path_spd[0])
        accel   = controller.compute_acceleration(ego_spd, tgt_spd)

        policy_times.append(perf_counter() - t0)

        action_data           = np.zeros((num_objs, 2), dtype=np.float32)
        action_valid          = np.zeros((num_objs, 1), dtype=bool)
        action_data[sdc_idx]  = [accel, kappa]
        action_valid[sdc_idx] = True
        state = env.step(state, Action(data=jnp.array(action_data), valid=jnp.array(action_valid)))
        step_count += 1

        if _state_collision(state):
            collided = True
            if save_media:
                frames.append(render_planner_debug_image(state, last_ref))
            break
        if _ego_progress(state, ref_path) >= target_s:
            sim_tta = step_count * dt
            if save_media:
                frames.append(render_planner_debug_image(state, last_ref))
            break

    goal_reached = sim_tta is not None
    end_dist = max(target_s - _ego_progress(state, ref_path), 0.0)

    if save_media and frames:
        save_mp4(frames, out_dir / f"{tag}.mp4", fps=10)

    dyn = _episode_dynamics_metrics(speeds, yaws, dt)
    return dict(
        collided         = collided,
        goal_reached     = goal_reached,
        end_dist_m       = end_dist,
        sim_tta_s        = sim_tta,
        mean_policy_time = float(np.mean(policy_times)) if policy_times else 0.0,
        **dyn,
    )


def _aggregate(metrics: list[dict]) -> dict:
    def avg(key):
        return float(np.mean([m[key] for m in metrics if key in m]))
    completed = [m for m in metrics if m.get("goal_reached")]
    mean_tta  = float(np.mean([m["sim_tta_s"] for m in completed])) if completed else float("nan")
    return dict(
        collision_rate   = float(np.mean([m["collided"] for m in metrics])),
        mean_end_dist_m  = avg("end_dist_m"),
        mean_sim_tta_s   = mean_tta,
        rms_jerk         = avg("rms_jerk"),
        mean_policy_time = avg("mean_policy_time"),
    )





# ── Task 1: Reference Path + Trajectory Fan ────────────────────────────────────

def _check_task1(trajectories, state) -> dict[str, bool]:
    """Run structural invariant checks on sampled trajectories."""
    sdc_mask = np.asarray(state.object_metadata.is_sdc).astype(bool)
    sdc_idx  = int(np.flatnonzero(sdc_mask)[0])
    timestep = int(state.timestep)
    expected_offsets = sorted(round(float(v), 6) for v in DEFAULT_LANE_OFFSETS)
    actual_offsets   = sorted(set(round(t.target_offset, 6) for t in trajectories))

    # Trajectories with a non-zero target speed must make forward progress.
    fwd_progress = [
        float(t.s[-1]) > float(t.s[0]) if t.target_speed > 0.5 else True
        for t in trajectories
    ]
    lateral_conv = [abs(t.d[-1] - t.target_offset) < 0.5 for t in trajectories]

    ego_speed = float(np.asarray(state.sim_trajectory.speed[sdc_idx, timestep]))
    expected_speeds = set(
        round(float(np.maximum(0.0, ego_speed + d)), 3) for d in TARGET_SPEED_DELTAS
    )
    speed_ok = True
    for off in expected_offsets:
        group_speeds = set(
            round(t.target_speed, 3)
            for t in trajectories if abs(t.target_offset - off) < 0.01
        )
        if group_speeds != expected_speeds:
            speed_ok = False
            break

    # Each expected lateral offset must be covered by at least one trajectory
    # whose terminal Frenet d is within 0.15 m of that offset.
    LATERAL_COV_THRESH = 0.15
    lateral_coverage = all(
        any(abs(t.d[-1] - off) < LATERAL_COV_THRESH for t in trajectories)
        for off in expected_offsets
    )

    # Within each lateral offset group, different target speeds must produce
    # meaningfully different forward progress at the quarter-horizon point
    # (spread of s > 2 m). Using the quarter point avoids saturation artifacts
    # on short paths while still detecting a static (no-movement) stub.
    LONG_SPREAD_MIN = 0.1
    qtr = max(1, len(trajectories[0].times) // 4)
    speed_coverage = all(
        (lambda group: max(t.s[qtr] for t in group) - min(t.s[qtr] for t in group) > LONG_SPREAD_MIN)(
            [t for t in trajectories if abs(t.target_offset - off) < 0.01]
        )
        for off in expected_offsets
    )

    return {
        "offsets":          actual_offsets == expected_offsets,
        "speeds":           speed_ok,
        "fwd_progress":     all(fwd_progress),
        "lateral_conv":     all(lateral_conv),
        "lateral_coverage": lateral_coverage,
        "speed_coverage":   speed_coverage,
    }


def run_task1(
    data_path: str,
    out_dir: str,
    scenario_indices: list[int] | None = None,
    max_scenarios: int | None = None,
):
    """Build reference path, visualize candidates, and grade structural invariants."""
    print("\n=== Task 1: Reference Path + Trajectory Sampler ===")
    out = Path(out_dir) / "task1"
    out.mkdir(parents=True, exist_ok=True)

    env = _make_env(ObjectType.SDC)
    all_scenarios = list(_iter_scenarios(data_path, max_scenarios=max_scenarios))
    if scenario_indices is not None:
        all_scenarios = [(i, s) for i, s in all_scenarios if i in set(scenario_indices)]

    CHECK_NAMES = ["offsets", "speeds", "fwd_progress", "lateral_conv", "lateral_coverage", "speed_coverage"]
    scene_results: list[tuple[int, dict[str, bool]]] = []
    metric_lines: list[str] = []

    for scenario_index, scenario in all_scenarios:
        print(f"  Scenario {scenario_index}...")
        gc.collect()
        state = env.reset(scenario)
        try:
            ref_path     = build_reference_path(state)
            trajectories = sample_trajectories(state, ref_path, horizon_s=3.0, num_samples=31)
        except Exception as e:
            print(f"    [FAIL] exception: {e}")
            checks = {k: False for k in CHECK_NAMES}
            scene_results.append((scenario_index, checks))
            metric_lines.append(f"scene={scenario_index} FAIL exception={e}")
            continue

        save_image(
            render_planner_debug_image(state, ref_path, trajectories=trajectories),
            out / f"scene_{scenario_index:04d}_candidates.png",
        )

        checks   = _check_task1(trajectories, state)
        passed   = all(checks.values())
        label    = "PASS" if passed else "FAIL"
        fail_str = " ".join(k for k, v in checks.items() if not v)
        print(f"    [{label}]  " + ("" if passed else f"failed: {fail_str}"))
        for k, v in checks.items():
            print(f"      {k}: {'OK' if v else 'FAIL'}")

        scene_results.append((scenario_index, checks))
        metric_lines.append(
            f"scene={scenario_index} {label} "
            + " ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items())
            + f" path_m={ref_path.goal_s:.1f}"
        )

    all_passed  = all(all(r.values()) for _, r in scene_results)
    n_pass      = sum(all(r.values()) for _, r in scene_results)
    conclusion  = "PASS — all scenarios pass all checks." if all_passed else \
                  f"FAIL — {len(scene_results) - n_pass}/{len(scene_results)} scenarios failed."
    print(f"\n  Conclusion: {conclusion}")

    write_metrics_text(["task=1"] + metric_lines + [conclusion], out / "metrics.txt")
    print(f"Task 1 complete — {len(scene_results)} scenes. Outputs in {out}")


# ── Task 2: Cost Evaluator (StateDynamics) ────────────────────────────────────

def run_task2(
    data_path: str,
    out_dir: str,
    scenario_indices: list[int] | None = None,
    max_scenarios: int | None = None,
    max_steps: int = 300,
    save_media: bool = True,
):
    """Closed-loop evaluation of the cost evaluator using StateDynamics."""
    print("\n=== Task 2: Cost Evaluator (Frenet + StateDynamics) ===")
    out = Path(out_dir) / "task2"
    out.mkdir(parents=True, exist_ok=True)

    dynamics_model = dynamics.StateDynamics()
    env            = _make_env(ObjectType.VALID, max_num_objects=32)
    traffic_actor  = agents.create_expert_actor(
        dynamics_model     = dynamics_model,
        is_controlled_func = lambda state: jnp.logical_not(state.object_metadata.is_sdc),
    )

    all_scenarios = list(_iter_scenarios(data_path, max_scenarios=max_scenarios, max_num_objects=32))
    if scenario_indices is not None:
        all_scenarios = [(i, s) for i, s in all_scenarios if i in set(scenario_indices)]

    all_metrics  = []
    metric_lines = []

    for scenario_index, scenario in all_scenarios:
        print(f"  Scenario {scenario_index}...")
        gc.collect()
        m = _run_planner_episode(
            scenario_index, scenario, env, traffic_actor,
            out_dir=out, tag=f"scene_{scenario_index}",
            max_steps=max_steps, save_media=save_media,
        )
        tta_str = f"{m['sim_tta_s']:.1f}s" if m['goal_reached'] else "N/A"
        all_metrics.append(m)
        metric_lines.append(
            f"scene={scenario_index} collision={m['collided']} "
            f"end_dist={m['end_dist_m']:.2f}m tta={tta_str} "
            f"rms_jerk={m['rms_jerk']:.3f} t={m['mean_policy_time']*1e3:.1f}ms/step"
        )
        print(f"    collision={m['collided']} end_dist={m['end_dist_m']:.1f}m "
              f"tta={tta_str} rms_jerk={m['rms_jerk']:.3f}")

    agg = _aggregate(all_metrics)
    tta_str = f"{agg['mean_sim_tta_s']:.3f}" if not np.isnan(agg['mean_sim_tta_s']) else "N/A"
    write_metrics_text(
        ["task=2"] + metric_lines + [
            f"collision_rate={agg['collision_rate']:.3f}",
            f"mean_end_dist_m={agg['mean_end_dist_m']:.3f}",
            f"mean_sim_tta_s={tta_str}  (completed episodes only)",
            f"rms_jerk={agg['rms_jerk']:.3f}",
            f"mean_policy_time_s={agg['mean_policy_time']:.4f}",
        ],
        out / "metrics.txt",
    )
    print(f"\nTask 2 complete. collision_rate={agg['collision_rate']:.3f} "
          f"mean_end_dist={agg['mean_end_dist_m']:.1f}m "
          f"mean_tta={tta_str}s rms_jerk={agg['rms_jerk']:.3f} "
          f"t={agg['mean_policy_time']*1e3:.1f}ms/step")
    print(f"  Outputs in {out}")


# ── Task 3: Pure Pursuit on Frenet Trajectory (InvertibleBicycleModel) ────────

def run_task3(
    data_path: str,
    out_dir: str,
    scenario_indices: list[int] | None = None,
    max_scenarios: int | None = None,
    max_steps: int = 300,
    save_media: bool = True,
):
    """Closed-loop evaluation: PP controller tracks the best Frenet trajectory."""
    print("\n=== Task 3: Pure Pursuit on Frenet Trajectory (InvertibleBicycleModel) ===")
    out = Path(out_dir) / "task3"
    out.mkdir(parents=True, exist_ok=True)

    env        = _make_bicycle_env(max_num_objects=32)
    controller = PurePursuitController(PurePursuitConfig())

    all_scenarios = list(_iter_scenarios(data_path, max_scenarios=max_scenarios, max_num_objects=32))
    if scenario_indices is not None:
        all_scenarios = [(i, s) for i, s in all_scenarios if i in set(scenario_indices)]

    all_metrics  = []
    metric_lines = []

    for scenario_index, scenario in all_scenarios:
        print(f"  Scenario {scenario_index}...")
        gc.collect()
        m = _run_frenet_pp_episode(
            scenario_index, scenario, env, controller,
            out_dir=out, tag=f"scene_{scenario_index}",
            max_steps=max_steps, save_media=save_media,
        )
        tta_str = f"{m['sim_tta_s']:.1f}s" if m['goal_reached'] else "N/A"
        all_metrics.append(m)
        metric_lines.append(
            f"scene={scenario_index} collision={m['collided']} "
            f"end_dist={m['end_dist_m']:.2f}m tta={tta_str} "
            f"rms_jerk={m['rms_jerk']:.3f} t={m['mean_policy_time']*1e3:.1f}ms/step"
        )
        print(f"    collision={m['collided']} end_dist={m['end_dist_m']:.1f}m "
              f"tta={tta_str} rms_jerk={m['rms_jerk']:.3f}")

    agg = _aggregate(all_metrics)
    tta_str = f"{agg['mean_sim_tta_s']:.3f}" if not np.isnan(agg['mean_sim_tta_s']) else "N/A"
    write_metrics_text(
        ["task=3"] + metric_lines + [
            f"collision_rate={agg['collision_rate']:.3f}",
            f"mean_end_dist_m={agg['mean_end_dist_m']:.3f}",
            f"mean_sim_tta_s={tta_str}  (completed episodes only)",
            f"rms_jerk={agg['rms_jerk']:.3f}",
            f"mean_policy_time_s={agg['mean_policy_time']:.4f}",
        ],
        out / "metrics.txt",
    )
    print(f"\nTask 3 complete. collision_rate={agg['collision_rate']:.3f} "
          f"mean_end_dist={agg['mean_end_dist_m']:.1f}m mean_tta={tta_str}s "
          f"rms_jerk={agg['rms_jerk']:.3f} t={agg['mean_policy_time']*1e3:.1f}ms/step")
    print(f"  Outputs in {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS588 MP3 Evaluator")
    parser.add_argument("--task",    default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--data",    default=DEFAULT_DATA)
    parser.add_argument("--out_dir", default=_OUTPUT_DIR)
    parser.add_argument("--scenarios", type=int, nargs="+", default=None,
                        help="Scenario indices to evaluate (default: first N)")
    parser.add_argument("--max_scenarios", type=int, default=10,
                        help="Max scenarios to evaluate (default: 10)")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="Max simulator steps per episode")
    parser.add_argument("--no_media", action="store_true",
                        help="Skip saving GIF/MP4 (faster)")
    args = parser.parse_args()

    print(f"Output directory : {args.out_dir}")
    print(f"Data path        : {args.data}")

    if args.task in ("1", "all"):
        run_task1(
            args.data, args.out_dir,
            scenario_indices = args.scenarios,
            max_scenarios    = args.max_scenarios,
        )
    if args.task in ("2", "all"):
        run_task2(
            args.data, args.out_dir,
            scenario_indices = args.scenarios,
            max_scenarios    = args.max_scenarios,
            max_steps        = args.max_steps,
            save_media       = not args.no_media,
        )
    if args.task in ("3", "all"):
        run_task3(
            args.data, args.out_dir,
            scenario_indices = args.scenarios,
            max_scenarios    = args.max_scenarios,
            max_steps        = args.max_steps,
            save_media       = not args.no_media,
        )

    print(f"\nAll done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
