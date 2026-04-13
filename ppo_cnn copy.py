"""
PPO-based subsumption controller for box-pushing robot.

Three agents (finder, pusher, unwedger) share a single environment loop.
Finder goal : cover maximum area without hitting walls; locate the box.
Pusher goal : push the box out of the boundary.
Unwedger goal: escape wall-contact situations quickly.

PositionMemory
--------------
  - Dead-reckoning from episode start (origin = 0,0 at reset).
  - Tracks two occupancy-grid channels:
      ch0 : visited cells  (1.0 = visited at least once, 0.0 = not visited)
      ch1 : wall cells     (1.0 = robot hit a wall trying to enter this cell)
  - Grid is robot-centred (11×11), north-up, east-right.
  - robot_vec = [cos(heading), sin(heading)] (2-D, no dummy constant).
  - No MEMORY_WINDOW reset — position is accumulated for the full episode.
  - No visit-count decay — the grid is a clean binary occupancy map.
"""

from __future__ import annotations
import argparse, random, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.amp import GradScaler, autocast

# ──────────────────────────────────────────────────────────────────────────────
# ACTIONS
# ──────────────────────────────────────────────────────────────────────────────
ACTIONS   = ["L22", "FW", "R22"]   # finder / pusher: 22° turns
ACTIONS_W = ["L45", "FW", "R45"]   # unwedger: wider 45° turns

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
CELL_SIZE  = 10     # 1 FW step ≈ 10 world-units → 1 cell
MAP_HALF   = 20     # 20 cells each side → 41×41 grid
MAP_SIZE   = MAP_HALF * 2 + 1   # 15
MAP_CH     = 2      # ch0=visited, ch1=wall


# ──────────────────────────────────────────────────────────────────────────────
# POSITION MEMORY  (fixed, clean)
# ──────────────────────────────────────────────────────────────────────────────
class PositionMemory:
    """
    Dead-reckoning position tracker anchored to episode start.

    Coordinate system
    -----------------
    Origin (0, 0) = robot position at episode start.
    +X  = East (initial heading direction for heading=0°).
    +Y  = North.
    Heading 0° points East; increases clockwise (matches most robot conventions).

    Occupancy grid (two binary channels)
    -------------------------------------
    ch0  visited  : 1.0 if the robot has ever been in this cell.
    ch1  wall     : 1.0 if a forward move into this cell was blocked (stuck).

    The grid is rendered robot-centred: the robot always appears at cell (H, H).
    Grid row 0 = north-most row; col 0 = west-most col.
    """

    STEP_LEN = 5.0     # world-units per FW step

    def __init__(self, cell: float = CELL_SIZE):
        self.cell = cell
        # Persistent sets for the full episode
        self._visited: set[tuple[int, int]] = set()
        self._walls:   set[tuple[int, int]] = set()
        self._reset_pose()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _reset_pose(self):
        self._x       = 0.0
        self._y       = 0.0
        self._heading = 0.0      # degrees, East=0, clockwise positive
        self._update_trig()

    def _update_trig(self):
        rad = math.radians(self._heading)
        self._cos_h = math.cos(rad)
        self._sin_h = math.sin(rad)

    def _to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World coords → integer grid cell index."""
        return (int(math.floor(x / self.cell)),
                int(math.floor(y / self.cell)))

    # ── public API ────────────────────────────────────────────────────────────

    def reset_episode(self):
        """Call once at the start of every episode."""
        self._visited.clear()
        self._walls.clear()
        self._reset_pose()
        # Mark the starting cell as visited
        self._visited.add(self._to_cell(self._x, self._y))

    def update(self, action_idx: int, was_stuck: bool,
               turn_deg: float = 22.0) -> tuple[float, float, bool]:
        """
        Update internal pose from the executed action.

        Parameters
        ----------
        action_idx : 0=Left, 1=Forward, 2=Right
        was_stuck  : True iff action_idx==1 AND raw[17]==1
        turn_deg   : rotation per turn step (22° finder/pusher, 45° unwedger)

        Returns
        -------
        (x, y, is_new_cell)
        x, y        : current position relative to episode start
        is_new_cell : True when the robot successfully moved into a never-visited cell
        """
        is_new_cell = False

        if action_idx == 0:                          # turn left
            self._heading = (self._heading - turn_deg) % 360
            self._update_trig()

        elif action_idx == 2:                        # turn right
            self._heading = (self._heading + turn_deg) % 360
            self._update_trig()

        else:                                        # forward
            next_x = self._x + self.STEP_LEN * self._cos_h
            next_y = self._y + self.STEP_LEN * self._sin_h
            next_cell = self._to_cell(next_x, next_y)

            if was_stuck:
                # Record the cell we tried but could not enter as a wall
                self._walls.add(next_cell)
            else:
                self._x, self._y = next_x, next_y
                cur_cell = self._to_cell(self._x, self._y)
                is_new_cell = cur_cell not in self._visited
                self._visited.add(cur_cell)

        return self._x, self._y, is_new_cell

    @property
    def position(self) -> tuple[float, float]:
        return self._x, self._y

    @property
    def heading(self) -> float:
        return self._heading

    def visited_count(self) -> int:
        return len(self._visited)

    def is_current_cell_revisit(self) -> bool:
        cell = self._to_cell(self._x, self._y)
        return cell in self._visited

    # ── map tensor ────────────────────────────────────────────────────────────

    def to_map_tensor(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        grid      : float32 (MAP_CH, MAP_SIZE, MAP_SIZE)
                    ch0 — visited cells (0 or 1)
                    ch1 — wall cells    (0 or 1)
        robot_vec : float32 (2,) — [cos(heading), sin(heading)]
        """
        H, S = MAP_HALF, MAP_SIZE
        grid = np.zeros((MAP_CH, S, S), dtype=np.float32)
        rcx, rcy = self._to_cell(self._x, self._y)

        # ch0: visited
        for (cx, cy) in self._visited:
            gi = H - (cy - rcy)   # north-up: higher cy → smaller row index
            gj = H + (cx - rcx)   # east-right
            if 0 <= gi < S and 0 <= gj < S:
                grid[0, gi, gj] = 1.0

        # ch1: walls
        for (cx, cy) in self._walls:
            gi = H - (cy - rcy)
            gj = H + (cx - rcx)
            if 0 <= gi < S and 0 <= gj < S:
                grid[1, gi, gj] = 1.0

        robot_vec = np.array([self._cos_h, self._sin_h], dtype=np.float32)
        return grid, robot_vec

    # ── optional visualisation ────────────────────────────────────────────────

    def render_map_grid(self, step: int = 0, mode: str = "", reward: float = 0.0):
        """Live matplotlib display of the occupancy grid (debug only)."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        H, S = MAP_HALF, MAP_SIZE
        grid_visited = np.zeros((S, S), dtype=np.float32)
        grid_walls   = np.zeros((S, S), dtype=np.float32)
        rcx, rcy     = self._to_cell(self._x, self._y)

        for (cx, cy) in self._visited:
            gi, gj = H - (cy - rcy), H + (cx - rcx)
            if 0 <= gi < S and 0 <= gj < S:
                grid_visited[gi, gj] = 1.0

        for (cx, cy) in self._walls:
            gi, gj = H - (cy - rcy), H + (cx - rcx)
            if 0 <= gi < S and 0 <= gj < S:
                grid_walls[gi, gj] = 1.0

        if not hasattr(self, "_fig") or not plt.fignum_exists(self._fig.number):
            plt.ion()
            self._fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                           gridspec_kw={"wspace": 0.3})
            self._ax_visited, self._ax_walls = axes
            self._fig.patch.set_facecolor("#12121f")

        cv = mcolors.LinearSegmentedColormap.from_list("v", ["#0d0d1f", "#378ADD"])
        cw = mcolors.LinearSegmentedColormap.from_list("w", ["#0d0d1f", "#E24B4A"])

        for ax, data, cmap, title in [
            (self._ax_visited, grid_visited, cv, "Ch0: Visited"),
            (self._ax_walls,   grid_walls,   cw, "Ch1: Walls"),
        ]:
            ax.cla(); ax.set_facecolor("#12121f")
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1,
                      interpolation="nearest", origin="upper")
            ax.set_title(title, color="white", fontsize=9)
            ax.plot(H, H, "w+", markersize=10, markeredgewidth=2, zorder=5)

        self._fig.texts.clear()
        self._fig.text(0.5, -0.02,
                       f"step {step} | mode: {mode} | heading: {self._heading:.0f}° | "
                       f"pos: ({self._x:.1f}, {self._y:.1f}) | "
                       f"visited: {len(self._visited)} | walls: {len(self._walls)} | "
                       f"r: {reward:+.2f}",
                       ha="center", fontsize=8, color="#888")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


# ──────────────────────────────────────────────────────────────────────────────
# OBSERVATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
FINDER_OBS_DIM   = 18
PUSHER_OBS_DIM   = 18
UNWEDGER_OBS_DIM = 18

ROBOT_VEC_DIM = 2   # [cos(h), sin(h)]


def get_finder_obs(raw: np.ndarray, pos_mem: PositionMemory):
    map_grid, robot_vec = pos_mem.to_map_tensor()
    return raw.astype(np.float32), map_grid, robot_vec


def get_pusher_obs(raw: np.ndarray, pos_mem: PositionMemory):
    map_grid, robot_vec = pos_mem.to_map_tensor()
    return raw.astype(np.float32), map_grid, robot_vec


def get_unwedger_obs(raw: np.ndarray, pos_mem: PositionMemory):
    map_grid, robot_vec = pos_mem.to_map_tensor()
    return raw.astype(np.float32), map_grid, robot_vec


# ──────────────────────────────────────────────────────────────────────────────
# BOX PROBE  (unchanged logic, kept clean)
# ──────────────────────────────────────────────────────────────────────────────
PROBE_STEPS  = 6
PROBE_ACTION = "FW"


def probe_box_attached(env, current_raw) -> tuple[bool, np.ndarray, float, bool]:
    """
    Drive forward PROBE_STEPS times and see if the robot gets stuck while IR is
    active — if it never gets stuck the box is confirmed attached.
    """
    probe_reward = 0.0
    obs = current_raw
    for _ in range(PROBE_STEPS):
        obs, r, done = env.step(PROBE_ACTION, render=False)
        probe_reward += r
        if done:
            return True, obs, probe_reward, True
        if obs[17] == 1:        # stuck → box not properly attached (or wall)
            return False, obs, probe_reward, False
    return True, obs, probe_reward, False


# ──────────────────────────────────────────────────────────────────────────────
# MAP CNN
# ──────────────────────────────────────────────────────────────────────────────
_CNN_OUT = 32


class MapCNN(nn.Module):
    """
    Encodes the (MAP_CH × MAP_SIZE × MAP_SIZE) occupancy grid together with
    the robot heading vector into a flat _CNN_OUT-dimensional feature.
    """
    def __init__(self, in_channels: int = MAP_CH,
                 robot_vec_dim: int = ROBOT_VEC_DIM,
                 cnn_out_dim: int = _CNN_OUT):
        super().__init__()
        # MAP_SIZE = 41 → after MaxPool2d(2) → 20 → after Conv2d(3,pad=0) → 18
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),  # (B,16,41,41)
            nn.MaxPool2d(2),                                                    # (B,16,7,7)
            nn.Conv2d(16, 32, kernel_size=3, padding=0), nn.ReLU(),            # (B,32,18,18)
        )
        # 32*18*18 = 10368
        self.fuse = nn.Sequential(
            nn.Linear(32 * 18 * 18 + robot_vec_dim, cnn_out_dim), nn.ReLU())

    def forward(self, mg: torch.Tensor, rv: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.conv(mg).flatten(1), rv], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# POLICY BASE  (shared get_action / evaluate interface)
# ──────────────────────────────────────────────────────────────────────────────
class _PolicyBase(nn.Module):
    """Mixin that provides get_action and evaluate on top of a forward()."""

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def get_action(self, x, mg=None, rv=None, logit_bias=None):
        logits, value = self(x, mg=mg, rv=rv)
        if logit_bias is not None:
            logits = logits + logit_bias
        dist = Categorical(logits=logits)
        a    = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), value

    def evaluate(self, x, actions, mg=None, rv=None):
        logits, value = self(x, mg=mg, rv=rv)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


# ──────────────────────────────────────────────────────────────────────────────
# FINDER POLICY
# Goal: cover maximum area, locate the box.
# Architecture: flat sensor branch + map-CNN branch → large backbone (256).
# Actions: ACTIONS = ["L22", "FW", "R22"]
# ──────────────────────────────────────────────────────────────────────────────
class FinderPolicy(_PolicyBase):
    N_ACTIONS = len(ACTIONS)     # 3  — fine-grained 22° turns for exploration

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.use_map     = True
        self.map_cnn     = MapCNN(in_channels=MAP_CH, cnn_out_dim=_CNN_OUT)
        fh               = max(hidden // 2, 64)
        self.flat_branch = nn.Sequential(
            nn.Linear(obs_dim, fh), nn.Tanh())
        # Wider backbone: exploration benefits from richer representations
        self.backbone    = nn.Sequential(
            nn.Linear(fh + _CNN_OUT, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),         nn.Tanh(),
            nn.Linear(hidden, hidden // 2),    nn.Tanh())
        self.actor  = nn.Sequential(
            nn.Linear(hidden // 2, 64), nn.Tanh(),
            nn.Linear(64, self.N_ACTIONS))
        self.critic = nn.Sequential(
            nn.Linear(hidden // 2, 64), nn.Tanh(),
            nn.Linear(64, 1))
        self._init_weights()

    def forward(self, x, mg=None, rv=None):
        feat = self.backbone(
            torch.cat([self.flat_branch(x), self.map_cnn(mg, rv)], dim=1))
        return self.actor(feat), self.critic(feat).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# PUSHER POLICY
# Goal: push the box out of the boundary once attached.
# Architecture: flat sensor branch + map-CNN branch → medium backbone (128).
#   The map here helps the pusher remember the boundary direction.
# Actions: ACTIONS = ["L22", "FW", "R22"]
# ──────────────────────────────────────────────────────────────────────────────
class PusherPolicy(_PolicyBase):
    N_ACTIONS = len(ACTIONS)     # 3  — same fine turns, but rewards are push-oriented

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.use_map     = True
        self.map_cnn     = MapCNN(in_channels=MAP_CH, cnn_out_dim=_CNN_OUT)
        fh               = max(hidden // 2, 64)
        self.flat_branch = nn.Sequential(
            nn.Linear(obs_dim, fh), nn.Tanh())
        # Shallower than finder — pusher task is simpler (push forward)
        self.backbone    = nn.Sequential(
            nn.Linear(fh + _CNN_OUT, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),         nn.Tanh())
        self.actor  = nn.Sequential(
            nn.Linear(hidden, 64), nn.Tanh(),
            nn.Linear(64, self.N_ACTIONS))
        self.critic = nn.Sequential(
            nn.Linear(hidden, 64), nn.Tanh(),
            nn.Linear(64, 1))
        self._init_weights()

    def forward(self, x, mg=None, rv=None):
        feat = self.backbone(
            torch.cat([self.flat_branch(x), self.map_cnn(mg, rv)], dim=1))
        return self.actor(feat), self.critic(feat).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# UNWEDGER POLICY
# Goal: escape wall contact as fast as possible.
# Architecture: flat-only (no map — no time to plan, just react).
# Actions: ACTIONS_W = ["L45", "FW", "R45"]  — wider turns to escape walls
# ──────────────────────────────────────────────────────────────────────────────
class UnwedgerPolicy(_PolicyBase):
    N_ACTIONS = len(ACTIONS_W)   # 3  — semantically different: 45° turns

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.use_map     = True
        self.map_cnn     = MapCNN(in_channels=MAP_CH, cnn_out_dim=_CNN_OUT)
        fh               = max(hidden, 32)
        self.flat_branch = nn.Sequential(
            nn.Linear(obs_dim, fh), nn.Tanh())
        # Compact backbone — unwedger must react quickly, not plan deeply
        self.backbone    = nn.Sequential(
            nn.Linear(fh + _CNN_OUT, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),         nn.Tanh())
        self.actor  = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(),
            nn.Linear(32, self.N_ACTIONS))
        self.critic = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(),
            nn.Linear(32, 1))
        self._init_weights()

    def forward(self, x, mg=None, rv=None):
        feat = self.backbone(
            torch.cat([self.flat_branch(x), self.map_cnn(mg, rv)], dim=1))
        return self.actor(feat), self.critic(feat).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# ROLLOUT BUFFER
# ──────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, horizon: int, obs_dim: int,
                 gamma: float, lam: float, use_map: bool = False):
        self.horizon  = horizon
        self.obs_dim  = obs_dim
        self.gamma    = gamma
        self.lam      = lam
        self.use_map  = use_map
        self._pin     = device.type == "cuda"
        self.reset()

    def reset(self):
        h = self.horizon
        self.obs      = np.zeros((h, self.obs_dim), dtype=np.float32)
        self.actions  = np.zeros(h, dtype=np.int64)
        self.rewards  = np.zeros(h, dtype=np.float32)
        self.dones    = np.zeros(h, dtype=np.float32)
        self.logprobs = np.zeros(h, dtype=np.float32)
        self.values   = np.zeros(h, dtype=np.float32)
        if self.use_map:
            self.map_grids  = np.zeros((h, MAP_CH, MAP_SIZE, MAP_SIZE),
                                       dtype=np.float32)
            self.robot_vecs = np.zeros((h, ROBOT_VEC_DIM), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, logprob, value,
            mg=None, rv=None):
        if self.ptr >= self.horizon:
            return
        i = self.ptr
        self.obs[i]      = obs
        self.actions[i]  = action
        self.rewards[i]  = reward
        self.dones[i]    = done
        self.logprobs[i] = logprob
        self.values[i]   = value
        if self.use_map and mg is not None and rv is not None:
            self.map_grids[i]  = mg
            self.robot_vecs[i] = rv
        self.ptr += 1

    def compute_gae(self, last_value: float):
        adv = np.zeros(self.ptr, dtype=np.float32)
        g   = 0.0
        for t in reversed(range(self.ptr)):
            nv = last_value if t == self.ptr - 1 else self.values[t + 1]
            d  = (self.rewards[t]
                  + self.gamma * nv * (1 - self.dones[t])
                  - self.values[t])
            g  = d + self.gamma * self.lam * (1 - self.dones[t]) * g
            adv[t] = g
        return adv, adv + self.values[:self.ptr]

    def get_batches(self, last_value: float, batch_size: int):
        if self.ptr < 2:
            return
        adv, ret = self.compute_gae(last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def _gpu(arr, dtype=torch.float32):
            t = torch.from_numpy(arr)
            if self._pin:
                t = t.pin_memory()
            return t.to(device, non_blocking=True).to(dtype)

        s_t   = _gpu(self.obs[:self.ptr])
        a_t   = _gpu(self.actions[:self.ptr],  dtype=torch.int64)
        lp_t  = _gpu(self.logprobs[:self.ptr])
        adv_t = _gpu(adv)
        ret_t = _gpu(ret)
        mg_t  = _gpu(self.map_grids[:self.ptr])  if self.use_map else None
        rv_t  = _gpu(self.robot_vecs[:self.ptr]) if self.use_map else None

        idx = torch.randperm(self.ptr, device=device)
        for start in range(0, self.ptr, batch_size):
            b = idx[start: start + batch_size]
            yield (s_t[b], a_t[b], lp_t[b], adv_t[b], ret_t[b],
                   mg_t[b] if mg_t is not None else None,
                   rv_t[b] if rv_t is not None else None)


# ──────────────────────────────────────────────────────────────────────────────
# PPO UPDATE
# ──────────────────────────────────────────────────────────────────────────────
def ppo_update(net, opt, scaler, buf, last_value,
               epochs, batch_size, clip_eps,
               vf_coef, ent_coef, max_grad_norm, use_amp):
    total_pg = total_vf = total_ent = n = 0.0
    for _ in range(epochs):
        for sb, ab, old_lp, adv, ret, mg_b, rv_b in buf.get_batches(last_value, batch_size):
            with autocast(device_type=device.type, enabled=use_amp):
                new_lp, entropy, values = net.evaluate(sb, ab, mg=mg_b, rv=rv_b)
                ratio   = torch.exp(new_lp - old_lp)
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                ).mean()
                vf_loss  = 0.5 * (values - ret).pow(2).mean()
                ent_loss = -entropy.mean()
                loss     = pg_loss + vf_coef * vf_loss + ent_coef * ent_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            total_pg  += pg_loss.item()
            total_vf  += vf_loss.item()
            total_ent += (-ent_loss.item())
            n         += 1

    return (total_pg / n, total_vf / n, total_ent / n) if n > 0 else (0.0, 0.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────
def _raw(net):
    return net._orig_mod if hasattr(net, "_orig_mod") else net


def save_checkpoint(path, agents, episode, steps):
    ck = {"episode": episode, "steps": steps}
    for name, ag in agents.items():
        ck[f"{name}_net"]     = _raw(ag["net"]).state_dict()
        ck[f"{name}_opt"]     = ag["opt"].state_dict()
        ck[f"{name}_scaler"]  = ag["scaler"].state_dict()
        ck[f"{name}_use_map"] = ag["use_map"]
    torch.save(ck, path)
    print(f"\n✅ Checkpoint saved → {path}  (ep {episode})")


def load_checkpoint(path, agents):
    ck = torch.load(path, map_location=device, weights_only=False)
    for name, ag in agents.items():
        saved = ck.get(f"{name}_use_map")
        if saved is not None and saved != ag["use_map"]:
            raise ValueError(f"[{name}] use_map mismatch in checkpoint.")
        _raw(ag["net"]).load_state_dict(ck[f"{name}_net"])
        ag["opt"].load_state_dict(ck[f"{name}_opt"])
        if f"{name}_scaler" in ck:
            ag["scaler"].load_state_dict(ck[f"{name}_scaler"])
    print(f"✅ Loaded checkpoint from {path}")
    return ck["steps"], ck["episode"]


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def get_arena_size(ep, total_eps, start_size, end_size):
    if total_eps <= 1 or start_size == end_size:
        return end_size
    t = ep / (total_eps - 1)
    return max(1, int(round(start_size * (end_size / start_size) ** t)))


def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",        type=str,   required=True)
    ap.add_argument("--episodes",         type=int,   default=3000)
    ap.add_argument("--max_steps",        type=int,   default=1000)
    ap.add_argument("--difficulty",       type=int,   default=0)
    ap.add_argument("--wall_obstacles",   action="store_true")
    ap.add_argument("--box_speed",        type=int,   default=2)
    ap.add_argument("--scaling_factor",   type=int,   default=5)
    ap.add_argument("--arena_size",       type=int,   default=500)
    ap.add_argument("--arena_curriculum", action="store_true")
    ap.add_argument("--arena_size_start", type=int,   default=300)
    ap.add_argument("--gamma",            type=float, default=0.99)
    ap.add_argument("--lam",              type=float, default=0.95)
    ap.add_argument("--lr",               type=float, default=3e-4)
    ap.add_argument("--horizon",          type=int,   default=2048)
    ap.add_argument("--batch",            type=int,   default=64)
    ap.add_argument("--epochs",           type=int,   default=10)
    ap.add_argument("--clip_eps",         type=float, default=0.2)
    ap.add_argument("--vf_coef",          type=float, default=0.5)
    ap.add_argument("--ent_coef",         type=float, default=0.01)
    ap.add_argument("--max_grad_norm",    type=float, default=0.5)
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--out_prefix",       type=str,   default="weights")
    ap.add_argument("--resume",           type=str,   default=None)
    ap.add_argument("--no_amp",           action="store_true")
    ap.add_argument("--no_compile",       action="store_true")
    ap.add_argument("--render_map",       action="store_true")
    ap.add_argument("--render_map_every", type=int,   default=50)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX  = import_obelix(args.obelix_py)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    pos_mem = PositionMemory()

    # ── Agent factory ─────────────────────────────────────────────────────────
    def make_agent(net_cls, obs_dim: int, hidden: int) -> dict:
        """
        Instantiate one agent dict for the given policy class.
        net_cls must be one of: FinderPolicy, PusherPolicy, UnwedgerPolicy.
        use_map is read directly from the instantiated network.
        """
        net     = net_cls(obs_dim, hidden=hidden).to(device)
        use_map = net.use_map

        if (not args.no_compile) and device.type == "cuda":
            try:
                net = torch.compile(net)
                print(f"[opt] torch.compile: {net_cls.__name__} obs_dim={obs_dim}")
            except Exception as e:
                print(f"[opt] torch.compile unavailable ({net_cls.__name__}): {e}")

        opt    = optim.Adam(_raw(net).parameters(), lr=args.lr, eps=1e-5)
        scaler = GradScaler("cuda", enabled=use_amp)
        buf    = RolloutBuffer(args.horizon, obs_dim, args.gamma, args.lam,
                               use_map=use_map)
        sbuf   = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)

        if use_map:
            msb  = torch.zeros(1, MAP_CH, MAP_SIZE, MAP_SIZE, dtype=torch.float32)
            rsb  = torch.zeros(1, ROBOT_VEC_DIM, dtype=torch.float32)
            if device.type == "cuda":
                msb = msb.pin_memory()
                rsb = rsb.pin_memory()
            mnp  = msb.numpy()
            rnp  = rsb.numpy()
            mgpu = msb.to(device, non_blocking=True) if device.type == "cuda" else msb
            rgpu = rsb.to(device, non_blocking=True) if device.type == "cuda" else rsb
        else:
            msb = mnp = mgpu = rsb = rnp = rgpu = None

        return {"net": net, "opt": opt, "scaler": scaler, "buf": buf,
                "sbuf": sbuf,
                "msb": msb, "rsb": rsb, "mnp": mnp, "rnp": rnp,
                "mgpu": mgpu, "rgpu": rgpu,
                "use_map": use_map, "obs_dim": obs_dim,
                "last_value": 0.0}

    agents = {
        "finder":   make_agent(FinderPolicy,   FINDER_OBS_DIM,   hidden=256),
        "pusher":   make_agent(PusherPolicy,   PUSHER_OBS_DIM,   hidden=128),
        "unwedger": make_agent(UnwedgerPolicy, UNWEDGER_OBS_DIM, hidden=64),
    }
    print(f"[opt] AMP={'enabled' if use_amp else 'off'}")
    print(f"[agents] FinderPolicy({FINDER_OBS_DIM}d, map=True, acts=ACTIONS)  "
          f"PusherPolicy({PUSHER_OBS_DIM}d, map=True, acts=ACTIONS)  "
          f"UnwedgerPolicy({UNWEDGER_OBS_DIM}d, map=False, acts=ACTIONS_W)")

    steps    = 0
    start_ep = 0
    if args.resume:
        steps, start_ep = load_checkpoint(args.resume, agents)

    # ── Device-transfer helpers ───────────────────────────────────────────────
    def to_dev(ag, obs_np: np.ndarray) -> torch.Tensor:
        ag["sbuf"][0].copy_(torch.from_numpy(obs_np), non_blocking=True)
        return ag["sbuf"]

    def map_to_dev(ag, mg: np.ndarray,
                   rv: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        np.copyto(ag["mnp"][0], mg,  casting="no")
        np.copyto(ag["rnp"][0], rv,  casting="no")
        if device.type == "cuda":
            ag["mgpu"].copy_(ag["msb"], non_blocking=True)
            ag["rgpu"].copy_(ag["rsb"], non_blocking=True)
        return ag["mgpu"], ag["rgpu"]

    def new_env(ep_idx: int):
        sz = (get_arena_size(ep_idx, args.episodes,
                             args.arena_size_start, args.arena_size)
              if args.arena_curriculum else args.arena_size)
        seed = args.seed * 100003 + ep_idx
        e = OBELIX(scaling_factor=args.scaling_factor,
                   arena_size=sz,
                   max_steps=args.max_steps,
                   wall_obstacles=args.wall_obstacles,
                   difficulty=args.difficulty,
                   box_speed=args.box_speed,
                   seed=seed)
        return e, e.reset(seed=seed), sz

    # ── Episode-level state ───────────────────────────────────────────────────
    PUSH_GRACE    = 7
    UNWEDGE_GRACE = 4

    ep             = start_ep
    ep_reward      = ep_steps = ep_finder = ep_pusher = ep_unwedger = 0
    box_attached   = False
    push_grace     = 0
    unwedge_steps  = 0
    unwedge_active = False
    unwedge_grace  = 0

    env, raw, cur_arena = new_env(ep)
    pos_mem.reset_episode()

    rewards_history: list[float] = []
    start_time = time.time()

    def _ep_reset():
        nonlocal ep_reward, ep_steps, ep_finder, ep_pusher, ep_unwedger
        nonlocal box_attached, push_grace, unwedge_steps, unwedge_active, unwedge_grace
        ep_reward = ep_steps = ep_finder = ep_pusher = ep_unwedger = 0
        box_attached   = False
        push_grace     = 0
        unwedge_steps  = 0
        unwedge_active = False
        unwedge_grace  = 0
        pos_mem.reset_episode()

    def _ep_log():
        avg100  = np.mean(rewards_history[-100:]) if rewards_history else 0.0
        elapsed = time.time() - start_time
        speed   = steps / elapsed if elapsed > 0 else 0
        print(
            f"Ep {ep+1}/{args.episodes} | R: {ep_reward:.1f} | "
            f"Avg100: {avg100:.1f} | Arena: {cur_arena} | "
            f"F/P/U: {ep_finder}/{ep_pusher}/{ep_unwedger} | "
            f"Box: {'✓' if box_attached else '✗'} | "
            f"Visited: {pos_mem.visited_count()} | "
            f"Walls: {len(pos_mem._walls)} | "
            f"Speed: {speed:.0f} sps"
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    try:
        while ep < args.episodes:
            for ag in agents.values():
                ag["buf"].reset()
            horizon_steps = 0

            while horizon_steps < args.horizon:
                ir_on    = bool(raw[16] == 1)
                stuck_on = bool(raw[17] == 1)

                # ── Subsumption priority ──────────────────────────────────────
                if stuck_on:
                    unwedge_active = True
                    unwedge_grace  = UNWEDGE_GRACE
                elif unwedge_grace > 0:
                    unwedge_grace -= 1
                    if unwedge_grace == 0:
                        unwedge_active = False
                        unwedge_steps  = 0   # clear stale count for next event

                if ir_on:
                    push_grace = PUSH_GRACE
                elif push_grace:
                    push_grace -= 1

                if unwedge_active:
                    mode = "unwedger"
                elif push_grace > 0:
                    mode = "pusher"
                else:
                    mode = "finder"

                ag = agents[mode]

                # ── Build observation ─────────────────────────────────────────
                if mode == "finder":
                    obs, mg, rv = get_finder_obs(raw, pos_mem)
                elif mode == "pusher":
                    obs, mg, rv = get_pusher_obs(raw, pos_mem)
                else:
                    obs, mg, rv = get_unwedger_obs(raw, pos_mem)

                # ── Inference ─────────────────────────────────────────────────
                with torch.no_grad():
                    st = to_dev(ag, obs)
                    if ag["use_map"] and mg is not None:
                        st_mg, st_rv = map_to_dev(ag, mg, rv)
                    else:
                        st_mg = st_rv = None

                    with autocast(device_type=device.type, enabled=use_amp):
                        if mode == "pusher":
                            # Tiny forward bias — agent can override freely
                            bias = torch.tensor([0.0, 1.0, 0.0], device=device)
                            act, lp, _, val = ag["net"].get_action(
                                st, mg=st_mg, rv=st_rv, logit_bias=bias)
                        else:
                            act, lp, _, val = ag["net"].get_action(
                                st, mg=st_mg, rv=st_rv)
                    a_idx = int(act.item())

                # ── Step environment ──────────────────────────────────────────
                action_str = ACTIONS_W[a_idx] if mode == "unwedger" else ACTIONS[a_idx]
                raw2, env_r, done = env.step(action_str, render=True)

                # Update dead-reckoning (only FW can be stuck)
                wstuck   = bool(raw2[17] == 1) and (a_idx == 1)
                turn_deg = 45.0 if mode == "unwedger" else 22.0
                _, _, is_new_cell = pos_mem.update(a_idx, wstuck, turn_deg=turn_deg)

                # Optional map render
                if args.render_map and steps % args.render_map_every == 0:
                    pos_mem.render_map_grid(step=steps, mode=mode, reward=env_r)

                # ── Box-attach probe ──────────────────────────────────────────
                if ir_on and not box_attached and not unwedge_active and not done:
                    confirmed, probe_raw, probe_r, probe_done = \
                        probe_box_attached(env, raw2)
                    box_attached = confirmed
                    if confirmed:
                        push_grace = PUSH_GRACE
                        print("[finder] Box attached!")
                    ep_reward    += probe_r
                    ep_steps     += PROBE_STEPS
                    steps        += PROBE_STEPS
                    horizon_steps += PROBE_STEPS
                    raw2 = probe_raw
                    if probe_done:
                        done = True

                # ── Reward shaping ────────────────────────────────────────────
                r = env_r

                if mode == "finder":
                    if a_idx == 1 and not wstuck:
                        if is_new_cell:
                            if raw2[16] == 1:
                              r += 10.0
                            elif raw2[4] == 1 or raw2[6] == 1 or raw2[8] == 1 or raw2[10] == 1:
                                r += 5
                            else:
                                r+=5                      # strong bonus: new territory
                        else:
                            # Revisit penalty: scales with how much area is already known
                            explored_ratio = pos_mem.visited_count() / max(
                                pos_mem.visited_count() + 1, 1)
                            r -= 3.0 * explored_ratio      # gentle until map fills up
                    if wstuck:
                        r -= 1.0
                    if raw2[4] == 1 or raw2[6] == 1 or raw2[8] == 1 or raw2[10] == 1:
                                r += 30
                    if raw2[16] == 1:
                                r += 50
                elif mode == "pusher":
                    # Reward forward push; penalise stalling or turning while attached
                    if a_idx == 1 and raw[16] == 1 and raw2[17] == 0 and box_attached:
                        r += 3.0
                    else:
                        r -= 1.0
                    r = float(np.clip(r, -10.0, 10.0))

                else:  # unwedger
                    still_stuck = bool(raw2[17] == 1)
                    just_moved  = (a_idx == 1 and not still_stuck)   # FW succeeded
                    ir_still_on = bool(raw2[16] == 1)                 # still near wall/box

                    if still_stuck:
                        # Still wedged: escalating penalty to drive the agent to try something
                        unwedge_steps += 1
                        r -= min(0.5 * unwedge_steps, 4.0)

                    elif just_moved and not ir_still_on:
                        # Moved forward AND IR clear → genuinely away from the wall
                        r += max(15.0 - unwedge_steps * 1.0, 3.0)
                        unwedge_steps = 0   # reset for next wedge event

                    elif just_moved and ir_still_on:
                        # Moved but still touching something — partial credit, keep trying
                        r += 1.0

                    else:
                        # Turn step: small reward for attempting to reorient
                        r += 0.2

                    r = float(np.clip(r, -15.0, 15.0))

                # ── Store transition ──────────────────────────────────────────
                ag["buf"].add(obs, a_idx, r, float(done),
                              float(lp.item()), float(val.item()),
                              mg=mg, rv=rv)

                raw        = raw2
                ep_reward  += env_r
                ep_steps   += 1
                steps      += 1
                horizon_steps += 1

                if mode == "finder":   ep_finder   += 1
                elif mode == "pusher": ep_pusher   += 1
                else:                  ep_unwedger += 1

                if done:
                    rewards_history.append(ep_reward)
                    _ep_log()
                    ep += 1
                    if ep >= args.episodes:
                        break
                    _ep_reset()
                    env, raw, cur_arena = new_env(ep)

            # ── PPO update ────────────────────────────────────────────────────
            with torch.no_grad():
                for name, agent in agents.items():
                    if agent["buf"].ptr < args.batch:
                        continue
                    if name == "finder":
                        obs_lv, mg_lv, rv_lv = get_finder_obs(raw, pos_mem)
                    elif name == "pusher":
                        obs_lv, mg_lv, rv_lv = get_pusher_obs(raw, pos_mem)
                    else:
                        obs_lv, mg_lv, rv_lv = get_unwedger_obs(raw, pos_mem)



                    st_lv = to_dev(agent, obs_lv)
                    if agent["use_map"] and mg_lv is not None:
                        st_mg_lv, st_rv_lv = map_to_dev(agent, mg_lv, rv_lv)
                    else:
                        st_mg_lv = st_rv_lv = None

                    _, lv = agent["net"](st_lv, mg=st_mg_lv, rv=st_rv_lv)
                    agent["last_value"] = float(lv.item()) * (1 - float(done))

            for name, agent in agents.items():
                buf = agent["buf"]
                if buf.ptr < args.batch:
                    continue
                pg, vf, ent = ppo_update(
                    agent["net"], agent["opt"], agent["scaler"],
                    buf, agent["last_value"],
                    args.epochs, args.batch, args.clip_eps,
                    args.vf_coef, args.ent_coef, args.max_grad_norm, use_amp)
                print(f"  [{name}] steps={buf.ptr} | "
                      f"PG:{pg:.4f} VF:{vf:.4f} Ent:{ent:.4f}")

            if ep > 0 and ep % 300 == 0:
                save_checkpoint("checkpoint_subsumption.pth", agents, ep, steps)

    except KeyboardInterrupt:
        print("\nInterrupted — saving...")
        save_checkpoint("checkpoint_subsumption_interrupt.pth", agents, ep, steps)

    # ── Save final weights ────────────────────────────────────────────────────
    for name, agent in agents.items():
        path = f"{args.out_prefix}_{name}.pth"
        torch.save(_raw(agent["net"]).state_dict(), path)
        print(f"Saved {name} → {path}")


if __name__ == "__main__":
    main()