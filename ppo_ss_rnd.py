from __future__ import annotations
import argparse, random, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.amp import GradScaler, autocast
from collections import deque

ACTIONS = ["L22", "FW", "R22"]   # 0=left, 1=forward, 2=right
ACTIONS_W = ["L45", "FW", "R45"]  # for unwedge agent only    
# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}")

import math as _math


# ──────────────────────────────────────────────────────────────────────────────
# OBSERVATION LAYOUT
#
# Raw 18 bits:
#   [0..3]   left sonar   (near0, far0, near1, far1)
#   [4..11]  front sonar  (near0,far0, near1,far1, near2,far2, near3,far3)
#   [12..15] right sonar  (near0, far0, near1, far1)
#   [16]     IR / BUMP    (something directly in front at <4")
#   [17]     STUCK        (motor current exceeded threshold)
#
# Each network receives a FOCUSED subset of these 18 bits + derived features
# relevant only to its job.  Giving each network only what it needs keeps the
# networks small and prevents irrelevant bits from adding noise.
#
# FINDER  — needs to know about objects in all directions to navigate toward box
# PUSHER  — needs front sensors + IR to keep box centered while pushing
# UNWEDGER — needs stuck bit + all sensors to plan an escape turn
# ──────────────────────────────────────────────────────────────────────────────

# Observation dimensions per agent
FINDER_OBS_DIM   = 18   # all 18 raw bits (needs full spatial awareness)
PUSHER_OBS_DIM   = 18  # front 8 sonar bits + IR + STUCK
UNWEDGER_OBS_DIM = 18   # all sonar near-bits (4 left + 4 front-near + 4 right → 12) ... see below


def get_finder_obs(raw: np.ndarray) -> np.ndarray:
    """All 18 raw bits. Finder needs full spatial awareness to locate the box."""
    base   = raw.astype(np.float32)
    #radar  = pos_mem.danger_radar()          # shape (3,)
    return base


def get_pusher_obs(raw: np.ndarray) -> np.ndarray:
    """Front 8 sonar bits + IR + STUCK.
    Pusher only cares about what is directly ahead — is the box centered?
    Is it still in contact? Is it hitting the far wall?
    """
    base   = raw.astype(np.float32)
    #radar  = pos_mem.danger_radar()          # shape (3,)
    return base


def get_unwedger_obs(raw: np.ndarray) -> np.ndarray:
    """Near bits from all directions + IR + STUCK.
    Unwedger needs to know which direction is clear so it can turn away
    from whatever it is stuck against.
    Left near: obs[0], obs[2]
    Front near: obs[4], obs[6], obs[8], obs[10]
    Right near: obs[12], obs[14]
    IR + STUCK: obs[16], obs[17]
    """
    base   = raw.astype(np.float32)
    left_near  = raw[[0, 2]]
    front_near = raw[[4, 6, 8, 10]]
    right_near = raw[[12, 14]]
    ir_stuck   = raw[16:18]
    return base


# ──────────────────────────────────────────────────────────────────────────────
# BOX PROBE
# Runs up to PROBE_STEPS forward steps when IR fires to confirm box vs wall.
# Uses IR persistence: a box travels with the robot so IR stays high;
# a wall causes immediate stuck and IR drops as the robot angles away.
# Probe steps are NOT stored in any buffer — they are a deterministic test.
# ──────────────────────────────────────────────────────────────────────────────
PROBE_STEPS  = 6
PROBE_ACTION = "FW"

def probe_box_attached(env, current_raw: np.ndarray) -> tuple[bool, np.ndarray, float, bool]:
    """
    Returns (box_attached, last_raw_obs, total_probe_reward, episode_done)

    Decision rule:
      - Stuck on step 0           → WALL  (robot couldn't move at all)
      - IR dropped on step 0      → WALL  (moved away, sensor no longer pointing at it)
      - IR still 1 after step 1   → BOX   (box moved with the robot)
      - IR persisted + stuck later → BOX wedged against far wall (still a box)
    """
    probe_reward   = 0.0
    obs            = current_raw
    ir_persisted   = False
    stuck_on_step0 = False
    stuck_i =False
    for i in range(PROBE_STEPS):
        obs, r, done = env.step(PROBE_ACTION, render=False)
        probe_reward += r

        if done:
            return True, obs, probe_reward, True   # success — box reached boundary

        if obs[17]==1:
            stuck_i = True
            break
    box_attached= not stuck_i
    return box_attached,obs,probe_reward,False


# ──────────────────────────────────────────────────────────────────────────────
# ACTOR-CRITIC NETWORK
# Intentionally small per-agent networks — each agent solves a simple sub-task.
# Finder gets a slightly larger network since its task (exploration) is hardest.
# ──────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, n_actions: int = len(ACTIONS),
                 hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(),
                                    nn.Linear(64, n_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(),
                                    nn.Linear(64, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        f = self.backbone(x)
        return self.actor(f), self.critic(f).squeeze(-1)

    def get_action(self, x, logit_bias: torch.Tensor | None = None):
        logits, value = self(x)
        if logit_bias is not None:
            logits = logits + logit_bias
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x, actions):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


# ──────────────────────────────────────────────────────────────────────────────
# ROLLOUT BUFFER  (one per agent — each agent fills its own buffer)
# ──────────────────────────────────────────────────────────────────────────────
class RNDNetwork(nn.Module):
    def __init__(self, in_dim, out_dim=128, hidden=256):
        super().__init__()
        # Target network: Fixed, random weights
        self.target = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        # Predictor network: Trained to match target
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.predictor(x), self.target(x)

class RolloutBuffer:
    def __init__(self, horizon: int, obs_dim: int, gamma: float, lam: float):
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.gamma   = gamma
        self.lam     = lam
        self._pin    = device.type == "cuda"
        self.reset()

    def reset(self):
        self.obs      = np.zeros((self.horizon, self.obs_dim), dtype=np.float32)
        self.actions  = np.zeros(self.horizon, dtype=np.int64)
        self.rewards  = np.zeros(self.horizon, dtype=np.float32)
        self.dones    = np.zeros(self.horizon, dtype=np.float32)
        self.logprobs = np.zeros(self.horizon, dtype=np.float32)
        self.values   = np.zeros(self.horizon, dtype=np.float32)
        self.ptr      = 0

    def add(self, obs, action, reward, done, logprob, value):
        if self.ptr >= self.horizon:
            return   # buffer full — skip (shouldn't happen in normal flow)
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr]   = value
        self.ptr += 1

    def compute_gae(self, last_value: float):
        adv      = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            nv       = last_value if t == self.ptr - 1 else self.values[t + 1]
            delta    = self.rewards[t] + self.gamma * nv * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * last_gae
            adv[t]   = last_gae
        return adv, adv + self.values[:self.ptr]

    def get_batches(self, last_value: float, batch_size: int):
        if self.ptr < 2:
            return   # not enough data to update
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

        idx = torch.randperm(self.ptr, device=device)
        for start in range(0, self.ptr, batch_size):
            b = idx[start : start + batch_size]
            yield s_t[b], a_t[b], lp_t[b], adv_t[b], ret_t[b]


# ──────────────────────────────────────────────────────────────────────────────
# PPO UPDATE  (shared logic, called per agent)
# ──────────────────────────────────────────────────────────────────────────────

def ppo_update(net, opt, scaler, buffer, last_value,
               epochs, batch_size, clip_eps, vf_coef, ent_coef,
               max_grad_norm, use_amp):
    total_pg = total_vf = total_ent = n = 0.0
    for _ in range(epochs):
        for sb, ab, old_lp, adv, ret in buffer.get_batches(last_value, batch_size):
            with autocast(device_type=device.type, enabled=use_amp):
                new_lp, entropy, values = net.evaluate(sb, ab)
                ratio    = torch.exp(new_lp - old_lp)
                pg_loss  = torch.max(
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

    return (total_pg / n, total_vf / n, total_ent / n) if n > 0 else (0, 0, 0)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _raw(net):
    return net._orig_mod if hasattr(net, "_orig_mod") else net

def save_checkpoint(path, agents, episode, steps):
    checkpoint_data = {
        "episode": episode,
        "steps":   steps,
    }
    
    for name, ag in agents.items():
        # Save standard PPO components
        checkpoint_data[f"{name}_net"] = _raw(ag["net"]).state_dict()
        checkpoint_data[f"{name}_opt"] = ag["opt"].state_dict()
        checkpoint_data[f"{name}_scaler"] = ag["scaler"].state_dict()
        
        # Save RND components if they exist (Finder)
        if ag.get("rnd") is not None:
            checkpoint_data[f"{name}_rnd"] = _raw(ag["rnd"]).state_dict()
            checkpoint_data[f"{name}_rnd_opt"] = ag["rnd_opt"].state_dict()

    torch.save(checkpoint_data, path)
    print(f"\n✅ Checkpoint saved → {path}  (ep {episode})")

def load_checkpoint(path, agents):
    ck = torch.load(path, map_location=device, weights_only=False)
    
    for name, ag in agents.items():
        # Load standard PPO components
        _raw(ag["net"]).load_state_dict(ck[f"{name}_net"])
        ag["opt"].load_state_dict(ck[f"{name}_opt"])
        if f"{name}_scaler" in ck:
            ag["scaler"].load_state_dict(ck[f"{name}_scaler"])
            
        # Load RND components if they exist in the agent AND the file
        if ag.get("rnd") is not None:
            if f"{name}_rnd" in ck:
                _raw(ag["rnd"]).load_state_dict(ck[f"{name}_rnd"])
                print(f"  [{name}] RND networks loaded")
            if f"{name}_rnd_opt" in ck:
                ag["rnd_opt"].load_state_dict(ck[f"{name}_rnd_opt"])
                print(f"  [{name}] RND optimizer loaded")

    print(f"✅ Loaded checkpoint from {path}")
    return ck["steps"], ck["episode"]


# ──────────────────────────────────────────────────────────────────────────────
# ARENA CURRICULUM
# ──────────────────────────────────────────────────────────────────────────────

def get_arena_size(ep, total_eps, start_size, end_size):
    if total_eps <= 1 or start_size == end_size:
        return end_size
    t = ep / (total_eps - 1)
    return max(1, int(round(start_size * (end_size / start_size) ** t)))


# ──────────────────────────────────────────────────────────────────────────────
# IMPORT OBELIX
# ──────────────────────────────────────────────────────────────────────────────

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

    # Environment
    ap.add_argument("--obelix_py",      type=str, required=True)
    ap.add_argument("--episodes",       type=int, default=3000)
    ap.add_argument("--max_steps",      type=int, default=1000)
    ap.add_argument("--difficulty",     type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size",     type=int, default=500)

    # Arena curriculum
    ap.add_argument("--arena_curriculum",  action="store_true")
    ap.add_argument("--arena_size_start",  type=int, default=300)

    # PPO shared hyper-parameters
    ap.add_argument("--gamma",         type=float, default=0.99)
    ap.add_argument("--lam",           type=float, default=0.95)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--horizon",       type=int,   default=2048)
    ap.add_argument("--batch",         type=int,   default=64)
    ap.add_argument("--epochs",        type=int,   default=10)
    ap.add_argument("--clip_eps",      type=float, default=0.2)
    ap.add_argument("--vf_coef",       type=float, default=0.5)
    ap.add_argument("--ent_coef",      type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)

    # Exploration
    ap.add_argument("--fw_explore_start", type=float, default=0.4,
                    help="Initial prob of forcing FW during finder mode")
    ap.add_argument("--fw_explore_end",   type=float, default=0.1,
                    help="Final prob of forcing FW during finder mode")

    # Misc
    ap.add_argument("--seed",       type=int,  default=0)
    ap.add_argument("--out_prefix", type=str,  default="weights")
    ap.add_argument("--resume",     type=str,  default=None)
    ap.add_argument("--no_amp",     action="store_true")
    ap.add_argument("--no_compile", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    use_amp = (device.type == "cuda") and (not args.no_amp)


    # ── Build the three agents ────────────────────────────────────────────────
    #
    # FINDER   — largest network, hardest task (exploration)
    # PUSHER   — medium network, simpler task (straight-line push)
    # UNWEDGER — small network, simplest task (escape turn)
    #
    def make_agent(obs_dim, hidden, use_rnd=False):
        net    = ActorCritic(obs_dim, hidden=hidden).to(device)
        rnd = None
        rnd_opt = None
        if use_rnd:
          rnd = RNDNetwork(obs_dim).to(device)
          rnd_opt = optim.Adam(rnd.predictor.parameters(), lr=1e-4)
        if (not args.no_compile) and device.type == "cuda":
            try:
                net = torch.compile(net)
                if rnd: rnd = torch.compile(rnd)
                print(f"[opt] torch.compile enabled for dim={obs_dim}")
            except Exception as e:
                print(f"[opt] torch.compile unavailable: {e}")
        opt    = optim.Adam(_raw(net).parameters(), lr=args.lr, eps=1e-5)
        scaler = GradScaler("cuda", enabled=use_amp)
        buf    = RolloutBuffer(args.horizon, obs_dim, args.gamma, args.lam)
        sbuf   = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
        return {"net": net, "opt": opt, "scaler": scaler,
                "buf": buf, "sbuf": sbuf, "obs_dim": obs_dim, "rnd": rnd, "rnd_opt": rnd_opt}

    agents = {
        "finder":   make_agent(FINDER_OBS_DIM,   hidden=256,use_rnd=True),
        "pusher":   make_agent(PUSHER_OBS_DIM,   hidden=128),
        "unwedger": make_agent(UNWEDGER_OBS_DIM, hidden=64),
    }
    print(f"[opt] AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"[agents] finder={FINDER_OBS_DIM}d  pusher={PUSHER_OBS_DIM}d  unwedger={UNWEDGER_OBS_DIM}d")

    steps    = 0
    start_ep = 0

    if args.resume:
        steps, start_ep = load_checkpoint(args.resume, agents)

    # ── Obs → GPU helper (per agent) ─────────────────────────────────────────
    def to_device(ag, obs_np: np.ndarray) -> torch.Tensor:
        ag["sbuf"][0].copy_(torch.from_numpy(obs_np), non_blocking=True)
        return ag["sbuf"]

    # ── New environment ───────────────────────────────────────────────────────
    def new_env(ep_idx):
        cur_size = (get_arena_size(ep_idx, 500,
                                   args.arena_size_start, args.arena_size)
                    if args.arena_curriculum else args.arena_size)
        ep_seed = args.seed * 100003 + ep_idx
        if ep_idx >= 500:
            cur_size = 500  # hard transition at ep 500 for testing
        e = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=500,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=ep_seed,
        )
        raw = e.reset(seed=ep_seed)
        return e, raw, cur_size

    if args.arena_curriculum:
        print(f"[curriculum] {args.arena_size_start} → {args.arena_size} "
              f"over {args.episodes} episodes")

    # ── Episode state ─────────────────────────────────────────────────────────
    ep          = start_ep
    ep_reward   = 0.0
    ep_steps    = 0
    ep_finder   = 0     # steps each agent was active this episode
    ep_pusher   = 0
    ep_unwedger = 0

    env, raw, cur_arena = new_env(ep)

    # ── Shaping state ─────────────────────────────────────────────────────────
    # box_attached: confirmed by probe, persists until IR drops for > grace steps
    box_attached      = False
    push_grace        = 0          # steps remaining in push grace period
    PUSH_GRACE_STEPS  = 7          # from the paper
    unwedge_steps    =0
    unwedge_active    = False
    unwedge_grace     = 0
    UNWEDGE_GRACE_STEPS = 5     # from the paper

    # Anti-spin tracker for finder
    action_history = deque(maxlen=20)

    rewards_history = []
    start_time      = time.time()

    # ── Horizon loop ──────────────────────────────────────────────────────────
    try:
        while ep < args.episodes:

            # Reset all buffers at the start of each horizon
            for ag in agents.values():
                ag["buf"].reset()

            horizon_steps = 0

            while horizon_steps < args.horizon:

                # ── PRIORITY NETWORK ─────────────────────────────────────────
                # Exactly as in the paper:
                #   1. Unwedger has highest priority (fires when STUCK)
                #   2. Pusher has middle priority (fires when box attached)
                #   3. Finder has lowest priority (fires otherwise)
                #
                # Grace periods (from the paper): each behavior stays active
                # for N steps after its trigger condition clears, so the agent
                # has time to recover rather than abruptly switching modes.

                ir_on     = bool(raw[16] == 1)
                stuck_on  = bool(raw[17] == 1)

                # Update unwedge grace
                if stuck_on:
                    unwedge_active = True
                    unwedge_grace  = UNWEDGE_GRACE_STEPS
                elif unwedge_grace > 0:
                    unwedge_grace -= 1
                    if unwedge_grace == 0:
                        unwedge_active = False

                # Update push grace (only if not already probed as box)
                
                if ir_on:
                        push_grace = PUSH_GRACE_STEPS   # IR on → refresh grace
                elif push_grace > 0:
                        push_grace -= 1

                # Select active agent by priority
                if unwedge_active:
                    mode = "unwedger"
                elif push_grace > 0:
                    mode = "pusher"
                else:
                    mode = "finder"

                ag = agents[mode]

                # ── Get mode-specific observation ────────────────────────────
                if mode == "finder":
                    obs = get_finder_obs(raw)
                elif mode == "pusher":
                    obs = get_pusher_obs(raw)
                else:
                    obs = get_unwedger_obs(raw)

                # ── Inference ────────────────────────────────────────────────
                with torch.no_grad():
                    st = to_device(ag, obs)
                    with autocast(device_type=device.type, enabled=use_amp):

                        if mode == "finder":
                            # Forward-biased exploration during find phase —
                            # decays from fw_explore_start to fw_explore_end
                            # over first 50% of training
                            #explore_t = min(1.0, ep / max(1, int(0.5 * args.episodes)))
                            #fw_prob   = (args.fw_explore_start
                                         #+ (args.fw_explore_end - args.fw_explore_start)
                                         #* explore_t)
                            
                            #if random.random() < fw_prob:
                                # Force FW — still compute value for buffer
                                #_, log_prob, _, value = ag["net"].get_action(st)
                                #a_idx = 1   # FW override
                            #else:
                            bias = torch.tensor([-0.25, 1, -0.25], device=device)
                            action, log_prob, _, value = ag["net"].get_action(st)
                            a_idx = int(action.item())

                        elif mode == "pusher":
                            # Pusher: soft bias toward FW (keeping the box)
                            # via logit adjustment — not a hard override
                            bias = torch.tensor([-1.0, 1.0,-1.0], device=device)
                            action, log_prob, _, value = ag["net"].get_action(st)
                            a_idx = int(action.item())

                        else:  # unwedger
                            # Unwedger: no bias — needs to freely choose turn direction
                            action, log_prob, _, value = ag["net"].get_action(st)
                            a_idx = int(action.item())

                # ── Step environment ─────────────────────────────────────────
                if mode == "unwedger":
                    raw2, env_r, done = env.step(ACTIONS_W[a_idx], render=True)
                else:
                    raw2, env_r, done = env.step(ACTIONS[a_idx], render=True)
                #raw2, env_r, done = env.step(ACTIONS[a_idx], render=False)

                wstuck = bool(raw2[17] == 1)

                # ── IR probe when IR fires and not already in box mode ────────
                # Run the confirmation probe to distinguish box from wall.
                # Probe steps are NOT stored in the buffer.
                
                if ir_on and not box_attached and not unwedge_active and not done:
                    confirmed, probe_raw, probe_r, probe_done = \
                        probe_box_attached(env, raw2)
                    box_attached = confirmed
                    if confirmed:
                        push_grace = PUSH_GRACE_STEPS
                        print("attached")
                    # Account for probe in episode stats but not in buffer
                    ep_reward += probe_r
                    ep_steps  += PROBE_STEPS
                    steps     += PROBE_STEPS
                    horizon_steps += PROBE_STEPS
                    if probe_done:
                        raw2 = probe_raw
                        done = True
                    else:
                        raw2 = probe_raw

                # ── Reward shaping per mode ───────────────────────────────────
                #
                # FINDER rewards
                #   + forward displacement if moving
                #   - spinning penalty
                #   - scaled wall penalty (soft early in training)
                #
                # PUSHER rewards
                #   + each forward step while holding box
                #   - turning (risks losing box)
                #   - losing IR contact
                #
                # UNWEDGER rewards
                #   + getting unstuck (IR and STUCK both clear)
                #   - remaining stuck
                #   (wall penalty still applies from env_r)
                #
                WALL_PENALTY_FULL = -201
                curriculum_t  = min(1.0, ep / max(1, int(0.4 * args.episodes)))

                r = env_r

                # Soften wall penalty early in training for all modes
                if r == WALL_PENALTY_FULL:
                    r = WALL_PENALTY_FULL * (0.15 + 0.85 * curriculum_t)
                    #print(f"Wall penalty softened to {r:.1f} (curriculum t={curriculum_t:.2f})")
                if mode == "finder":
                    with torch.no_grad():
                        st_rnd = to_device(ag, obs)
                        pred, target = ag["rnd"](st_rnd)
                        intrinsic_r = (pred - target).pow(2).mean().item()
                    r += 0.5 * intrinsic_r
                    #print(intrinsic_r)  # scale intrinsic reward
                    
                    """action_history.append(a_idx)
                    if len(action_history) == action_history.maxlen:
                        fwd_ratio = action_history.count(1) / action_history.maxlen
                        if fwd_ratio >= 0.4:
                            r += 1
                        elif fwd_ratio <= 0.1:
                            r -= 3"""
                    #if raw2[4]==1 or raw2[6]==1 or raw2[8]==1 or raw2[10]==1 and raw2[17]==0:
                        #r+=5
                    #if was_stuck and not now_stuck:
                        #r +=  10.0    # escaped  (was +5)
                    #elif now_stuck:
                        # -=5.0  

                    # Reward confirmed box contact
                    #if a_idx == 1 and not pos_memory.next_cell_danger():
                        #r+=5
                        #print("danger")
                    if raw2[16]==1 and raw2[17]==0:
                        r+=20
                    if raw2[4]==1 or raw2[6]==1 or raw2[8]==1 or raw2[10]==1 and raw2[17]==0:
                        r+=10
                    
                    

                elif mode == "pusher":
                    if a_idx == 1 and raw[16]==1 and raw2[17]==0 and box_attached:    # FW
                        r += 3.0
                    else:
                        r -= 3.0      # turning while pushing
                    

                else:   # unwedger
                    now_stuck = bool(raw2[17] == 1)
                    unwedge_steps += 1
 
                    if now_stuck:
                        # Still stuck — small per-step penalty so it doesn't
                        # accumulate too fast but still signals urgency.
                        # Scales down after first few steps so the agent isn't
                        # immediately overwhelmed before it can turn.
                        r -= min(1.0 * unwedge_steps, 5.0)
                    elif now_stuck==False and raw2[16]==0:
                        # Escaped! Reward inversely proportional to how long it
                        # took — quick escape = big reward, slow = smaller.
                        # This fires exactly once: the first step where stuck clears.
                        escape_bonus = max(20.0 - unwedge_steps * 1.5, 5.0)
                        r += escape_bonus
                        unwedge_steps = 0 
                        #r =-20.0
                    if sum(raw2)>0 and raw2[16]==0:
                        r-=5    # just got stuck
                if mode == "pusher":
                  r = np.clip(r, -10.0, 10.0)
                elif mode == "unwedger":
                   r = np.clip(r, -10.0, 10.0)
                # ── Store in active agent's buffer ────────────────────────────
                ag["buf"].add(obs, a_idx, r, float(done),
                              float(log_prob.item()), float(value.item()))
                # ── Bookkeeping ───────────────────────────────────────────────
                raw        = raw2
                ep_reward += env_r
                ep_steps  += 1
                steps     += 1
                horizon_steps += 1
 # render every 20 steps
                
                

                if mode == "finder":   ep_finder   += 1
                elif mode == "pusher": ep_pusher   += 1
                else:                  ep_unwedger += 1

                if done:
                    rewards_history.append(ep_reward)
                    avg100  = np.mean(rewards_history[-100:])
                    elapsed = time.time() - start_time
                    speed   = steps / elapsed if elapsed > 0 else 0
                    print(
                        f"Ep {ep+1}/{args.episodes} | "
                        f"R: {ep_reward:.1f} | "
                        f"Avg100: {avg100:.1f} | "
                        f"Arena: {cur_arena} | "
                        f"F/P/U: {ep_finder}/{ep_pusher}/{ep_unwedger} | "
                        f"Box: {'✓' if box_attached else '✗'} | "
                        f"Speed: {speed:.0f} sps"
                    )
                    ep          += 1
                    ep_reward    = 0.0
                    ep_steps     = 0
                    ep_finder    = ep_pusher = ep_unwedger = 0
                    box_attached = False
                    push_grace   = 0
                    unwedge_active = False
                    unwedge_grace  = 0
                    action_history.clear()
                    # inside the  if done:  block, near where box_attached etc. are reset
                      # Clear position memory at the end of each episode

                    if ep >= args.episodes:
                        break

                    env, raw, cur_arena = new_env(ep)
            # ── PPO update for each agent ─────────────────────────────────────
            # Only update agents that collected enough data this horizon.
            # Bootstrap last value from whichever agent is currently active.
            with torch.no_grad():
                for name, agent in agents.items():
                    if agent["buf"].ptr < args.batch:
                        continue
                    obs_for_lv = (get_finder_obs(raw) if name == "finder" else
                                  get_pusher_obs(raw) if name == "pusher" else
                                  get_unwedger_obs(raw))
                    st_lv = to_device(agent, obs_for_lv)
                    _, lv_tensor = agent["net"](st_lv)
                    agent["last_value"] = float(lv_tensor.item()) * (1 - float(done))

            for name, agent in agents.items():
                buf = agent["buf"]
                if buf.ptr < args.batch:
                    continue   # not enough samples — skip update for this agent
                
                pg, vf, ent = ppo_update(
                    agent["net"], agent["opt"], agent["scaler"],
                    buf, agent["last_value"],
                    args.epochs, args.batch, args.clip_eps,
                    args.vf_coef, args.ent_coef, args.max_grad_norm, use_amp
                )
                if agent.get("rnd") is not None:
        # We perform one epoch of RND training using the buffer data
                    for sb, _, _, _, _ in buf.get_batches(lv, args.batch):
                        with autocast(device_type=device.type, enabled=use_amp):
                            pred, targ = agent["rnd"](sb)
                            # Predictor tries to match the fixed Random Target
                            rnd_loss = (pred - targ).pow(2).mean()
            
                        agent["rnd_opt"].zero_grad(set_to_none=True)
                        agent["scaler"].scale(rnd_loss).backward()
                        agent["scaler"].step(agent["rnd_opt"])
                        agent["scaler"].update()
            
                    print(f"  [{name}] RND Loss: {rnd_loss.item():.6f}")
                if buf.ptr > 0:
                    print(f"  [{name}] steps={buf.ptr} | "
                          f"PG:{pg:.4f} VF:{vf:.4f} Ent:{ent:.4f}")

            if ep > 0 and ep % 1 == 0:
                save_checkpoint("checkpoint_subsumption.pth", agents, ep, steps)
                print(f"Checkpoint saved at episode {ep}")

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
