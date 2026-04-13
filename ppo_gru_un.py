from __future__ import annotations
import argparse, random, time, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.amp import GradScaler, autocast
from collections import deque

ACTIONS   = ["L22", "FW", "R22"]    # 0=left, 1=forward, 2=right
ACTIONS_W = ["L45", "FW", "R45"]    # for unwedge agent only
N_ACTIONS_W = len(ACTIONS_W)        # 3

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"[device] GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# OBSERVATION DIMENSIONS
#
# Raw 18 bits:
#   [0..3]   left sonar   (near0, far0, near1, far1)
#   [4..11]  front sonar  (near0,far0, near1,far1, near2,far2, near3,far3)
#   [12..15] right sonar  (near0, far0, near1, far1)
#   [16]     IR / BUMP    (something directly in front at <4")
#   [17]     STUCK        (motor current exceeded threshold)
# ──────────────────────────────────────────────────────────────────────────────
FINDER_OBS_DIM   = 18
PUSHER_OBS_DIM   = 18
UNWEDGER_OBS_DIM = 18   # GRU sees raw obs only; temporal context lives in h


def euclidean_distance(bot_x, bot_y, box_x, box_y):
    return math.sqrt((box_x - bot_x) ** 2 + (box_y - bot_y) ** 2)


def angle_to_box_deg(bot_x, bot_y, bot_theta_deg, box_x, box_y):
    angle_diff = math.degrees(math.atan2(box_y - bot_y, box_x - bot_x)) - bot_theta_deg
    return (angle_diff + 180) % 360 - 180


def get_find_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


def get_push_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


def get_unwedge_obs(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# BOX PROBE
# ──────────────────────────────────────────────────────────────────────────────
PROBE_STEPS  = 6
PROBE_ACTION = "FW"


def probe_box_attached(env, current_raw: np.ndarray):
    """Returns (box_attached, last_raw_obs, total_probe_reward, episode_done)."""
    probe_reward = 0.0
    obs          = current_raw
    stuck_i      = False
    for _ in range(PROBE_STEPS):
        obs, r, done = env.step(PROBE_ACTION, render=False)
        probe_reward += r
        if done:
            return True, obs, probe_reward, True
        if obs[17] == 1:
            stuck_i = True
            break
    return not stuck_i, obs, probe_reward, False


# ──────────────────────────────────────────────────────────────────────────────
# MLP ACTOR-CRITIC  (find + push)
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
# GRU ACTOR-CRITIC  (unwedge only)
#
# Architecture:
#   obs (18) → Linear encoder → GRU(hidden=64) → actor + critic heads
#
# The live hidden state h (1 × 1 × GRU_HIDDEN) is:
#   - kept as ag["h"] during rollout collection
#   - stored per-step in GRURolloutBuffer so the PPO update can replay
#     exact hidden states via truncated BPTT (forward_sequence)
#   - zeroed at episode boundaries
#   - NOT reset between unwedge mode activations within an episode
#     (the GRU remembers past stuck events even after a brief mode switch)
# ──────────────────────────────────────────────────────────────────────────────
GRU_HIDDEN  = 64
GRU_CHUNK_LEN = 16   # truncated BPTT chunk length during PPO update


class GRUActorCritic(nn.Module):
    def __init__(self, obs_dim: int = UNWEDGER_OBS_DIM,
                 n_actions: int = N_ACTIONS_W,
                 enc_hidden: int = 64,
                 gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.gru_hidden = gru_hidden

        # Small MLP encoder before the GRU compresses the 18-bit obs
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, enc_hidden), nn.Tanh(),
        )
        self.gru = nn.GRU(enc_hidden, gru_hidden, batch_first=True)

        self.actor  = nn.Sequential(nn.Linear(gru_hidden, 32), nn.Tanh(),
                                    nn.Linear(32, n_actions))
        self.critic = nn.Sequential(nn.Linear(gru_hidden, 32), nn.Tanh(),
                                    nn.Linear(32, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(p, gain=math.sqrt(2))
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def zero_hidden(self) -> torch.Tensor:
        """Zeroed hidden state (1, 1, gru_hidden) on the correct device."""
        return torch.zeros(1, 1, self.gru_hidden, device=device)

    def forward_step(self, obs: torch.Tensor, h: torch.Tensor):
        """
        Single-step inference used during rollout collection.

        obs : (1, obs_dim)
        h   : (1, 1, gru_hidden)

        Returns  logits (1, n_actions),  value (1,),  h_new (1, 1, gru_hidden)
        """
        enc = self.encoder(obs).unsqueeze(1)        # (1, 1, enc_hidden)
        out, h_new = self.gru(enc, h.to(enc.dtype)) # (1, 1, gru_hidden)
        out = out.squeeze(1)                        # (1, gru_hidden)
        return self.actor(out), self.critic(out).squeeze(-1), h_new

    def forward_sequence(self, obs_seq: torch.Tensor, h0: torch.Tensor):
        """
        Full-sequence forward pass used during PPO update (truncated BPTT).

        obs_seq : (B, T, obs_dim)
        h0      : (1, B, gru_hidden)

        Returns  logits (B*T, n_actions),  values (B*T,)
        """
        B, T, _ = obs_seq.shape
        enc = self.encoder(obs_seq.view(B * T, -1)).view(B, T, -1)
        out, _ = self.gru(enc, h0.to(enc.dtype))    # (B, T, gru_hidden)
        out = out.contiguous().view(B * T, -1)
        return self.actor(out), self.critic(out).squeeze(-1)

    def get_action(self, obs: torch.Tensor, h: torch.Tensor, logit_bias: torch.Tensor | None = None):
        logits, value, h_new = self.forward_step(obs, h)
        if logit_bias is not None:
            logits += logit_bias
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, h_new


# ──────────────────────────────────────────────────────────────────────────────
# STANDARD ROLLOUT BUFFER  (find + push)
# ──────────────────────────────────────────────────────────────────────────────

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
            return
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
            return
        adv, ret = self.compute_gae(last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def _gpu(arr, dtype=torch.float32):
            t = torch.from_numpy(arr)
            if self._pin:
                t = t.pin_memory()
            return t.to(device, non_blocking=True).to(dtype)

        s_t   = _gpu(self.obs[:self.ptr])
        a_t   = _gpu(self.actions[:self.ptr], dtype=torch.int64)
        lp_t  = _gpu(self.logprobs[:self.ptr])
        adv_t = _gpu(adv)
        ret_t = _gpu(ret)

        idx = torch.randperm(self.ptr, device=device)
        for start in range(0, self.ptr, batch_size):
            b = idx[start : start + batch_size]
            yield s_t[b], a_t[b], lp_t[b], adv_t[b], ret_t[b]


# ──────────────────────────────────────────────────────────────────────────────
# GRU ROLLOUT BUFFER  (unwedge)
#
# Stores the GRU hidden state h at the START of each step in addition to the
# standard (obs, action, reward, done, logprob, value) tuple.
#
# During the PPO update we split the buffer into non-overlapping chunks of
# GRU_CHUNK_LEN steps and replay each chunk using its stored h0.  This is
# truncated BPTT: gradients flow within a chunk but not across chunks,
# which keeps training stable and memory bounded.
# ──────────────────────────────────────────────────────────────────────────────

class GRURolloutBuffer:
    def __init__(self, horizon: int, obs_dim: int,
                 gru_hidden: int, gamma: float, lam: float,
                 chunk_len: int = GRU_CHUNK_LEN):
        self.horizon    = horizon
        self.obs_dim    = obs_dim
        self.gru_hidden = gru_hidden
        self.gamma      = gamma
        self.lam        = lam
        self.chunk_len  = chunk_len
        self._pin       = device.type == "cuda"
        self.reset()

    def reset(self):
        self.obs      = np.zeros((self.horizon, self.obs_dim),    dtype=np.float32)
        self.actions  = np.zeros(self.horizon,                    dtype=np.int64)
        self.rewards  = np.zeros(self.horizon,                    dtype=np.float32)
        self.dones    = np.zeros(self.horizon,                    dtype=np.float32)
        self.logprobs = np.zeros(self.horizon,                    dtype=np.float32)
        self.values   = np.zeros(self.horizon,                    dtype=np.float32)
        # h stored as (gru_hidden,) per step — squeezed from (1,1,gru_hidden)
        self.hiddens  = np.zeros((self.horizon, self.gru_hidden), dtype=np.float32)
        self.ptr      = 0

    def add(self, obs, action, reward, done, logprob, value, h: torch.Tensor):
        """h is the hidden state BEFORE this step, shape (1, 1, gru_hidden)."""
        if self.ptr >= self.horizon:
            return
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr]   = value
        self.hiddens[self.ptr]  = h.squeeze().cpu().numpy()
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

    def get_chunks(self, last_value: float):
        """
        Yields (obs_chunk, actions, old_logprobs, advantages, returns, h0)
        in sequential chunks of self.chunk_len.

        obs_chunk : (1, T, obs_dim)   — batch=1 for forward_sequence
        h0        : (1, 1, gru_hidden) — initial hidden for the chunk
        others    : (T,)
        """
        if self.ptr < 2:
            return
        adv, ret = self.compute_gae(last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def _t(arr, dtype=torch.float32):
            t = torch.from_numpy(arr)
            if self._pin:
                t = t.pin_memory()
            return t.to(device, non_blocking=True).to(dtype)

        for start in range(0, self.ptr, self.chunk_len):
            end = min(start + self.chunk_len, self.ptr)
            sl  = slice(start, end)
            yield (
                _t(self.obs[sl]).unsqueeze(0),              # (1, T, obs_dim)
                _t(self.actions[sl], dtype=torch.int64),    # (T,)
                _t(self.logprobs[sl]),                      # (T,)
                _t(adv[sl]),                                # (T,)
                _t(ret[sl]),                                # (T,)
                _t(self.hiddens[start]).view(1, 1, -1),     # (1, 1, gru_h)
            )


# ──────────────────────────────────────────────────────────────────────────────
# PPO UPDATE — standard MLP agents
# ──────────────────────────────────────────────────────────────────────────────

def ppo_update(net, opt, scaler, buffer, last_value,
               epochs, batch_size, clip_eps, vf_coef, ent_coef,
               max_grad_norm, use_amp):
    total_pg = total_vf = total_ent = n = 0.0
    for _ in range(epochs):
        for sb, ab, old_lp, adv, ret in buffer.get_batches(last_value, batch_size):
            with autocast(device_type=device.type, enabled=use_amp):
                new_lp, entropy, values = net.evaluate(sb, ab)
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

    return (total_pg / n, total_vf / n, total_ent / n) if n > 0 else (0, 0, 0)


# ──────────────────────────────────────────────────────────────────────────────
# PPO UPDATE — GRU unwedge (truncated BPTT over stored chunks)
# ──────────────────────────────────────────────────────────────────────────────

def ppo_update_gru(net: GRUActorCritic, opt, scaler,
                   buffer: GRURolloutBuffer, last_value,
                   epochs, clip_eps, vf_coef, ent_coef,
                   max_grad_norm, use_amp):
    chunks = list(buffer.get_chunks(last_value))
    if not chunks:
        return 0.0, 0.0, 0.0

    total_pg = total_vf = total_ent = n = 0.0

    for _ in range(epochs):
        random.shuffle(chunks)
        for obs_c, act_c, old_lp_c, adv_c, ret_c, h0_c in chunks:
            with autocast(device_type=device.type, enabled=use_amp):
                # obs_c: (1, T, obs_dim),  h0_c: (1, 1, gru_hidden)
                logits, values = net.forward_sequence(obs_c, h0_c)
                # logits: (T, n_actions),  values: (T,)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(act_c)
                entropy = dist.entropy()

                ratio   = torch.exp(new_lp - old_lp_c)
                pg_loss = torch.max(
                    -adv_c * ratio,
                    -adv_c * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                ).mean()
                vf_loss  = 0.5 * (values - ret_c).pow(2).mean()
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
    data = {"episode": episode, "steps": steps}
    for name, ag in agents.items():
        data[f"{name}_net"]    = _raw(ag["net"]).state_dict()
        data[f"{name}_opt"]    = ag["opt"].state_dict()
        data[f"{name}_scaler"] = ag["scaler"].state_dict()
    torch.save(data, path)
    print(f"\n[checkpoint] saved -> {path} (ep {episode})")


def load_checkpoint(path, agents):
    ck = torch.load(path, map_location=device, weights_only=False)
    for name, ag in agents.items():
        _raw(ag["net"]).load_state_dict(ck[f"{name}_net"])
        ag["opt"].load_state_dict(ck[f"{name}_opt"])
        if f"{name}_scaler" in ck:
            ag["scaler"].load_state_dict(ck[f"{name}_scaler"])
    print(f"[resume] loaded checkpoint from {path}")
    return ck["steps"], ck["episode"]


def load_weights_dir(load_dir, agents):
    name_to_file = {
        "find": "weights_find.pth",
        "push": "weights_push.pth",
        "unwedge": "weights_unwedge.pth",
    }
    missing = []
    for name, fname in name_to_file.items():
        path = os.path.join(load_dir, fname)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError("Missing warm-start weights:\n" + "\n".join(missing))

    for name, fname in name_to_file.items():
        payload = torch.load(os.path.join(load_dir, fname), map_location=device, weights_only=False)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        _raw(agents[name]["net"]).load_state_dict(state_dict, strict=True)

    print(f"[warm-start] loaded weights from {load_dir}")


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

    # Misc
    ap.add_argument("--seed",       type=int,  default=0)
    ap.add_argument("--out_dir",    type=str,  default="ppo_gru_un_weights")
    ap.add_argument("--resume",     type=str,  default=None)
    ap.add_argument("--load_dir",   type=str,  default=None)
    ap.add_argument("--no_amp",     action="store_true")
    ap.add_argument("--no_compile", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    OBELIX  = import_obelix(args.obelix_py)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    # ── Build agents ──────────────────────────────────────────────────────────
    def make_mlp_agent(obs_dim, hidden, n_actions=len(ACTIONS)):
        net    = ActorCritic(obs_dim, n_actions=n_actions, hidden=hidden).to(device)
        if (not args.no_compile) and device.type == "cuda":
            try:
                net = torch.compile(net)
                print(f"[opt] torch.compile enabled for MLP dim={obs_dim}")
            except Exception as e:
                print(f"[opt] torch.compile unavailable: {e}")
        opt    = optim.Adam(_raw(net).parameters(), lr=args.lr, eps=1e-5)
        scaler = GradScaler("cuda", enabled=use_amp)
        buf    = RolloutBuffer(args.horizon, obs_dim, args.gamma, args.lam)
        sbuf   = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
        return {"net": net, "opt": opt, "scaler": scaler,
                "buf": buf, "sbuf": sbuf, "obs_dim": obs_dim,
                "last_value": 0.0, "is_gru": False}

    def make_gru_agent():
        net = GRUActorCritic(obs_dim=UNWEDGER_OBS_DIM,
                             n_actions=N_ACTIONS_W,
                             enc_hidden=64,
                             gru_hidden=GRU_HIDDEN).to(device)
        if (not args.no_compile) and device.type == "cuda":
            try:
                net = torch.compile(net)
                print("[opt] torch.compile enabled for GRU unwedge")
            except Exception as e:
                print(f"[opt] torch.compile unavailable: {e}")
        opt    = optim.Adam(_raw(net).parameters(), lr=args.lr, eps=1e-5)
        scaler = GradScaler("cuda", enabled=use_amp)
        buf    = GRURolloutBuffer(args.horizon, UNWEDGER_OBS_DIM,
                                  GRU_HIDDEN, args.gamma, args.lam)
        sbuf   = torch.zeros(1, UNWEDGER_OBS_DIM, dtype=torch.float32, device=device)
        h      = _raw(net).zero_hidden()   # live hidden state (1, 1, GRU_HIDDEN)
        return {"net": net, "opt": opt, "scaler": scaler,
                "buf": buf, "sbuf": sbuf, "obs_dim": UNWEDGER_OBS_DIM,
                "last_value": 0.0, "is_gru": True, "h": h}

    agents = {
        "find":     make_mlp_agent(FINDER_OBS_DIM,   hidden=256),
        "push":     make_mlp_agent(PUSHER_OBS_DIM,   hidden=128),
        "unwedge":  make_gru_agent(),
    }
    print(f"[opt] AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"[agents] find={FINDER_OBS_DIM}d MLP | "
          f"push={PUSHER_OBS_DIM}d MLP | "
          f"unwedge={UNWEDGER_OBS_DIM}d GRU(h={GRU_HIDDEN}, chunk={GRU_CHUNK_LEN})")

    steps    = 0
    start_ep = 0

    if args.resume:
        steps, start_ep = load_checkpoint(args.resume, agents)
    elif args.load_dir:
        load_weights_dir(args.load_dir, agents)

    # ── Obs → GPU helper ─────────────────────────────────────────────────────
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
            cur_size = 500
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
    ep_find = ep_push = ep_unwedge = 0

    env, raw, cur_arena = new_env(ep)
    last_dist  = euclidean_distance(env.bot_center_x, env.bot_center_y,
                                    env.box_center_x, env.box_center_y)
    last_angle = angle_to_box_deg(env.bot_center_x, env.bot_center_y,
                                  env.facing_angle,
                                  env.box_center_x, env.box_center_y)

    box_attached        = False
    push_grace          = 0
    PUSH_GRACE_STEPS    = 7
    unwedge_steps       = 0
    unwedge_active      = False
    unwedge_grace       = 0
    UNWEDGE_GRACE_STEPS = 10
    # After unwedge exits, suppress push activation for this many steps.
    # Prevents push from immediately acting on the wall-contact IR that
    # triggered the unwedge in the first place.
    POST_UNWEDGE_COOLDOWN = 10
    post_unwedge_cooldown = 0

    rewards_history = []
    start_time      = time.time()

    # ── Horizon loop ──────────────────────────────────────────────────────────
    try:
        while ep < args.episodes:

            for ag in agents.values():
                ag["buf"].reset()

            horizon_steps = 0

            while horizon_steps < args.horizon:

                # ── Priority network selection ────────────────────────────────
                ir_on    = bool(raw[16] == 1)
                stuck_on = bool(raw[17] == 1)

                if stuck_on:
                    unwedge_active = True
                    unwedge_grace  = UNWEDGE_GRACE_STEPS
                elif unwedge_grace > 0:
                    unwedge_grace -= 1
                    if unwedge_grace == 0:
                        unwedge_active = False
                        # Start cooldown — block push from the stale wall IR
                        post_unwedge_cooldown = POST_UNWEDGE_COOLDOWN

                # Tick cooldown down; push is blocked while it is active
                if post_unwedge_cooldown > 0:
                    post_unwedge_cooldown -= 1

                if ir_on and post_unwedge_cooldown == 0:
                    push_grace = PUSH_GRACE_STEPS
                elif push_grace > 0:
                    push_grace -= 1

                if unwedge_active:
                    mode = "unwedge"
                elif push_grace > 0:
                    mode = "push"
                else:
                    mode = "find"

                ag = agents[mode]

                # ── Observation ───────────────────────────────────────────────
                if mode == "find":
                    obs = get_find_obs(raw)
                elif mode == "push":
                    obs = get_push_obs(raw)
                else:
                    obs = get_unwedge_obs(raw)

                # ── Inference ────────────────────────────────────────────────
                with torch.no_grad():
                    st = to_device(ag, obs)
                    with autocast(device_type=device.type, enabled=use_amp):

                        if mode == "find":
                            bias = torch.tensor([-0.25, 1.0, -0.25], device=device)
                            action, log_prob, _, value = ag["net"].get_action(st)
                            a_idx = int(action.item())

                        elif mode == "push":
                            bias = torch.tensor([-1.0, 1.0, -1.0], device=device)
                            action, log_prob, _, value = ag["net"].get_action(st)
                            a_idx = int(action.item())

                        else:  # unwedge — GRU path
                            h_before = ag["h"] 

                            bias = torch.tensor([-0.25, 1.0, -0.25], device=device)         # hidden state BEFORE this step
                            action, log_prob, _, value, h_new = \
                                ag["net"].get_action(st, ag["h"])
                            a_idx   = int(action.item())
                            #if sum(raw)==0 :
                               # a_idx=1
                                #print("empty obs, forcing forward")
                            ag["h"] = h_new             # advance live hidden state

                # ── Step environment ──────────────────────────────────────────
                if mode == "unwedge":
                    raw2, env_r, done = env.step(ACTIONS_W[a_idx], render=False)
                else:
                    raw2, env_r, done = env.step(ACTIONS[a_idx], render=False)
                

                # ── IR probe ──────────────────────────────────────────────────
                if ir_on and not box_attached and not unwedge_active and not done:
                    confirmed, probe_raw, probe_r, probe_done = \
                        probe_box_attached(env, raw2)
                    box_attached = confirmed
                    if confirmed:
                        push_grace = PUSH_GRACE_STEPS
                        print("attached")
                    ep_reward     += probe_r
                    ep_steps      += PROBE_STEPS
                    steps         += PROBE_STEPS
                    horizon_steps += PROBE_STEPS
                    raw2 = probe_raw
                    if probe_done:
                        done = True

                # ── Reward shaping ────────────────────────────────────────────
                r = env_r

                #if mode == "find":
                    #if (raw2[4] == 1 or raw2[6] == 1 or
                            #raw2[8] == 1 or raw2[10] == 1) and raw2[17] == 0:
                        #r += 20

                if mode == "push":
                    if a_idx == 1 and raw[16] == 1 and raw2[17] == 0 and box_attached:
                        r += 3.0
                    else:
                        r -= 3.0

                else:  # unwedge
                    now_stuck = bool(raw2[17] == 1)
                    unwedge_steps += 1
                    if now_stuck:
                        r -= min(1.0 * unwedge_steps, 5.0)
                    elif not now_stuck and raw2[16] == 0:
                        escape_bonus = max(20.0 - unwedge_steps * 1.5, 5.0)
                        r += escape_bonus
                        unwedge_steps = 0
                    #if sum(raw2) > 0:
                        #r -= sum(raw2) * 0.3

                if mode == "push":
                    r = np.clip(r, -10.0, 10.0)
                elif mode == "unwedge":
                    r = np.clip(r, -20.0, 20.0)

                # ── Store in buffer ───────────────────────────────────────────
                if mode == "unwedge":
                    ag["buf"].add(obs, a_idx, r, float(done),
                                  float(log_prob.item()), float(value.item()),
                                  h=h_before)
                else:
                    ag["buf"].add(obs, a_idx, r, float(done),
                                  float(log_prob.item()), float(value.item()))

                last_angle = angle_to_box_deg(env.bot_center_x, env.bot_center_y,
                                              env.facing_angle,
                                              env.box_center_x, env.box_center_y)
                last_dist  = euclidean_distance(env.bot_center_x, env.bot_center_y,
                                                env.box_center_x, env.box_center_y)
                raw        = raw2
                ep_reward += env_r
                ep_steps  += 1
                steps     += 1
                horizon_steps += 1

                if mode == "find":      ep_find += 1
                elif mode == "push":    ep_push += 1
                else:                   ep_unwedge += 1

                if done:
                    rewards_history.append(ep_reward)
                    avg100  = np.mean(rewards_history[-100:])
                    elapsed = time.time() - start_time
                    speed   = steps / elapsed if elapsed > 0 else 0
                    print(
                        f"[train] ep={ep+1}/{args.episodes} | "
                        f"return={ep_reward:.1f} | avg100={avg100:.1f} | "
                        f"arena={cur_arena} | "
                        f"find/push/unwedge={ep_find}/{ep_push}/{ep_unwedge} | "
                        f"box_attached={'yes' if box_attached else 'no'} | "
                        f"speed={speed:.0f} sps"
                    )
                    ep             += 1
                    ep_reward       = 0.0
                    ep_steps        = 0
                    ep_find = ep_push = ep_unwedge = 0
                    box_attached          = False
                    push_grace            = 0
                    unwedge_active        = False
                    unwedge_grace         = 0
                    unwedge_steps         = 0
                    post_unwedge_cooldown = 0
                    # Zero GRU hidden state at episode boundary
                    agents["unwedge"]["h"] = \
                        _raw(agents["unwedge"]["net"]).zero_hidden()

                    if ep >= args.episodes:
                        break

                    env, raw, cur_arena = new_env(ep)
                    last_angle = angle_to_box_deg(env.bot_center_x, env.bot_center_y,
                                                  env.facing_angle,
                                                  env.box_center_x, env.box_center_y)
                    last_dist  = euclidean_distance(env.bot_center_x, env.bot_center_y,
                                                    env.box_center_x, env.box_center_y)

            # ── PPO update ────────────────────────────────────────────────────
            with torch.no_grad():
                for name, agent in agents.items():
                    if agent["buf"].ptr < 2:
                        continue
                    obs_lv = (get_find_obs(raw) if name == "find" else
                              get_push_obs(raw) if name == "push" else
                              get_unwedge_obs(raw))
                    st_lv = to_device(agent, obs_lv)
                    if agent["is_gru"]:
                        _, lv_tensor, _ = agent["net"].forward_step(st_lv, agent["h"])
                    else:
                        _, lv_tensor = agent["net"](st_lv)
                    agent["last_value"] = float(lv_tensor.item()) * (1 - float(done))

            for name, agent in agents.items():
                buf = agent["buf"]
                if buf.ptr < 2:
                    continue
                if agent["is_gru"]:
                    pg, vf, ent = ppo_update_gru(
                        agent["net"], agent["opt"], agent["scaler"],
                        buf, agent["last_value"],
                        args.epochs, args.clip_eps,
                        args.vf_coef, args.ent_coef, args.max_grad_norm, use_amp
                    )
                else:
                    if buf.ptr < args.batch:
                        continue
                    pg, vf, ent = ppo_update(
                        agent["net"], agent["opt"], agent["scaler"],
                        buf, agent["last_value"],
                        args.epochs, args.batch, args.clip_eps,
                        args.vf_coef, args.ent_coef, args.max_grad_norm, use_amp
                    )
                print(f"  [update:{name}] steps={buf.ptr} | "
                      f"pg={pg:.4f} vf={vf:.4f} ent={ent:.4f}")

            if ep > 0 and ep % 1 == 0:
                save_checkpoint(os.path.join(args.out_dir, "checkpoint_latest.pth"), agents, ep, steps)

    except KeyboardInterrupt:
        print("\nInterrupted — saving...")
        save_checkpoint(os.path.join(args.out_dir, "checkpoint_interrupt.pth"), agents, ep, steps)

    for name, agent in agents.items():
        path = os.path.join(args.out_dir, f"weights_{name}.pth")
        torch.save(_raw(agent["net"]).state_dict(), path)
        print(f"[save] {name} -> {path}")


if __name__ == "__main__":
    main()
