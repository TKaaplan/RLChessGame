"""
RL training script — MaskablePPO + Stockfish opponent.

Usage
-----
1. Set STOCKFISH_PATH below (absolute path to stockfish.exe).
2. Run:  python train.py
3. Monitor:  tensorboard --logdir tb_logs/
4. Curriculum (after first run):  python train.py curriculum checkpoints/chess_ppo_*.zip

Mode
----
VS_STOCKFISH = True   (default): Agent ('b') vs Stockfish ('w').
  Start at STOCKFISH_OPPONENT_DEPTH=1, increase via curriculum when win_rate > 40%.

VS_STOCKFISH = False  (legacy):  Self-play with Stockfish reward shaping.

Outputs
-------
  checkpoints/chess_ppo_<N>_steps.zip   — saved every SAVE_EVERY steps
  chess_ppo_final.zip                   — final model after full training
  tb_logs/                              — TensorBoard logs
"""

import os
import sys
from datetime import datetime

# ── CONFIGURE THESE ────────────────────────────────────────────────────────────
STOCKFISH_PATH = r"C:\Users\tolga\Desktop\stockfish\stockfish-windows-x86-64-avx2.exe"

TOTAL_TIMESTEPS        = 2_000_000
DEPTH                  = 5           # eval depth (self-play mode only)
VS_STOCKFISH           = True        # True → agent vs Stockfish opponent
STOCKFISH_OPPONENT_DEPTH = 1         # depth=1 ≈ 1200 Elo (beatable by a learning agent)
N_ENVS                 = 6           # SubprocVecEnv workers (= physical cores)
SAVE_EVERY             = 100_000

_HERE          = os.path.dirname(os.path.abspath(__file__))
LOG_DIR        = os.path.join(_HERE, "tb_logs")
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")

NET_ARCH = [256, 256]
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── BUG FIX: PyTorch float32 precision on 4096-action Simplex check ───────────
import torch as _th
from sb3_contrib.common.maskable.distributions import MaskableCategorical as _MC
from torch.distributions.utils import logits_to_probs as _l2p

def _safe_apply_masking(self, masks):
    if masks is not None:
        device = self.logits.device
        self.masks = _th.as_tensor(masks, dtype=_th.bool, device=device).reshape(self.logits.shape)
        HUGE_NEG   = _th.tensor(-1e8, dtype=self.logits.dtype, device=device)
        logits     = _th.where(self.masks, self._original_logits, HUGE_NEG)
    else:
        self.masks = None
        logits     = self._original_logits
    _th.distributions.Categorical.__init__(self, logits=logits, validate_args=False)
    self.probs = _l2p(self.logits)

_MC.apply_masking = _safe_apply_masking
# ──────────────────────────────────────────────────────────────────────────────

import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rl_env import ChessRLEnv


class ChessCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for (14, 8, 8) chess observations.

    3 conv layers (no downsampling) preserve full board resolution.
    Spatial relationships (attacks, defences, pawn chains) remain intact.

    Input:  (14, 8, 8) — 12 piece planes + turn + move counter
    Output: features_dim-dimensional vector fed into the MLP head.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, features_dim), nn.ReLU(),
        )

    def forward(self, obs: _th.Tensor) -> _th.Tensor:
        return self.net(obs)


def _make_env():
    """Module-level env factory — picklable for SubprocVecEnv."""
    return ChessRLEnv(
        stockfish_path=STOCKFISH_PATH,
        depth=DEPTH,
        vs_stockfish=VS_STOCKFISH,
        stockfish_opponent_depth=STOCKFISH_OPPONENT_DEPTH,
    )


class EpisodeStatsCallback(BaseCallback):
    """
    Logs per-episode statistics to TensorBoard:
      chess/win_rate, chess/loss_rate, chess/draw_rate, chess/stalemate_rate,
      chess/mean_ep_reward, chess/mean_ep_length, chess/stockfish_depth
    """
    def __init__(self, stockfish_depth: int = 1, win_rate_threshold: float = 0.40, verbose=0):
        super().__init__(verbose)
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int]   = []
        self._outcomes:   list[str]   = []   # 'win', 'loss', 'stalemate', 'draw'
        self._window = 100
        self._sf_depth = stockfish_depth
        self._win_threshold = win_rate_threshold
        self._depth_promoted = False  # flag: already printed the depth-up message

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])

                r = ep["r"]
                if   r >= 0.9:              self._outcomes.append("win")
                elif r <= -0.9:             self._outcomes.append("loss")
                elif abs(r + 0.05) < 0.01: self._outcomes.append("stalemate")
                else:                       self._outcomes.append("draw")

                if len(self._outcomes) >= self._window:
                    recent     = self._outcomes[-self._window:]
                    win_rate   = recent.count("win")       / self._window
                    loss_rate  = recent.count("loss")      / self._window
                    draw_rate  = recent.count("draw")      / self._window
                    stale_rate = recent.count("stalemate") / self._window

                    self.logger.record("chess/win_rate",      win_rate)
                    self.logger.record("chess/loss_rate",     loss_rate)
                    self.logger.record("chess/draw_rate",     draw_rate)
                    self.logger.record("chess/stalemate_rate",stale_rate)
                    self.logger.record("chess/mean_ep_reward",
                                       sum(self._ep_rewards[-self._window:]) / self._window)
                    self.logger.record("chess/mean_ep_length",
                                       sum(self._ep_lengths[-self._window:]) / self._window)
                    self.logger.record("chess/stockfish_depth", self._sf_depth)

                    # Notify when win_rate crosses threshold (don't auto-change depth
                    # mid-training — use curriculum_vs_stockfish() to progress instead)
                    if (not self._depth_promoted
                            and win_rate >= self._win_threshold
                            and VS_STOCKFISH):
                        self._depth_promoted = True
                        print(
                            f"\n{'='*60}\n"
                            f"  WIN RATE {win_rate:.0%} >= {self._win_threshold:.0%} at depth {self._sf_depth}!\n"
                            f"  Run curriculum to advance to the next depth:\n"
                            f"    python train.py curriculum <latest_checkpoint.zip>\n"
                            f"{'='*60}\n"
                        )
        return True


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    mode = f"vs_stockfish_depth{STOCKFISH_OPPONENT_DEPTH}" if VS_STOCKFISH else f"self_play_depth{DEPTH}"
    print(f"Stockfish path : {STOCKFISH_PATH}")
    print(f"Mode           : {mode}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print()

    print(f"Initialising {N_ENVS} parallel environments …")
    env = make_vec_env(_make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    model = MaskablePPO(
        policy          = "MlpPolicy",
        env             = env,
        verbose         = 1,
        tensorboard_log = LOG_DIR,

        learning_rate       = 1e-4,
        n_steps             = 1024,
        batch_size          = 512,
        n_epochs            = 10,
        gamma               = 0.995,
        gae_lambda          = 0.98,
        clip_range          = 0.1,
        ent_coef            = 0.015,
        normalize_advantage = True,
        max_grad_norm       = 0.4,

        policy_kwargs = {
            "features_extractor_class":  ChessCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": NET_ARCH,
        },
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(SAVE_EVERY // N_ENVS, 1),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "chess_ppo",
        save_replay_buffer = False,
        verbose     = 1,
    )
    stats_cb = EpisodeStatsCallback(
        stockfish_depth=STOCKFISH_OPPONENT_DEPTH if VS_STOCKFISH else DEPTH,
        verbose=0,
    )

    run_name = f"ppo_{mode}_{datetime.now().strftime('%m%d_%H%M')}"
    print(f"Starting training …  run={run_name}")
    print(f"(TensorBoard → tensorboard --logdir {LOG_DIR})\n")
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = [checkpoint_cb, stats_cb],
        progress_bar        = True,
        reset_num_timesteps = True,
        tb_log_name         = run_name,
    )

    model.save("chess_ppo_final")
    print("\nTraining complete!  Model saved → chess_ppo_final.zip")
    env.close()


# ── resume training from a checkpoint ─────────────────────────────────────────

def resume(
    checkpoint_path: str,
    extra_timesteps: int = 500_000,
    depth:           int = DEPTH,
    vs_stockfish:    bool = VS_STOCKFISH,
    sf_opponent_depth: int = STOCKFISH_OPPONENT_DEPTH,
    lr:              float = 1e-4,
    ent_coef:        float = 0.04,
    clip_range:      float = 0.2,
    batch_size:      int   = 512,
    n_epochs:        int   = 10,
    save_prefix:     str   = "chess_ppo_resumed",
):
    """
    Resume training from a checkpoint with overridden hyperparameters.

    Example:
        from train import resume
        resume("checkpoints/chess_ppo_500000_steps.zip",
               sf_opponent_depth=2, extra_timesteps=1_000_000)
    """
    import functools
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Resuming from : {checkpoint_path}")
    print(f"  vs_stockfish         : {vs_stockfish}")
    print(f"  sf_opponent_depth    : {sf_opponent_depth}")
    print(f"  lr                   : {lr}")
    print(f"  ent_coef             : {ent_coef}")
    print(f"  clip_range           : {clip_range}")
    print(f"  extra_steps          : {extra_timesteps:,}\n")

    env_fn = functools.partial(
        ChessRLEnv,
        stockfish_path=STOCKFISH_PATH,
        depth=depth,
        vs_stockfish=vs_stockfish,
        stockfish_opponent_depth=sf_opponent_depth,
    )
    env   = make_vec_env(env_fn, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    model = MaskablePPO.load(checkpoint_path, env=env)
    model.tensorboard_log = LOG_DIR   # checkpoint'teki eski relative path'i ezmek icin

    model.learning_rate = lr
    model.lr_schedule   = get_schedule_fn(lr)
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = lr

    model.ent_coef   = ent_coef
    model.clip_range = get_schedule_fn(clip_range)
    model.batch_size = batch_size
    model.n_epochs   = n_epochs

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(SAVE_EVERY // N_ENVS, 1),
        save_path   = CHECKPOINT_DIR,
        name_prefix = save_prefix,
        save_replay_buffer = False,
        verbose     = 1,
    )
    stats_cb = EpisodeStatsCallback(
        stockfish_depth=sf_opponent_depth if vs_stockfish else depth
    )

    run_name = f"resume_sfd{sf_opponent_depth}_{datetime.now().strftime('%m%d_%H%M')}"
    model.learn(
        total_timesteps     = extra_timesteps,
        callback            = [checkpoint_cb, stats_cb],
        progress_bar        = True,
        reset_num_timesteps = False,
        tb_log_name         = run_name,
    )
    out = f"{save_prefix}_sfd{sf_opponent_depth}.zip"
    model.save(out)
    print(f"\nSaved → {out}")
    env.close()
    return out


def curriculum_vs_stockfish(start_checkpoint: str):
    """
    Progressive curriculum: train agent vs Stockfish at increasing depths.

    Depth 1 → 2 → 3 → 5, with decreasing lr and entropy.

    Usage:
        python train.py curriculum checkpoints/chess_ppo_2000000_steps.zip
    """
    stages = [
        # (sf_opponent_depth, extra_steps, lr,    ent_coef, clip_range)
        (1, 1_000_000, 1e-4,  0.04,  0.20),   # Beat depth-1 reliably
        (2, 1_000_000, 7e-5,  0.025, 0.18),   # Step up
        (3, 1_000_000, 5e-5,  0.015, 0.15),   # Approach depth-3
        (5,   800_000, 2e-5,  0.010, 0.12),   # Challenge depth-5
    ]

    ckpt = start_checkpoint
    for sf_d, steps, lr, ent, clip in stages:
        print(f"\n{'='*60}")
        print(f"  CURRICULUM  sf_depth={sf_d}  steps={steps:,}")
        print(f"{'='*60}")
        prefix = f"curriculum_sfd{sf_d}"
        ckpt = resume(
            checkpoint_path   = ckpt,
            extra_timesteps   = steps,
            vs_stockfish      = True,
            sf_opponent_depth = sf_d,
            lr                = lr,
            ent_coef          = ent,
            clip_range        = clip,
            save_prefix       = prefix,
        )

    print("\nCurriculum complete!  Final model:", ckpt)


def _latest_checkpoint() -> str:
    """checkpoints/ klasöründeki en yeni .zip dosyasını döndürür."""
    import glob as _glob
    files = _glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    if not files:
        raise FileNotFoundError(f"Checkpoint bulunamadi: {CHECKPOINT_DIR}")
    return max(files, key=os.path.getmtime)


def improve(start_checkpoint: str = None):
    """
    En son checkpoint'ten depth-2 → depth-3 curriculum.

    depth-1 zaten %85 win rate ile masterlanmis — tekrar egitmek zaman kaybi.
    depth-2: 1.5M adim — kopru seviyesi
    depth-3: 2.5M adim — hedef seviye (oncekinden daha uzun)

    Toplam: ~4M ek adim.

    Kullanim:
        python train.py improve
        python train.py improve checkpoints/baska_model.zip
    """
    stages = [
        # (sf_opponent_depth, extra_steps, lr,    ent_coef, clip_range, prefix)
        (2, 1_500_000, 7e-5,  0.025, 0.18, "improved_sfd2"),
        (3, 2_500_000, 5e-5,  0.015, 0.15, "improved_sfd3"),
    ]

    if start_checkpoint is None:
        start_checkpoint = _latest_checkpoint()

    print(f"\nBaslangic checkpoint: {start_checkpoint}")
    print(f"Toplam ek adim: {sum(s[1] for s in stages):,}")
    print(f"Hedef cikti : improved_sfd3_sfd3.zip  (play_vs_bot icin hazir)\n")

    ckpt = start_checkpoint
    for sf_d, steps, lr, ent, clip, prefix in stages:
        print(f"\n{'='*60}")
        print(f"  IMPROVE  sf_depth={sf_d}  steps={steps:,}  lr={lr}  ent={ent}")
        print(f"{'='*60}")
        ckpt = resume(
            checkpoint_path   = ckpt,
            extra_timesteps   = steps,
            vs_stockfish      = True,
            sf_opponent_depth = sf_d,
            lr                = lr,
            ent_coef          = ent,
            clip_range        = clip,
            save_prefix       = prefix,
        )

    print(f"\nImprove tamamlandi! Final model: {ckpt}")
    print("play_vs_bot.py ile oynamak icin:")
    print(f"  python play_vs_bot.py {ckpt}")
    return ckpt


def stable_finetune(start_checkpoint: str = None):
    """
    Depth-3 sonrasi policy collapse'i onlemek icin dusuk LR ile fine-tune.

    improved_sfd3 checkpoint'inden baslayarak:
      depth-2: 2M adim, lr=2e-5, ent=0.008, clip=0.10  — stabilize + exploit
      depth-3: 3M adim, lr=1e-5, ent=0.005, clip=0.08  — daha guclu rakibe karsı
      depth-5: 1M adim, lr=5e-6, ent=0.003, clip=0.06  — son polish

    Toplam: ~6M ek adim.

    Kullanim:
        python train.py stable
        python train.py stable checkpoints/improved_sfd3_9204996_steps.zip
    """
    stages = [
        # (sf_opponent_depth, extra_steps, lr,   ent_coef, clip_range, prefix)
        (2, 2_000_000, 2e-5, 0.008, 0.10, "stable_sfd2"),
        (3, 3_000_000, 1e-5, 0.005, 0.08, "stable_sfd3"),
        (5, 1_000_000, 5e-6, 0.003, 0.06, "stable_sfd5"),
    ]

    if start_checkpoint is None:
        # Oncelikle improved_sfd3 ara; yoksa en yeni checkpoint'i al
        import glob as _glob
        sfd3 = sorted(_glob.glob(os.path.join(CHECKPOINT_DIR, "improved_sfd3*.zip")))
        start_checkpoint = sfd3[-1] if sfd3 else _latest_checkpoint()

    print(f"\nBaslangic checkpoint : {start_checkpoint}")
    print(f"Toplam ek adim       : {sum(s[1] for s in stages):,}")
    print(f"Strateji             : dusuk LR + kucuk clip → policy collapse engellenir\n")

    ckpt = start_checkpoint
    for sf_d, steps, lr, ent, clip, prefix in stages:
        print(f"\n{'='*60}")
        print(f"  STABLE  sf_depth={sf_d}  steps={steps:,}  lr={lr}  ent={ent}  clip={clip}")
        print(f"{'='*60}")
        ckpt = resume(
            checkpoint_path   = ckpt,
            extra_timesteps   = steps,
            vs_stockfish      = True,
            sf_opponent_depth = sf_d,
            lr                = lr,
            ent_coef          = ent,
            clip_range        = clip,
            save_prefix       = prefix,
        )

    print(f"\nStable fine-tune tamamlandi! Final model: {ckpt}")
    print("play_vs_bot.py ile oynamak icin:")
    print(f"  python play_vs_bot.py {ckpt}")
    return ckpt


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) >= 1 and args[0] == "stable":
        # python train.py stable
        # python train.py stable checkpoints/improved_sfd3_9204996_steps.zip
        ckpt = args[1] if len(args) >= 2 else None
        stable_finetune(ckpt)
    elif len(args) >= 1 and args[0] == "improve":
        # python train.py improve
        # python train.py improve checkpoints/baska_model.zip
        ckpt = args[1] if len(args) >= 2 else None   # None → _latest_checkpoint() otomatik
        improve(ckpt)
    elif len(args) == 2 and args[0] == "curriculum":
        # python train.py curriculum checkpoints/chess_ppo_2000000_steps.zip
        curriculum_vs_stockfish(args[1])
    elif len(args) == 1 and args[0].endswith(".zip"):
        # python train.py checkpoints/chess_ppo_500000_steps.zip
        import re
        m = re.search(r'_(\d+)_steps', args[0])
        done = int(m.group(1)) if m else 0
        remaining = max(TOTAL_TIMESTEPS - done, 0)
        print(f"Checkpoint: {done:,} steps done → remaining: {remaining:,}")
        resume(args[0], extra_timesteps=remaining)
    else:
        main()
