"""
Chess RL environment (gymnasium-compatible).

Observation  : (14, 8, 8) float32  [channels-first]
  channels 0-5   → white pieces  (BP BN BB BR BQ BK)
  channels 6-11  → black pieces  (p  n  b  r  q  k )
  channel  12    → 1.0 when it is WHITE's turn, 0.0 when BLACK's
  channel  13    → normalised move counter  (move / MAX_MOVES)

Action space : Discrete(4096)  →  from_sq * 64 + to_sq
  where  sq = row * 8 + col

Reward design
─────────────
vs_stockfish=False  (self-play, original mode):
  checkmate delivered → +1.0
  stalemate           → −0.05
  move limit reached  → −0.3
  all other moves     → Δ tanh(cp/400) − 0.001

vs_stockfish=True  (agent='b' vs Stockfish='w'):
  agent delivers checkmate   → +1.0
  Stockfish delivers checkmate → −1.0
  agent causes stalemate     → −0.05  (draw, slight penalty)
  Stockfish causes stalemate → −0.3   (agent got stalemated)
  move limit reached         → −0.5
  all other moves            → Δ tanh(cp/400) − 0.001

Requires:
  pip install gymnasium stable-baselines3 sb3-contrib python-chess tensorboard
  + Stockfish binary (https://stockfishchess.org/download/)
"""

import math
import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_env import ChessEnv
from stockfish_eval import StockfishEvaluator

# ── constants ─────────────────────────────────────────────────────────────────

# Maps our piece strings → observation channel index
PIECE_TO_CH: dict[str, int] = {
    'BP': 0, 'BN': 1, 'BB': 2, 'BR': 3, 'BQ': 4, 'BK': 5,
    'p':  6, 'n':  7, 'b':  8, 'r':  9, 'q': 10, 'k': 11,
}

MAX_MOVES = 200   # hard draw limit per episode (100 moves each side)


def _cp_to_value(cp: int) -> float:
    """Normalise centipawns to (−1, +1) using tanh.  400 cp ≈ winning."""
    return math.tanh(cp / 400.0)


# ── environment ───────────────────────────────────────────────────────────────

class ChessRLEnv(gym.Env):
    """
    Chess RL environment with two operating modes:

    vs_stockfish=False (default):
        Self-play — the same agent plays both sides.
        Stockfish provides centipawn reward shaping.

    vs_stockfish=True:
        Agent always plays as 'b' (light pieces, bottom, moves first).
        Stockfish plays as 'w' at stockfish_opponent_depth.
        Reward is terminal-only: +1 win, -1 loss, -0.05 stalemate, -0.3 draw.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stockfish_path: str,
        depth: int = 8,
        vs_stockfish: bool = False,
        stockfish_opponent_depth: int = 1,
    ):
        """
        stockfish_path           : absolute path to the Stockfish executable.
        depth                    : Stockfish search depth for reward shaping (self-play mode).
        vs_stockfish             : if True, Stockfish plays the opponent side ('w').
        stockfish_opponent_depth : Stockfish depth when acting as opponent.
        """
        super().__init__()
        self._chess          = ChessEnv()
        self._vs_stockfish   = vs_stockfish

        if vs_stockfish:
            # Eval shaping (depth) + opponent (stockfish_opponent_depth) — two separate instances
            self._sf          = StockfishEvaluator(stockfish_path, depth=depth)
            self._sf_opponent = StockfishEvaluator(stockfish_path, depth=stockfish_opponent_depth)
        else:
            # Self-play: Stockfish for centipawn reward shaping only
            self._sf          = StockfishEvaluator(stockfish_path, depth=depth)
            self._sf_opponent = None

        self.action_space      = spaces.Discrete(4096)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14, 8, 8), dtype=np.float32
        )  # channels-first: (C, H, W) = (14 piece-planes, 8 rows, 8 cols)

        # internal state updated on every step/reset
        self._legal_moves: list[tuple] = []
        self._eval_before: int = 0
        self._move_count:  int = 0

    # ── encoding ──────────────────────────────────────────────────────────────

    def _encode_state(self) -> np.ndarray:
        board = self._chess.board
        obs   = np.zeros((14, 8, 8), dtype=np.float32)  # (channel, row, col)

        for r in range(8):
            for c in range(8):
                ch = PIECE_TO_CH.get(board[r][c])
                if ch is not None:
                    obs[ch, r, c] = 1.0

        if self._chess.current_player == 'w':
            obs[12, :, :] = 1.0

        obs[13, :, :] = self._move_count / MAX_MOVES
        return obs

    @staticmethod
    def encode_action(sr: int, sc: int, er: int, ec: int) -> int:
        return (sr * 8 + sc) * 64 + (er * 8 + ec)

    @staticmethod
    def decode_action(action: int) -> tuple[int, int, int, int]:
        from_sq = action // 64
        to_sq   = action  % 64
        return from_sq // 8, from_sq % 8, to_sq // 8, to_sq % 8

    # ── action mask (called by MaskablePPO) ───────────────────────────────────

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(4096, dtype=bool)
        for (sr, sc, er, ec) in self._legal_moves:
            mask[self.encode_action(sr, sc, er, ec)] = True
        # Safety: all-False mask → MaskablePPO Simplex error.
        # Shouldn't happen normally; open dummy action 0 so episode can terminate.
        if not mask.any():
            mask[0] = True
        return mask

    # ── Stockfish helpers ─────────────────────────────────────────────────────

    def _get_eval(self) -> int:
        return self._sf.evaluate(
            self._chess.board,
            self._chess.current_player,
            self._chess.en_passant_possible,
            self._chess.king_moved,
            self._chess.rook_moved,
        )

    def _stockfish_make_move(self):
        """Ask Stockfish to play the current position and apply the move. Returns new status."""
        result = self._sf_opponent.get_best_move(
            self._chess.board,
            self._chess.current_player,
            self._chess.en_passant_possible,
            self._chess.king_moved,
            self._chess.rook_moved,
        )
        if result is None:
            # No move returned — shouldn't happen; treat as stalemate
            return 'stalemate', []

        sr, sc, er, ec, promotion = result
        self._chess.make_move(sr, sc, er, ec, promotion=promotion)
        self._move_count += 1
        return self._chess.get_status_and_moves()

    # ── reward ────────────────────────────────────────────────────────────────

    def _compute_reward_self_play(
        self,
        player_who_moved: str,
        status: str,
        eval_after: int,
    ) -> float:
        if status == 'checkmate':
            return 1.0          # player_who_moved checkmated the opponent

        if status == 'stalemate':
            return -0.05        # slight penalty — stalemate wastes a win chance

        if self._move_count >= MAX_MOVES:
            return -0.3

        # Stockfish eval delta (core shaping signal)
        sign  = 1 if player_who_moved == 'w' else -1
        delta = _cp_to_value(sign * eval_after) - _cp_to_value(sign * self._eval_before)
        return delta - 0.001

    def _compute_reward_vs_sf(self, mover: str, status: str, eval_after: int = 0) -> float:
        """Reward always from agent ('b') perspective, with centipawn shaping."""
        if status == 'checkmate':
            return 1.0 if mover == 'b' else -1.0
        if status == 'stalemate':
            # Agent caused stalemate (opponent has no moves) → slight penalty (draw, not a win)
            # Stockfish caused stalemate (agent has no moves) → larger penalty
            return -0.05 if mover == 'b' else -0.3
        if self._move_count >= MAX_MOVES:
            return -0.5
        # Dense shaping: delta from 'b' perspective (negative cp = good for black)
        delta = _cp_to_value(-eval_after) - _cp_to_value(-self._eval_before)
        return delta - 0.001

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._chess.reset()
        self._move_count  = 0
        self._legal_moves = self._chess.get_legal_moves()
        self._eval_before = self._get_eval() 
        return self._encode_state(), {"action_mask": self.action_masks()}

    def step(self, action: int):
        if self._vs_stockfish:
            return self._step_vs_stockfish(action)
        return self._step_self_play(action)

    def _step_vs_stockfish(self, action: int):
        """Agent ('b') makes a move, then Stockfish ('w') responds."""
        sr, sc, er, ec = self.decode_action(action)

        if (sr, sc, er, ec) not in self._legal_moves:
            obs = self._encode_state()
            return obs, 0.0, True, False, {"action_mask": self.action_masks()}

        promotion = None
        if self._chess.needs_promotion(sr, sc, er, ec):
            promotion = 'q' 

        self._chess.make_move(sr, sc, er, ec, promotion=promotion)
        self._move_count += 1

        status, next_moves = self._chess.get_status_and_moves()

        # Eval after agent's move (terminal positions get fixed values)
        if status == 'checkmate':
            eval_after = -30_000   # agent delivered checkmate → very good for 'b'
        elif status != 'playing':
            eval_after = 0
        else:
            eval_after = self._get_eval()

        # If agent ended the game (checkmate / stalemate) or move limit
        if status != 'playing' or self._move_count >= MAX_MOVES:
            reward = self._compute_reward_vs_sf('b', status, eval_after)
            done   = True
            self._legal_moves = next_moves
            return self._encode_state(), reward, done, False, {"action_mask": self.action_masks()}

        # Update eval baseline before Stockfish moves
        self._eval_before = eval_after

        # Stockfish responds
        status, next_moves = self._stockfish_make_move()

        # Eval after Stockfish's move
        if status == 'checkmate':
            eval_after_sf = 30_000   # Stockfish checkmated agent → very bad for 'b'
        elif status != 'playing':
            eval_after_sf = 0
        else:
            eval_after_sf = self._get_eval()

        reward = self._compute_reward_vs_sf('w', status, eval_after_sf)
        done   = (status != 'playing') or (self._move_count >= MAX_MOVES)

        if not done:
            self._eval_before = eval_after_sf

        self._legal_moves = next_moves
        return self._encode_state(), reward, done, False, {"action_mask": self.action_masks()}

    def _step_self_play(self, action: int):
        """Original self-play mode with Stockfish centipawn reward shaping."""
        sr, sc, er, ec = self.decode_action(action)
        player = self._chess.current_player

        # Invalid action → terminate episode
        if (sr, sc, er, ec) not in self._legal_moves:
            obs = self._encode_state()
            return obs, 0.0, True, False, {"action_mask": self.action_masks()}

        # RL agent always promotes to queen
        promotion = None
        if self._chess.needs_promotion(sr, sc, er, ec):
            promotion = 'BQ' if player == 'w' else 'q'

        self._chess.make_move(sr, sc, er, ec, promotion=promotion)
        self._move_count += 1

        status, next_moves = self._chess.get_status_and_moves()

        # Use fixed value for terminal positions (Stockfish not needed)
        if status == 'checkmate':
            eval_after = 30_000 if player == 'w' else -30_000
        elif status != 'playing':
            eval_after = 0
        else:
            eval_after = self._get_eval()

        reward = self._compute_reward_self_play(player, status, eval_after)
        done   = (status != 'playing') or (self._move_count >= MAX_MOVES)

        if not done:
            self._eval_before = eval_after
            self._legal_moves = next_moves

        return (
            self._encode_state(),
            reward,
            done,
            False,
            {"action_mask": self.action_masks()},
        )

    def close(self):
        if self._sf is not None:
            self._sf.close()
        if self._sf_opponent is not None:
            self._sf_opponent.close()
