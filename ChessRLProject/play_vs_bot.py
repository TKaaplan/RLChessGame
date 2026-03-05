"""
RL botuna karşı oyna.

Bot → 'b' (açık taşlar, altta, ilk hamleci)  — eğitildiği taraf
Insan → 'w' (koyu taşlar, üstte)

Kullanım:
    python play_vs_bot.py
    python play_vs_bot.py path/to/model.zip
"""

import sys
import os
import numpy as np
import pygame

# ── Torch monkey-patch (train.py ile aynı) ────────────────────────────────────
import torch as _th
from sb3_contrib.common.maskable.distributions import MaskableCategorical as _MC
from torch.distributions.utils import logits_to_probs as _l2p


def _safe_apply_masking(self, masks):
    if masks is not None:
        device = self.logits.device
        self.masks = _th.as_tensor(masks, dtype=_th.bool, device=device).reshape(self.logits.shape)
        HUGE_NEG = _th.tensor(-1e8, dtype=self.logits.dtype, device=device)
        logits = _th.where(self.masks, self._original_logits, HUGE_NEG)
    else:
        self.masks = None
        logits = self._original_logits
    _th.distributions.Categorical.__init__(self, logits=logits, validate_args=False)
    self.probs = _l2p(self.logits)


_MC.apply_masking = _safe_apply_masking

# ── SB3 imports ───────────────────────────────────────────────────────────────
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Yerel imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_env import ChessEnv
from renderer import (
    WIDTH, HEIGHT,
    draw_board, draw_pieces, draw_highlight, draw_check_highlight,
    draw_promotion_ui, get_promotion_click,
    draw_game_over,
    get_square_from_mouse,
)


# ── CNN (train.py ile birebir aynı olmalı) ────────────────────────────────────
class ChessCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


# ── Gözlem kodlama ────────────────────────────────────────────────────────────
PIECE_TO_CH = {
    'BP': 0, 'BN': 1, 'BB': 2, 'BR': 3, 'BQ': 4, 'BK': 5,
    'p':  6, 'n':  7, 'b':  8, 'r':  9, 'q': 10, 'k': 11,
}
MAX_MOVES = 200


def encode_state(env, move_count):
    obs = np.zeros((14, 8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            ch = PIECE_TO_CH.get(env.board[r][c])
            if ch is not None:
                obs[ch, r, c] = 1.0
    if env.current_player == 'w':
        obs[12, :, :] = 1.0
    obs[13, :, :] = move_count / MAX_MOVES
    return obs


def build_action_mask(env):
    mask = np.zeros(4096, dtype=bool)
    for (sr, sc, er, ec) in env.get_legal_moves():
        mask[(sr * 8 + sc) * 64 + (er * 8 + ec)] = True
    if not mask.any():
        mask[0] = True
    return mask


def decode_action(action):
    from_sq = action // 64
    to_sq   = action % 64
    return from_sq // 8, from_sq % 8, to_sq // 8, to_sq % 8


# ── Oyun durumları ─────────────────────────────────────────────────────────────
STATE_PLAYING   = 'playing'
STATE_PROMOTING = 'promoting'
STATE_GAME_OVER = 'game_over'

BOT_PLAYER   = 'b'   # eğitimde kullandığı taraf
HUMAN_PLAYER = 'w'


def _is_own(piece, player):
    return (piece.startswith('B') and player == 'w') or \
           (not piece.startswith('B') and player == 'b')


def _after_move(env):
    status = env.get_game_status()
    if status == 'checkmate':
        winner = 'Bot' if env.current_player == HUMAN_PLAYER else 'Sen'
        return STATE_GAME_OVER, f'SAH MAT! {winner} kazandi!', None
    if status == 'stalemate':
        return STATE_GAME_OVER, 'PAT! Beraberlik!', None
    return STATE_PLAYING, '', env.get_checked_king_pos()


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "curriculum_sfd3_stable_sfd3.zip")

    print(f"Model yukleniyor: {model_path}")
    model = MaskablePPO.load(
        model_path,
        custom_objects={
            "features_extractor_class": ChessCNN,
            "features_extractor_kwargs": {"features_dim": 256},
        },
        device="cpu",
    )
    print("Model yuklendi! Oyun basliyor...\n")
    print("  Koyu taslar (ust) → SEN (insan)")
    print("  Acik taslar (alt) → BOT")
    print("  R → yeniden baslat\n")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock  = pygame.time.Clock()

    env             = ChessEnv()
    state           = STATE_PLAYING
    selected_piece  = None
    pending_move    = None
    promotion_rects = []
    game_over_msg   = ''
    checked_king_pos = None
    move_count      = 0

    def update_caption():
        if state == STATE_PLAYING:
            if env.current_player == BOT_PLAYER:
                pygame.display.set_caption("Chess vs Bot  |  Bot düsünüyor...")
            else:
                pygame.display.set_caption("Chess vs Bot  |  Senin siran (koyu taslar)")
        elif state == STATE_PROMOTING:
            pygame.display.set_caption("Chess vs Bot  |  Terfi secimi")
        else:
            pygame.display.set_caption(f"Chess vs Bot  |  {game_over_msg}")

    update_caption()

    while True:
        # ── Botun sırası ───────────────────────────────────────────────────────
        if state == STATE_PLAYING and env.current_player == BOT_PLAYER:
            update_caption()
            pygame.display.flip()   # "düşünüyor" başlığını göster

            obs  = encode_state(env, move_count)[np.newaxis]   # (1,14,8,8)
            mask = build_action_mask(env)[np.newaxis]           # (1,4096)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            sr, sc, er, ec = decode_action(int(action.flat[0]))

            promotion = None
            if env.needs_promotion(sr, sc, er, ec):
                promotion = 'q'   # bot 'b' tarafı → küçük harf

            env.make_move(sr, sc, er, ec, promotion=promotion)
            move_count += 1
            state, game_over_msg, checked_king_pos = _after_move(env)
            selected_piece = None

        update_caption()

        # ── Event döngüsü ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    state = STATE_PLAYING
                    selected_piece   = None
                    pending_move     = None
                    checked_king_pos = None
                    move_count       = 0

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

                # Terfi seçimi
                if state == STATE_PROMOTING:
                    choice = get_promotion_click(pygame.mouse.get_pos(), promotion_rects)
                    if choice:
                        sr, sc, er, ec = pending_move
                        env.make_move(sr, sc, er, ec, promotion=choice)
                        move_count  += 1
                        pending_move = None
                        state, game_over_msg, checked_king_pos = _after_move(env)
                    continue

                if state != STATE_PLAYING:
                    continue

                # İnsan sırası değilse tıklamayı yoksay
                if env.current_player != HUMAN_PLAYER:
                    continue

                row, col = get_square_from_mouse(pygame.mouse.get_pos())
                clicked_piece = env.board[row][col]

                if selected_piece is None:
                    if clicked_piece != ' ' and _is_own(clicked_piece, env.current_player):
                        selected_piece = (row, col)
                else:
                    start_row, start_col = selected_piece
                    if env.is_valid_move(start_row, start_col, row, col):
                        if env.needs_promotion(start_row, start_col, row, col):
                            pending_move = (start_row, start_col, row, col)
                            state = STATE_PROMOTING
                        else:
                            env.make_move(start_row, start_col, row, col)
                            move_count += 1
                            state, game_over_msg, checked_king_pos = _after_move(env)
                        selected_piece = None
                    elif clicked_piece != ' ' and _is_own(clicked_piece, env.current_player):
                        selected_piece = (row, col)
                    else:
                        selected_piece = None

        # ── Çizim ─────────────────────────────────────────────────────────────
        draw_board(screen)
        draw_pieces(screen, env.board)

        if checked_king_pos and state == STATE_PLAYING:
            draw_check_highlight(screen, *checked_king_pos)

        if selected_piece and state == STATE_PLAYING:
            draw_highlight(screen, *selected_piece)

        if state == STATE_PROMOTING:
            promotion_rects = draw_promotion_ui(screen, env.current_player)

        if state == STATE_GAME_OVER:
            draw_game_over(screen, game_over_msg)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
