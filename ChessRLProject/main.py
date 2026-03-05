import pygame
import sys

from chess_env import ChessEnv
from renderer import (
    WIDTH, HEIGHT,
    draw_board, draw_pieces, draw_highlight, draw_check_highlight,
    draw_promotion_ui, get_promotion_click,
    draw_game_over,
    get_square_from_mouse,
)

# Game states
STATE_PLAYING   = 'playing'
STATE_PROMOTING = 'promoting'
STATE_GAME_OVER = 'game_over'


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess")
    clock = pygame.time.Clock()

    env = ChessEnv()
    state = STATE_PLAYING
    selected_piece = None    # (row, col) of the selected piece
    pending_move = None      # (sr, sc, er, ec) waiting for promotion choice
    promotion_rects = []     # list of (piece, rect) shown during promotion
    game_over_msg = ''
    checked_king_pos = None  # (row, col) of the king in check, or None

    while True:
        # ----------------------------------------------------------------
        # Event handling
        # ----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    state = STATE_PLAYING
                    selected_piece = None
                    pending_move = None
                    checked_king_pos = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

                # --- Promotion choice ---
                if state == STATE_PROMOTING:
                    choice = get_promotion_click(pygame.mouse.get_pos(), promotion_rects)
                    if choice:
                        sr, sc, er, ec = pending_move
                        env.make_move(sr, sc, er, ec, promotion=choice)
                        pending_move = None
                        state, game_over_msg, checked_king_pos = _after_move(env)
                    continue

                # --- Game over: ignore clicks ---
                if state != STATE_PLAYING:
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
                            state, game_over_msg, checked_king_pos = _after_move(env)
                        selected_piece = None
                    elif clicked_piece != ' ' and _is_own(clicked_piece, env.current_player):
                        selected_piece = (row, col)
                    else:
                        selected_piece = None

        # ----------------------------------------------------------------
        # Drawing
        # ----------------------------------------------------------------
        draw_board(screen)
        draw_pieces(screen, env.board)

        # Red ring on king when in check
        if checked_king_pos and state == STATE_PLAYING:
            draw_check_highlight(screen, *checked_king_pos)

        # Yellow ring on selected piece
        if selected_piece and state == STATE_PLAYING:
            draw_highlight(screen, *selected_piece)

        if state == STATE_PROMOTING:
            promotion_rects = draw_promotion_ui(screen, env.current_player)

        if state == STATE_GAME_OVER:
            draw_game_over(screen, game_over_msg)

        pygame.display.flip()
        clock.tick(60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_own(piece, player):
    return (piece.startswith('B') and player == 'w') or \
           (not piece.startswith('B') and player == 'b')


def _after_move(env):
    """Compute state, game-over message and check position after a move."""
    status = env.get_game_status()

    if status == 'checkmate':
        winner = 'Beyaz' if env.current_player == 'b' else 'Siyah'
        return STATE_GAME_OVER, f'SAH MAT! {winner} kazandi!', None

    if status == 'stalemate':
        return STATE_GAME_OVER, 'PAT! Beraberlik!', None

    # Game continues — check if current player is in check
    checked_pos = env.get_checked_king_pos()
    return STATE_PLAYING, '', checked_pos


if __name__ == "__main__":
    main()
