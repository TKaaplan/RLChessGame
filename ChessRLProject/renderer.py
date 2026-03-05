import pygame
import os

WIDTH, HEIGHT = 512, 512
DIMENSION = 8
SQUARE_SIZE = HEIGHT // DIMENSION

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (255, 255, 0)
PROMOTION_BG = (30, 30, 30, 200)      # semi-transparent dark panel
PROMOTION_SQUARE = (255, 255, 255)
PROMOTION_BORDER = (0, 0, 0)
GAMEOVER_BG = (0, 0, 0, 160)

PIECES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pieces")

_image_cache = {}


def _load_piece_image(piece):
    if piece not in _image_cache:
        path = os.path.join(PIECES_FOLDER, piece + ".png")
        img = pygame.image.load(path)
        _image_cache[piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
    return _image_cache[piece]


# ---------------------------------------------------------------------------
# Board drawing
# ---------------------------------------------------------------------------

def draw_board(screen):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color,
                             pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE,
                                         SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(screen, board):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != ' ':
                img = _load_piece_image(piece)
                screen.blit(img, pygame.Rect(col * SQUARE_SIZE,
                                             (7 - row) * SQUARE_SIZE,
                                             SQUARE_SIZE, SQUARE_SIZE))


def draw_highlight(screen, row, col):
    pygame.draw.rect(screen, HIGHLIGHT_COLOR,
                     pygame.Rect(col * SQUARE_SIZE,
                                 (7 - row) * SQUARE_SIZE,
                                 SQUARE_SIZE, SQUARE_SIZE), 3)


def draw_check_highlight(screen, row, col):
    """Draw a red ring around the king that is in check."""
    pygame.draw.rect(screen, (220, 30, 30),
                     pygame.Rect(col * SQUARE_SIZE,
                                 (7 - row) * SQUARE_SIZE,
                                 SQUARE_SIZE, SQUARE_SIZE), 4)


# ---------------------------------------------------------------------------
# Pawn promotion overlay
# ---------------------------------------------------------------------------

# White promotes to: BQ, BR, BN, BB
# Black promotes to:  q,  r,  n,  b
_PROMOTION_PIECES = {
    'w': ['BQ', 'BR', 'BN', 'BB'],
    'b': ['q',  'r',  'n',  'b'],
}

_PADDING = 12   # px between promotion squares


def draw_promotion_ui(screen, player):
    """
    Draw a centered promotion choice panel.
    Returns list of (piece_str, pygame.Rect) for hit-testing.
    """
    pieces = _PROMOTION_PIECES[player]
    n = len(pieces)
    panel_w = n * SQUARE_SIZE + (n + 1) * _PADDING
    panel_h = SQUARE_SIZE + 2 * _PADDING

    panel_x = (WIDTH - panel_w) // 2
    panel_y = (HEIGHT - panel_h) // 2

    # Semi-transparent overlay over the whole board
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    # Panel background
    panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    pygame.draw.rect(screen, (50, 50, 50), panel_rect, border_radius=8)
    pygame.draw.rect(screen, (200, 200, 200), panel_rect, 2, border_radius=8)

    # Piece squares
    rects = []
    for i, piece in enumerate(pieces):
        x = panel_x + _PADDING + i * (SQUARE_SIZE + _PADDING)
        y = panel_y + _PADDING
        rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(screen, PROMOTION_SQUARE, rect, border_radius=4)
        pygame.draw.rect(screen, PROMOTION_BORDER, rect, 1, border_radius=4)
        screen.blit(_load_piece_image(piece), rect)
        rects.append((piece, rect))

    return rects


def get_promotion_click(mouse_pos, promotion_rects):
    """Return piece string if mouse_pos is inside one of the promotion rects, else None."""
    for piece, rect in promotion_rects:
        if rect.collidepoint(mouse_pos):
            return piece
    return None


# ---------------------------------------------------------------------------
# Game-over overlay
# ---------------------------------------------------------------------------

def draw_game_over(screen, message):
    """Draw a centered game-over message overlay."""
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    # Use default pygame font (no external font dependency)
    font_big   = pygame.font.Font(None, 52)
    font_small = pygame.font.Font(None, 30)

    text_surf = font_big.render(message, True, (255, 220, 50))
    hint_surf = font_small.render("R: yeniden baslat", True, (200, 200, 200))

    screen.blit(text_surf,
                text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
    screen.blit(hint_surf,
                hint_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 36)))


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def get_square_from_mouse(mouse_pos):
    col = mouse_pos[0] // SQUARE_SIZE
    row = 7 - (mouse_pos[1] // SQUARE_SIZE)
    return row, col
