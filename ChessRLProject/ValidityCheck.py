import pygame

king_moved = {'w': False, 'b': False}
rook_moved = {'w': [False, False], 'b': [False, False]}
castling_done = {'w': False, 'b': False}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_opponent(piece, player):
    """Return True if piece belongs to the opponent of player."""
    if piece == ' ':
        return False
    return piece.startswith('B') if player == 'b' else not piece.startswith('B')


def _piece_type(piece):
    """Return single-char lowercase piece type (e.g. 'BP' → 'p', 'k' → 'k')."""
    return (piece[1:] if piece.startswith('B') else piece).lower()


# ---------------------------------------------------------------------------
# Square attack detection
# ---------------------------------------------------------------------------

def is_square_attacked(board, row, col, player):
    """
    Return True if (row, col) is attacked by any opponent piece.
    player: the player whose square is being checked (opponent of player attacks it).
    """
    # --- Rook / Queen (ranks and files) ---
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            p = board[r][c]
            if p != ' ':
                if _is_opponent(p, player) and _piece_type(p) in ('r', 'q'):
                    return True
                break
            r += dr
            c += dc

    # --- Bishop / Queen (diagonals) ---
    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8:
            p = board[r][c]
            if p != ' ':
                if _is_opponent(p, player) and _piece_type(p) in ('b', 'q'):
                    return True
                break
            r += dr
            c += dc

    # --- Knight ---
    for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                   (1, 2), (1, -2), (-1, 2), (-1, -2)]:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            p = board[r][c]
            if _is_opponent(p, player) and _piece_type(p) == 'n':
                return True

    # --- King ---
    for dr, dc in [(1, 1), (1, 0), (1, -1), (0, 1),
                   (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            p = board[r][c]
            if _is_opponent(p, player) and _piece_type(p) == 'k':
                return True

    # --- Pawn ---
    # White pawns (B-prefix) move in direction -1 (decreasing row).
    # A white pawn at (row+1, col±1) attacks (row, col).
    # Black pawns move in direction +1.
    # A black pawn at (row-1, col±1) attacks (row, col).
    if player == 'b':  # opponent is white, white pawns attack from below
        for dc in [-1, 1]:
            r, c = row + 1, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == 'BP':
                return True
    else:  # opponent is black, black pawns attack from above
        for dc in [-1, 1]:
            r, c = row - 1, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == 'p':
                return True

    return False


def find_king(board, player):
    """Return (row, col) of the player's king, or None if not found."""
    king = 'BK' if player == 'w' else 'k'
    for r in range(8):
        for c in range(8):
            if board[r][c] == king:
                return r, c
    return None


def is_in_check(board, player):
    """Return True if the player's king is currently in check."""
    pos = find_king(board, player)
    if pos is None:
        return False
    return is_square_attacked(board, pos[0], pos[1], player)


# ---------------------------------------------------------------------------
# Piece move validators
# ---------------------------------------------------------------------------

def is_valid_king_move(board, start_row, start_col, end_row, end_col, player):
    row_diff = abs(end_row - start_row)
    col_diff = abs(end_col - start_col)
    # Standard king movement: at most one square in any direction
    return row_diff <= 1 and col_diff <= 1


def is_valid_castling(board, start_row, start_col, end_row, end_col, player):
    """Check only that squares are empty and rook is present (no check logic here)."""
    if player == 'w':
        if start_row == 7 and start_col == 4:
            if end_row == 7 and end_col == 6:   # King-side
                return (board[7][5] == ' ' and board[7][6] == ' '
                        and board[7][7] == 'BR')
            elif end_row == 7 and end_col == 2:  # Queen-side
                return (board[7][1] == ' ' and board[7][2] == ' '
                        and board[7][3] == ' ' and board[7][0] == 'BR')
    elif player == 'b':
        if start_row == 0 and start_col == 4:
            if end_row == 0 and end_col == 6:   # King-side
                return (board[0][5] == ' ' and board[0][6] == ' '
                        and board[0][7] == 'r')
            elif end_row == 0 and end_col == 2:  # Queen-side
                return (board[0][1] == ' ' and board[0][2] == ' '
                        and board[0][3] == ' ' and board[0][0] == 'r')
    return False


def perform_castling(board, start_row, start_col, end_row, end_col, player):
    if player == 'w':
        if end_col == 6:    # King-side
            board[7][4] = ' '
            board[7][7] = ' '
            board[7][6] = 'BK'
            board[7][5] = 'BR'
        elif end_col == 2:  # Queen-side
            board[7][4] = ' '
            board[7][0] = ' '
            board[7][2] = 'BK'
            board[7][3] = 'BR'
    elif player == 'b':
        if end_col == 6:    # King-side
            board[0][4] = ' '
            board[0][7] = ' '
            board[0][6] = 'k'
            board[0][5] = 'r'
        elif end_col == 2:  # Queen-side
            board[0][4] = ' '
            board[0][0] = ' '
            board[0][2] = 'k'
            board[0][3] = 'r'
    king_moved[player] = True
    castling_done[player] = True


def is_valid_pawn_move(board, start_row, start_col, end_row, end_col, player, en_passant_possible):
    direction = -1 if player == 'w' else 1

    # One square forward
    if start_col == end_col and end_row == start_row + direction and board[end_row][end_col] == ' ':
        return True

    # Two squares forward from starting rank
    if (start_col == end_col and end_row == start_row + 2 * direction
            and board[end_row][end_col] == ' '
            and board[start_row + direction][start_col] == ' '):
        return (player == 'w' and start_row == 6) or (player == 'b' and start_row == 1)

    # Diagonal capture
    if abs(start_col - end_col) == 1 and end_row == start_row + direction:
        target = board[end_row][end_col]
        if player == 'w' and not target.startswith('B') and target != ' ':
            return True
        if player == 'b' and target.startswith('B'):
            return True

    # En passant
    if en_passant_possible and abs(start_col - end_col) == 1 and end_row == start_row + direction:
        if player == 'w' and start_row == 3 and board[start_row][end_col] == 'p':
            return True
        if player == 'b' and start_row == 4 and board[start_row][end_col] == 'BP':
            return True

    return False


def is_valid_rook_move(board, start_row, start_col, end_row, end_col):
    if start_row != end_row and start_col != end_col:
        return False
    row_dir = 0 if start_row == end_row else (1 if end_row > start_row else -1)
    col_dir = 0 if start_col == end_col else (1 if end_col > start_col else -1)
    r, c = start_row + row_dir, start_col + col_dir
    while (r, c) != (end_row, end_col):
        if board[r][c] != ' ':
            return False
        r += row_dir
        c += col_dir
    return True


def is_valid_knight_move(start_row, start_col, end_row, end_col):
    rd = abs(end_row - start_row)
    cd = abs(end_col - start_col)
    return (rd == 2 and cd == 1) or (rd == 1 and cd == 2)


def is_valid_bishop_move(board, start_row, start_col, end_row, end_col):
    if abs(end_row - start_row) != abs(end_col - start_col):
        return False
    row_dir = 1 if end_row > start_row else -1
    col_dir = 1 if end_col > start_col else -1
    r, c = start_row + row_dir, start_col + col_dir
    while (r, c) != (end_row, end_col):
        if board[r][c] != ' ':
            return False
        r += row_dir
        c += col_dir
    return True


def is_valid_queen_move(board, start_row, start_col, end_row, end_col):
    return (is_valid_rook_move(board, start_row, start_col, end_row, end_col) or
            is_valid_bishop_move(board, start_row, start_col, end_row, end_col))


# ---------------------------------------------------------------------------
# Legacy helpers (kept for compatibility)
# ---------------------------------------------------------------------------

def is_invalid_move(board, start_row, start_col, end_row, end_col, current_player):
    if start_row == end_row and start_col == end_col:
        return True
    if board[start_row][start_col] == ' ':
        return True
    if current_player.lower() == 'w':
        if not board[start_row][start_col].isupper():
            return True
    else:
        if not board[start_row][start_col].islower():
            return True
    if board[end_row][end_col] != ' ':
        if current_player.lower() == 'w' and board[end_row][end_col].isupper():
            return True
        if current_player.lower() == 'b' and board[end_row][end_col].islower():
            return True
    return False
