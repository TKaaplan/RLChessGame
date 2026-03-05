"""
Chess environment — board state and game logic.

Piece notation:
  White pieces: 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK'  (B = Bottom/White)
  Black pieces: 'p',  'r',  'n',  'b',  'q',  'k'   (lowercase)
  Empty square: ' '

Players: 'w' = white, 'b' = black
"""

from ValidityCheck import (
    is_valid_pawn_move,
    is_valid_rook_move,
    is_valid_knight_move,
    is_valid_bishop_move,
    is_valid_queen_move,
    is_valid_king_move,
    is_valid_castling,
    perform_castling,
    is_square_attacked,
    is_in_check,
)

STARTING_BOARD = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["BP", "BP", "BP", "BP", "BP", "BP", "BP", "BP"],
    ["BR", "BN", "BB", "BQ", "BK", "BB", "BN", "BR"],
]


class ChessEnv:
    """
    Self-contained chess environment.
    Suitable for human play and as a base for RL agents.
    """

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        self.board = [row[:] for row in STARTING_BOARD]
        self.current_player = 'b'        # white moves first (as in the original)
        self.en_passant_possible = None  # (row, col) of double-moved pawn, or None
        self.king_moved = {'w': False, 'b': False}
        # [queen-side rook moved, king-side rook moved]
        self.rook_moved = {'w': [False, False], 'b': [False, False]}
        return self.get_state()

    def get_state(self):
        """Return a deep copy of the board (safe to inspect / store for RL)."""
        return [row[:] for row in self.board]

    # ------------------------------------------------------------------
    # Check / safety helpers
    # ------------------------------------------------------------------

    def get_checked_king_pos(self):
        """Return (row, col) of the current player's king if it is in check, else None."""
        from ValidityCheck import find_king
        if is_in_check(self.board, self.current_player):
            return find_king(self.board, self.current_player)
        return None

    def _leaves_king_in_check(self, start_row, start_col, end_row, end_col):
        """Return True if making this move leaves our own king in check."""
        board_copy = [row[:] for row in self.board]
        piece = board_copy[start_row][start_col]

        if abs(start_col - end_col) == 2 and piece in ('k', 'BK'):
            perform_castling(board_copy, start_row, start_col, end_row, end_col,
                             'b' if piece.islower() else 'w')
        else:
            board_copy[end_row][end_col] = piece
            board_copy[start_row][start_col] = ' '
            # En passant capture
            if piece in ('p', 'BP') and start_col != end_col and self.board[end_row][end_col] == ' ':
                board_copy[start_row][end_col] = ' '

        return is_in_check(board_copy, self.current_player)

    # ------------------------------------------------------------------
    # Move validation
    # ------------------------------------------------------------------

    def is_valid_move(self, start_row, start_col, end_row, end_col):
        board = self.board
        player = self.current_player
        piece = board[start_row][start_col]
        target = board[end_row][end_col]

        # Must move own piece
        if (player == 'w' and not piece.startswith('B')) or \
           (player == 'b' and piece.startswith('B')):
            return False

        # Cannot capture own piece
        if target != ' ':
            if player == 'w' and target.isupper():
                return False
            if player == 'b' and target.islower():
                return False

        # Cannot capture the opponent's king (kings are never legally capturable)
        if target in ('k', 'BK'):
            return False

        piece_type = piece[1:] if piece.startswith('B') else piece

        # --- King ---
        if piece_type.lower() == 'k':
            if abs(end_col - start_col) == 2:
                return self._validate_castling(start_row, start_col, end_row, end_col, player)
            if not is_valid_king_move(board, start_row, start_col, end_row, end_col, player):
                return False
            return not self._leaves_king_in_check(start_row, start_col, end_row, end_col)

        # --- Other pieces ---
        if piece_type.lower() == 'p':
            ok = is_valid_pawn_move(board, start_row, start_col, end_row, end_col,
                                    player, self.en_passant_possible)
        elif piece_type.lower() == 'r':
            ok = is_valid_rook_move(board, start_row, start_col, end_row, end_col)
        elif piece_type.lower() == 'n':
            ok = is_valid_knight_move(start_row, start_col, end_row, end_col)
        elif piece_type.lower() == 'b':
            ok = is_valid_bishop_move(board, start_row, start_col, end_row, end_col)
        elif piece_type.lower() == 'q':
            ok = is_valid_queen_move(board, start_row, start_col, end_row, end_col)
        else:
            return False

        if not ok:
            return False
        return not self._leaves_king_in_check(start_row, start_col, end_row, end_col)

    def _validate_castling(self, start_row, start_col, end_row, end_col, player):
        """Full castling validation: rook/king history + empty squares + check path."""
        # King must not have moved
        if self.king_moved[player]:
            return False

        # Corresponding rook must not have moved
        rook_idx = 1 if end_col > start_col else 0   # 1 = king-side, 0 = queen-side
        if self.rook_moved[player][rook_idx]:
            return False

        # Squares between king and rook must be empty and rook must be present
        if not is_valid_castling(self.board, start_row, start_col, end_row, end_col, player):
            return False

        # King cannot currently be in check
        if is_in_check(self.board, player):
            return False

        # King cannot pass through or land on an attacked square
        direction = 1 if end_col > start_col else -1
        for c in range(start_col, end_col + direction, direction):
            if is_square_attacked(self.board, start_row, c, player):
                return False

        return True

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def needs_promotion(self, start_row, start_col, end_row, end_col):
        """Return True if this move results in a pawn promotion."""
        piece = self.board[start_row][start_col]
        return piece in ('p', 'BP') and (end_row == 0 or end_row == 7)

    def make_move(self, start_row, start_col, end_row, end_col, promotion=None):
        """
        Apply a move. Assumes is_valid_move() returned True.
        promotion: piece string to promote to (e.g. 'BQ', 'q'). If None, auto-queens.
        """
        board = self.board
        piece = board[start_row][start_col]
        player = self.current_player

        # Detect en passant before moving (destination is still empty)
        is_en_passant = (
            piece in ('p', 'BP') and
            start_col != end_col and
            board[end_row][end_col] == ' '
        )

        if abs(start_col - end_col) == 2 and piece in ('k', 'BK'):
            # Castling
            perform_castling(board, start_row, start_col, end_row, end_col,
                             'b' if piece.islower() else 'w')
            self.king_moved[player] = True
        else:
            # Track king / rook history before moving
            if piece in ('k', 'BK'):
                self.king_moved[player] = True
            elif piece == 'BR':
                if start_row == 7 and start_col == 0:
                    self.rook_moved['w'][0] = True
                elif start_row == 7 and start_col == 7:
                    self.rook_moved['w'][1] = True
            elif piece == 'r':
                if start_row == 0 and start_col == 0:
                    self.rook_moved['b'][0] = True
                elif start_row == 0 and start_col == 7:
                    self.rook_moved['b'][1] = True

            board[end_row][end_col] = piece
            board[start_row][start_col] = ' '

        # Pawn promotion
        if piece in ('p', 'BP') and (end_row == 0 or end_row == 7):
            if promotion:
                board[end_row][end_col] = promotion
            else:
                board[end_row][end_col] = 'BQ' if piece == 'BP' else 'q'

        # Track en passant availability for the next move
        if piece in ('p', 'BP') and abs(start_row - end_row) == 2:
            self.en_passant_possible = (end_row, end_col)
        else:
            self.en_passant_possible = None

        # Remove captured pawn for en passant
        if is_en_passant:
            board[start_row][end_col] = ' '

        # Switch turn
        self.current_player = 'b' if player == 'w' else 'w'

    # ------------------------------------------------------------------
    # Game status
    # ------------------------------------------------------------------

    def get_game_status(self):
        """
        Return one of:
          'playing'   – game continues
          'checkmate' – current player has no legal moves and is in check
          'stalemate' – current player has no legal moves but is NOT in check
        """
        status, _ = self.get_status_and_moves()
        return status

    def get_status_and_moves(self):
        """
        Return (status, legal_moves) in a single call to avoid computing
        legal moves twice (once for status, once for the next action mask).
        """
        moves = self.get_legal_moves()
        if moves:
            return 'playing', moves
        if is_in_check(self.board, self.current_player):
            return 'checkmate', []
        return 'stalemate', []

    # ------------------------------------------------------------------
    # Legal move generation (used by RL agents)
    # ------------------------------------------------------------------

    def get_legal_moves(self):
        """Return all legal moves as list of (start_row, start_col, end_row, end_col)."""
        moves = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece == ' ':
                    continue
                is_own = (self.current_player == 'w' and piece.startswith('B')) or \
                         (self.current_player == 'b' and not piece.startswith('B'))
                if not is_own:
                    continue
                for er in range(8):
                    for ec in range(8):
                        if self.is_valid_move(r, c, er, ec):
                            moves.append((r, c, er, ec))
        return moves
