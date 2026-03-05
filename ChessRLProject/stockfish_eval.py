"""
Stockfish position evaluator.

Converts our custom board representation to python-chess format,
then queries Stockfish for a centipawn evaluation.

Piece notation (ours → FEN symbol):
  'BP'→'P', 'BN'→'N', 'BB'→'B', 'BR'→'R', 'BQ'→'Q', 'BK'→'K'
  'p' →'p', 'n' →'n', 'b' →'b', 'r' →'r', 'q' →'q', 'k' →'k'

Board indexing:
  Our row 0 = black back rank (top),  row 7 = white back rank (bottom)
  python-chess rank = 7 - our_row,    file  = our_col
"""

import chess
import chess.engine

# ── piece mapping ─────────────────────────────────────────────────────────────

_PIECE_TO_SYM: dict[str, str] = {
    'BP': 'P', 'BN': 'N', 'BB': 'B', 'BR': 'R', 'BQ': 'Q', 'BK': 'K',
    'p':  'p', 'n':  'n', 'b':  'b', 'r':  'r', 'q':  'q', 'k':  'k',
}


# ── board conversion ──────────────────────────────────────────────────────────

def board_to_chess(
    board,
    current_player: str,
    en_passant_possible=None,
    king_moved: dict | None = None,
    rook_moved: dict | None = None,
) -> chess.Board:
    """
    Convert our 8×8 list-of-lists board to a python-chess Board object.

    en_passant_possible: (pawn_row, pawn_col) of the double-moved pawn,
                         as stored in ChessEnv (not the FEN target square).
    king_moved:  {'w': bool, 'b': bool}
    rook_moved:  {'w': [queen_side_moved, king_side_moved],
                  'b': [queen_side_moved, king_side_moved]}
    """
    cb = chess.Board(fen=None)
    cb.clear()

    # Place pieces
    for r in range(8):
        for c in range(8):
            sym = _PIECE_TO_SYM.get(board[r][c])
            if sym:
                cb.set_piece_at(
                    chess.square(c, 7 - r),          # file=col, rank=7-row
                    chess.Piece.from_symbol(sym),
                )

    # Side to move
    cb.turn = chess.WHITE if current_player == 'w' else chess.BLACK

    # Castling rights
    if king_moved and rook_moved:
        rights = chess.BB_EMPTY
        if not king_moved['w']:
            if not rook_moved['w'][1]: rights |= chess.BB_H1   # K-side white
            if not rook_moved['w'][0]: rights |= chess.BB_A1   # Q-side white
        if not king_moved['b']:
            if not rook_moved['b'][1]: rights |= chess.BB_H8   # K-side black
            if not rook_moved['b'][0]: rights |= chess.BB_A8   # Q-side black
        cb.castling_rights = rights
    else:
        # Assume all castling available (opening position)
        cb.castling_rights = chess.BB_A1 | chess.BB_H1 | chess.BB_A8 | chess.BB_H8

    # En passant target square
    # en_passant_possible = (pawn_row, pawn_col) — the pawn's current position
    # after a double move.  The capture TARGET square is one row behind it.
    #   White pawn double-moved → sits at row 4 → target at row 5
    #   Black pawn double-moved → sits at row 3 → target at row 2
    if en_passant_possible:
        ep_row, ep_col = en_passant_possible
        ep_target_row = ep_row + 1 if ep_row == 4 else ep_row - 1
        cb.ep_square = chess.square(ep_col, 7 - ep_target_row)

    return cb


# ── evaluator ─────────────────────────────────────────────────────────────────

class StockfishEvaluator:
    """
    Thin wrapper around Stockfish via python-chess.

    evaluate() returns centipawns from WHITE's perspective:
      positive  → white is better
      negative  → black is better
      ±30000    → forced mate
    """

    def __init__(self, path: str, depth: int = 8):
        """
        path:  absolute path to the Stockfish executable.
        depth: search depth (lower = faster; 8 is a good training default).
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth  = depth

    def evaluate(
        self,
        board,
        current_player: str,
        en_passant_possible=None,
        king_moved: dict | None = None,
        rook_moved: dict | None = None,
    ) -> int:
        """Return centipawn score from white's perspective."""
        cb = board_to_chess(
            board, current_player, en_passant_possible,
            king_moved, rook_moved,
        )
        try:
            info  = self.engine.analyse(cb, chess.engine.Limit(depth=self.depth))
            score = info['score'].white()
            if score.is_mate():
                return 30_000 if score.mate() > 0 else -30_000
            return score.score(mate_score=30_000)
        except Exception:
            return 0

    def get_best_move(
        self,
        board,
        current_player: str,
        en_passant_possible=None,
        king_moved: dict | None = None,
        rook_moved: dict | None = None,
    ):
        """
        Return Stockfish's best move as (from_row, from_col, to_row, to_col, promotion).

        promotion is None or a piece string matching our notation:
          'BQ'/'BN'/'BR'/'BB' for 'w' player, 'q'/'n'/'r'/'b' for 'b' player.
        Returns None if Stockfish has no legal move (shouldn't happen in normal play).
        """
        cb = board_to_chess(
            board, current_player, en_passant_possible,
            king_moved, rook_moved,
        )
        try:
            result = self.engine.play(cb, chess.engine.Limit(depth=self.depth))
            move   = result.move
            if move is None:
                return None

            from_row = 7 - chess.square_rank(move.from_square)
            from_col = chess.square_file(move.from_square)
            to_row   = 7 - chess.square_rank(move.to_square)
            to_col   = chess.square_file(move.to_square)

            promotion = None
            if move.promotion:
                sym = chess.piece_symbol(move.promotion)  # e.g. 'q', 'n', 'r', 'b'
                promotion = ('B' + sym.upper()) if current_player == 'w' else sym

            return from_row, from_col, to_row, to_col, promotion
        except Exception:
            return None

    def close(self):
        self.engine.quit()
