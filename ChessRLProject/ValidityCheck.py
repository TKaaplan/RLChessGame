import pygame

def is_invalid_move(board, start_row, start_col, end_row, end_col, current_player):
    if start_row == end_row and start_col == end_col:
        print("Başlangıç ve bitiş pozisyonları aynı, geçersiz hamle.")
        return True

    if board[start_row][start_col] == ' ':
        print("Başlangıç karesinde taş yok, geçersiz hamle.")
        return True

    if current_player.lower() == 'w':
        if not board[start_row][start_col].isupper():
            print("Başlangıç karesinde kendi renginizden bir taş yok, geçersiz hamle.")
            return True
    else:  
        if not board[start_row][start_col].islower():
            print("Başlangıç karesinde kendi renginizden bir taş yok, geçersiz hamle.")
            return True

    if board[end_row][end_col] != ' ':
        if current_player.lower() == 'w':
            if board[end_row][end_col].isupper():
                print("Hedef karede aynı renkten bir taş var, geçersiz hamle.")
                return True
        else:  
            if board[end_row][end_col].islower():
                print("Hedef karede aynı renkten bir taş var, geçersiz hamle.")
                return True

    return False

def is_square_attacked(board, row, col, player):
    for i in range(8):
        if board[row][i] != ' ' and board[row][i].lower() != player:
            pygame.display.flip()
            return True
        if board[i][col] != ' ' and board[i][col].lower() != player:
            pygame.display.flip()
            return True

    for i in range(1, 8):
        if row - i >= 0 and col + i < 8:
            if board[row - i][col + i] != ' ' and board[row - i][col + i].lower() != player:
                pygame.display.flip()
                return True
        if row - i >= 0 and col - i >= 0:
            if board[row - i][col - i] != ' ' and board[row - i][col - i].lower() != player:
                pygame.display.flip()
                return True
        if row + i < 8 and col + i < 8:
            if board[row + i][col + i] != ' ' and board[row + i][col + i].lower() != player:
                pygame.display.flip()
                return True
        if row + i < 8 and col - i >= 0:
            if board[row + i][col - i] != ' ' and board[row + i][col - i].lower() != player:
                pygame.display.flip()
                return True

    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
    for dr, dc in knight_moves:
        if 0 <= row + dr < 8 and 0 <= col + dc < 8:
            if board[row + dr][col + dc].lower() == 'n' and board[row + dr][col + dc].islower() != player.islower():
                pygame.display.flip()
                return True

    rook_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dr, dc in rook_directions:
        r, c = row, col
        while 0 <= r + dr < 8 and 0 <= c + dc < 8:
            r += dr
            c += dc
            if board[r][c] != ' ':
                if board[r][c].lower() == 'r' and board[r][c].islower() != player.islower():
                    pygame.display.flip()
                    return True
                else:
                    break

    bishop_directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for dr, dc in bishop_directions:
        r, c = row, col
        while 0 <= r + dr < 8 and 0 <= c + dc < 8:
            r += dr
            c += dc
            if board[r][c] != ' ':
                if board[r][c].lower() == 'b' and board[r][c].islower() != player.islower():
                    return True
                else:
                    break
                
    king_moves = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
    for dr, dc in king_moves:
        if 0 <= row + dr < 8 and 0 <= col + dc < 8:
            if board[row + dr][col + dc].lower() == 'k' and board[row + dr][col + dc].islower() != player.islower():
                return True

    return False


def is_valid_move(board, start_row, start_col, end_row, end_col, player, en_passant_possible):
    piece = board[start_row][start_col]
    target = board[end_row][end_col]

    
    if (player == 'w' and not piece.startswith('B')) or (player == 'b' and piece.startswith('B')):
        return False
    
    if target != ' ':
        if player == 'w' and target.isupper():
            return False
        if player == 'b' and target.islower():
            return False    
    piece = piece[1:] if piece.startswith('B') else piece
    
    if piece.lower() == 'p':
        return is_valid_pawn_move(board, start_row, start_col, end_row, end_col, player, en_passant_possible)
    elif piece.lower() == 'r':
        return is_valid_rook_move(board, start_row, start_col, end_row, end_col)
    elif piece.lower() == 'n':
        return is_valid_knight_move(start_row, start_col, end_row, end_col)
    elif piece.lower() == 'b':
        return is_valid_bishop_move(board, start_row, start_col, end_row, end_col)
    elif piece.lower() == 'q':
        return is_valid_queen_move(board, start_row, start_col, end_row, end_col)
    elif piece.lower() == 'k':
        return is_valid_king_move(start_row, start_col, end_row, end_col) or is_valid_castling(board, start_row, start_col, end_row, end_col, player)
    
    return False

def is_valid_pawn_move(board, start_row, start_col, end_row, end_col, player, en_passant_possible):
    direction = -1 if player == 'w' else 1
    
    if start_col == end_col and end_row == start_row + direction and board[end_row][end_col] == ' ':
        return True
    
    if start_col == end_col and end_row == start_row + 2 * direction and board[end_row][end_col] == ' ' and board[start_row + direction][start_col] == ' ':
        return (player == 'w' and start_row == 6) or (player == 'b' and start_row == 1)
    
    if abs(start_col - end_col) == 1 and end_row == start_row + direction:
        if player == 'w' and not board[end_row][end_col].startswith('B') and board[end_row][end_col] != ' ':
            return True
        if player == 'b' and board[end_row][end_col].startswith('B'):
            return True
    
    if en_passant_possible and abs(start_col - end_col) == 1 and end_row == start_row + direction:
        if player == 'w' and start_row == 3 and board[start_row][end_col] == 'p':
            return True
        if player == 'b' and start_row == 4 and board[start_row][end_col] == 'P':
            return True
    
    return False

def is_valid_rook_move(board, start_row, start_col, end_row, end_col):
    if start_row != end_row and start_col != end_col:
        return False
    
    row_direction = 0 if start_row == end_row else (1 if end_row > start_row else -1)
    col_direction = 0 if start_col == end_col else (1 if end_col > start_col else -1)
    
    current_row, current_col = start_row + row_direction, start_col + col_direction
    while (current_row, current_col) != (end_row, end_col):
        if board[current_row][current_col] != ' ':
            return False
        current_row += row_direction
        current_col += col_direction
    
    return True

def is_valid_knight_move(start_row, start_col, end_row, end_col):
    row_diff = abs(end_row - start_row)
    col_diff = abs(end_col - start_col)
    return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)

def is_valid_bishop_move(board, start_row, start_col, end_row, end_col):
    if abs(end_row - start_row) != abs(end_col - start_col):
        return False
    
    row_direction = 1 if end_row > start_row else -1
    col_direction = 1 if end_col > start_col else -1
    
    current_row, current_col = start_row + row_direction, start_col + col_direction
    while (current_row, current_col) != (end_row, end_col):
        if board[current_row][current_col] != ' ':
            return False
        current_row += row_direction
        current_col += col_direction
    
    return True

def is_valid_queen_move(board, start_row, start_col, end_row, end_col):
    return is_valid_rook_move(board, start_row, start_col, end_row, end_col) or is_valid_bishop_move(board, start_row, start_col, end_row, end_col)

def is_valid_king_move(start_row, start_col, end_row, end_col):
    row_diff = abs(end_row - start_row)
    col_diff = abs(end_col - start_col)
    return row_diff <= 1 and col_diff <= 1

def is_valid_castling(board, start_row, start_col, end_row, end_col, player):
    if board[start_row][start_col].lower() != 'k':
        return False
    
    if end_row != start_row:
        return False
    
    if abs(end_col - start_col) != 2:
        return False
    
    if player == 'w':
        row = 7
    else:
        row = 0
    
    if end_col == 6:  # Kingside castling
        if board[row][5] != ' ' or board[row][6] != ' ':
            return False
        if is_square_attacked(board, row, 4, player) or is_square_attacked(board, row, 5, player):
            return False
    elif end_col == 2:  # Queenside castling
        if board[row][1] != ' ' or board[row][2] != ' ' or board[row][3] != ' ':
            return False
        if is_square_attacked(board, row, 2, player) or is_square_attacked(board, row, 3, player):
            return False
    else:
        return False
    
    return True