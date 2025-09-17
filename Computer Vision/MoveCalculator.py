import chess

# -----------------------------------------------------------------------------
#  UTILITY: Converting a python-chess Board => 8x8 color layout
# -----------------------------------------------------------------------------
def board_to_color_layout(board: chess.Board):
    """
    Convert a python-chess Board into an 8x8 top-down layout of 'W','B', or None.
    row=0 => rank=8, row=7 => rank=1
    col=0 => file='a', col=7 => file='h'
    """
    layout = []
    for row in range(8):
        rank = 7 - row  # top row=0 => rank=7, bottom row=7 => rank=0
        row_colors = []
        for col in range(8):
            sq = chess.square(col, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row_colors.append(None)
            else:
                row_colors.append('W' if piece.color else 'B')
        layout.append(row_colors)
    return layout

# -----------------------------------------------------------------------------
#  UTILITY: Convert (row,col) => python-chess square index
# -----------------------------------------------------------------------------
def array_to_square(row, col):
    """
    row=0 => top row (rank=7),
    row=7 => bottom row (rank=0).
    col=0 => file 'a', col=7 => file 'h'.
    """
    rank = 7 - row
    file = col
    return chess.square(file, rank)

# -----------------------------------------------------------------------------
#  Compare two color-only boards => list of differences
# -----------------------------------------------------------------------------
def compare_color_boards(before, after):
    """
    Returns a list of (row, col, oldColor, newColor) for every square that differs.
    """
    diffs = []
    for r in range(8):
        for c in range(8):
            if before[r][c] != after[r][c]:
                diffs.append((r, c, before[r][c], after[r][c]))
    return diffs

# -----------------------------------------------------------------------------
#  MAIN: difference-based inference from FEN + color layout
# -----------------------------------------------------------------------------
def difference_based_infer_fen(initial_fen, color_only_board):
    """
    1) Load 'initial_fen' into a python-chess Board (arbitrary position).
    2) Compare its color layout to 'color_only_board'.
    3) If exactly 2 squares differ => interpret as a normal/capture move.
       If 4 squares differ => interpret as castling.
       (No advanced en passant / promotion logic here.)
    4) Update the board (removing piece from 'from' square, placing on 'to' square, etc.).
    5) Return the FEN of the resulting board.
    """
    board = chess.Board(initial_fen)

    # Build the old color layout
    old_layout = board_to_color_layout(board)

    # Normalize 'color_only_board': replace 'N' with None
    new_layout = []
    for row in color_only_board:
        new_row = []
        for val in row:
            if val == 'N':
                new_row.append(None)
            else:
                new_row.append(val)
        new_layout.append(new_row)

    # Find squares that changed
    diffs = compare_color_boards(old_layout, new_layout)
    num_diffs = len(diffs)

    # Helper to "move" a piece on the python-chess Board
    def move_piece(from_rc, to_rc):
        (fr, fc) = from_rc
        (tr, tc) = to_rc
        from_sq = array_to_square(fr, fc)
        to_sq   = array_to_square(tr, tc)

        piece = board.piece_at(from_sq)
        if piece is None:
            raise ValueError(f"No piece at {from_rc} to move.")

        # Remove piece from old square
        board.remove_piece_at(from_sq)
        # If there's a piece at the new square, remove it (capture)
        board.remove_piece_at(to_sq)
        # Place the piece at the new square
        board.set_piece_at(to_sq, piece)

    # -----------------------------------------------------
    #   2 squares changed => normal or capture
    # -----------------------------------------------------
    if num_diffs == 2:
        (r1, c1, oldC1, newC1) = diffs[0]
        (r2, c2, oldC2, newC2) = diffs[1]

        possible_from = None
        possible_to   = None

        # Attempt pattern: color => None, and None/opponent => color
        if oldC1 is not None and newC1 is None:
            if newC2 == oldC1:
                possible_from = (r1, c1)
                possible_to   = (r2, c2)
        if oldC2 is not None and newC2 is None:
            if newC1 == oldC2:
                possible_from = (r2, c2)
                possible_to   = (r1, c1)

        if not possible_from or not possible_to:
            raise ValueError("2 diffs do not match a normal/capture color pattern.")

        move_piece(possible_from, possible_to)
        return board.fen()

    # -----------------------------------------------------
    #   4 squares changed => castling
    # -----------------------------------------------------
    elif num_diffs == 4:
        from_squares = []
        to_squares   = []
        for (r,c, oldC, newC) in diffs:
            # from-square: color => None
            if oldC is not None and newC is None:
                from_squares.append((r,c))
            # to-square: None => color
            elif oldC is None and newC is not None:
                to_squares.append((r,c))
            else:
                raise ValueError("4 diffs include unexpected color transitions (e.g. B->W).")

        if len(from_squares) != 2 or len(to_squares) != 2:
            raise ValueError("Not a standard castling pattern (need exactly 2 from-squares, 2 to-squares).")

        # Identify which 'from' belongs to the king vs. rook by piece type
        king_from = None
        rook_from = None
        for (fr, fc) in from_squares:
            sq = array_to_square(fr, fc)
            piece = board.piece_at(sq)
            if not piece:
                raise ValueError(f"No piece at from-square {fr,fc}")
            if piece.piece_type == chess.KING:
                king_from = (fr, fc)
            elif piece.piece_type == chess.ROOK:
                rook_from = (fr, fc)

        if not king_from or not rook_from:
            raise ValueError("Did not find exactly one King and one Rook among the from-squares.")

        king_from_sq = array_to_square(*king_from)
        king_from_file = chess.square_file(king_from_sq)

        king_to = None
        rook_to = None

        for (tr, tc) in to_squares:
            sq = array_to_square(tr, tc)
            f  = chess.square_file(sq)
            # If the file differs from king's file by exactly 2 => that's the king's new square
            if abs(f - king_from_file) == 2:
                king_to = (tr, tc)
            else:
                rook_to = (tr, tc)

        if not king_to or not rook_to:
            raise ValueError("Could not identify the king's to-square vs. rook's to-square for castling.")

        # Move the King, then the Rook
        move_piece(king_from, king_to)
        move_piece(rook_from, rook_to)

        return board.fen()

    # -----------------------------------------------------
    #   3 squares => likely en passant (not implemented)
    # -----------------------------------------------------
    elif num_diffs == 3:
        raise ValueError("3 squares changed => possibly en passant, not supported here.")

    else:
        # More or fewer => no single standard move
        raise ValueError(f"{num_diffs} squares changed => no single (non-EP) move can do that.")