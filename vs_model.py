import numpy as np
import torch
from model import Net

# 盤面の表示関数
def print_board(board):
    board_symbols = {0: '.', 1: 'B', -1: 'W'}
    print("  0 1 2 3 4 5 6 7")
    i = 0
    for row in board:
        print(i, ' '.join(board_symbols[cell] for cell in row))
        i+=1
    print()

# 初期盤面の生成関数
def initial_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3], board[4, 4] = -1, -1
    board[3, 4], board[4, 3] = 1, 1
    return board

# 方向を考慮した合法手の確認関数
def is_valid_move(board, player, x, y):
    if board[x, y] != 0:
        return False
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    opponent = -player
    valid = False
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found_opponent = False
        while 0 <= nx < 8 and 0 <= ny < 8 and board[nx, ny] == opponent:
            nx += dx
            ny += dy
            found_opponent = True
        if found_opponent and 0 <= nx < 8 and 0 <= ny < 8 and board[nx, ny] == player:
            valid = True
            break
    return valid

# 合法手のリストを取得する関数
def get_valid_moves(board, player):
    valid_moves = [(x, y) for x in range(8) for y in range(8) if is_valid_move(board, player, x, y)]
    return valid_moves

# 手を適用する関数
def apply_move(board, player, x, y):
    board[x, y] = player
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    opponent = -player
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        cells_to_flip = []
        while 0 <= nx < 8 and 0 <= ny < 8 and board[nx, ny] == opponent:
            cells_to_flip.append((nx, ny))
            nx += dx
            ny += dy
        if cells_to_flip and 0 <= nx < 8 and 0 <= ny < 8 and board[nx, ny] == player:
            for cx, cy in cells_to_flip:
                board[cx, cy] = player

# モデルを使ってAIの手を選択する関数
def ai_move(model, board, player):
    model.eval()
    with torch.no_grad():
        board_tensor = np.zeros((2, 8, 8), dtype=np.float32)
        board_tensor[0] = (board == player).astype(np.float32)
        board_tensor[1] = (board == -player).astype(np.float32)
        board_tensor = torch.tensor(board_tensor).unsqueeze(0).cuda()
        outputs = model(board_tensor).cpu()
        valid_moves = get_valid_moves(board, player)
        move_probs = np.zeros((8, 8))
        for x, y in valid_moves:
            move_probs[x, y] = outputs[0, x * 8 + y]
        best_move = np.unravel_index(np.argmax(move_probs), (8, 8))
    return best_move

# ゲームの進行
def play_game(model):
    board = initial_board()
    current_player = 1  # 黒からスタート
    while True:
        print_board(board)
        valid_moves = get_valid_moves(board, current_player)
        if not valid_moves:
            if not get_valid_moves(board, -current_player):
                break  # 両方のプレイヤーが合法手を持たない場合、ゲーム終了
            current_player = -current_player  # 手番をスキップ
            continue

        if current_player == 1:
            x, y = map(int, input("Enter move for black (row col): ").split())
            while (x, y) not in valid_moves:
                print("Invalid move. Try again.")
                x, y = map(int, input("Enter move for black (row col): ").split())
        else:
            x, y = ai_move(model, board, current_player)
            print(f"AI (white) moves: {x} {y}")

        apply_move(board, current_player, x, y)
        current_player = -current_player

    print_board(board)
    black_score = np.sum(board == 1)
    white_score = np.sum(board == -1)
    print(f"Final Score - Black: {black_score}, White: {white_score}")

# モデルの読み込み
model = Net().to("cuda")
model.load_state_dict(torch.load('best_model.pth'))

# ゲームの開始
play_game(model)
