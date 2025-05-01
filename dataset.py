import os
import torch
import numpy as np
import shogi
import shogi.CSA
from torch.utils.data import Dataset
from gamestate import GameState

def move_to_xy(move_str):
    from_x = 9 - int(move_str[0])
    from_y = ord(move_str[1]) - ord('a')
    to_x = 9 - int(move_str[2])
    to_y = ord(move_str[3]) - ord('a')
    return (from_x, from_y), (to_x, to_y)

def move_to_class_id(move_str):
    (from_x, from_y), (to_x, to_y) = move_to_xy(move_str)
    return (from_y * 9 + from_x) * 81 + (to_y * 9 + to_x)

def state_to_tensor(grid, hands, turn):
    tensor = np.zeros((40, 9, 9), dtype=np.float32)
    for y in range(9):
        for x in range(9):
            piece = grid[y][x]
            if piece > 0:
                tensor[piece, y, x] = 1
    tensor[39, :, :] = turn / 2  # 手番チャネル
    return torch.tensor(tensor)

class ShogiDataset(Dataset):
    def __init__(self, folder_path, max_files=None):
        self.data = []
        files = [f for f in os.listdir(folder_path) if f.endswith(".csa")]
        if max_files:
            files = files[:max_files]

        for file in files:
            try:
                path = os.path.join(folder_path, file)
                records = shogi.CSA.Parser.parse_file(path)[0]
                moves = records['moves']
            except Exception as e:
                print(f"スキップ: {file} → {e}")
                continue

            gs = GameState()

            for move in moves:
                try:
                    x_tensor = state_to_tensor(gs.board.grid, gs.hands, gs.current_turn)
                    y_label = move_to_class_id(move.usi())
                    self.data.append((x_tensor, y_label))

                    # 手を GameState に反映
                    (fx, fy), (tx, ty) = move_to_xy(move.usi())
                    piece = gs.board.grid[fy][fx]
                    captured = gs.board.grid[ty][tx]
                    if captured != 0:
                        gs.add_to_hand(captured)
                    gs.board.grid[fy][fx] = 0
                    gs.board.grid[ty][tx] = piece
                    gs.switch_turn()
                except Exception as e:
                    print(f"手の適用失敗: {move.usi()} → {e}")
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
