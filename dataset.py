import os
import torch
import numpy as np
import shogi
import shogi.CSA
from torch.utils.data import Dataset
from gamestate import GameState

def move_to_xy(move_str):
    if "*" in move_str:  # 打ち駒
        to_x = 9 - int(move_str[2])
        to_y = ord(move_str[3]) - ord('a')
        return None, (to_x, to_y)
    else:  # 通常の移動
        from_x = 9 - int(move_str[0])
        from_y = ord(move_str[1]) - ord('a')
        to_x = 9 - int(move_str[2])
        to_y = ord(move_str[3]) - ord('a')
        return (from_x, from_y), (to_x, to_y)

def move_to_class_id(move_str):
    (from_pos, to_pos) = move_to_xy(move_str)
    to_x, to_y = to_pos
    if from_pos is None:
        from_x, from_y = 0, 0  # 打ち駒用の仮位置（分類用に固定）
    else:
        from_x, from_y = from_pos
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
                    y_label = move_to_class_id(move)
                    self.data.append((x_tensor, y_label))

                    from_pos, to_pos = move_to_xy(move)

                    if from_pos is None:
                        # 打ち駒処理
                        piece_symbol = move[0]  # 例: "P"
                        to_x, to_y = to_pos

                        symbol_to_code = {"P": 8, "L": 7, "N": 6, "S": 5, "G": 4, "B": 3, "R": 2}
                        piece_code = symbol_to_code.get(piece_symbol.upper(), None)
                        if piece_code is None:
                            raise ValueError(f"未知の駒: {piece_symbol}")

                        # 持ち駒がない、空いていない、二歩などで失敗する場合も強制的に反映
                        success = gs.drop_from_hand(piece_code, to_x, to_y)
                        if not success:
                            # 強制適用（注意：正しくない可能性もある）
                            gs.board.grid[to_y][to_x] = piece_code if gs.current_turn == 1 else piece_code + 10
                            gs.hands[gs.current_turn][piece_code] = gs.hands[gs.current_turn].get(piece_code, 1) - 1
                            if gs.hands[gs.current_turn][piece_code] <= 0:
                                del gs.hands[gs.current_turn][piece_code]

                    else:
                        fx, fy = from_pos
                        tx, ty = to_pos
                        piece = gs.board.grid[fy][fx]
                        captured = gs.board.grid[ty][tx]
                        if captured != 0:
                            gs.add_to_hand(captured)
                        gs.board.grid[fy][fx] = 0
                        gs.board.grid[ty][tx] = piece

                    gs.switch_turn()
                except Exception as e:
                    print(f"手の適用失敗: {move} → {e}")
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
