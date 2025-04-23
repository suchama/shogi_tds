import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


#ファイルを開く。一局文の棋譜がlinesというリストで取得できる。
file_path = ""
with open(file_path, "r", encodeing="utf-8") as f:
    lines = f.readlines()

#棋譜読み込み関数
def load_csa_files(folder_path):
    file_list = glob.glob(f"{folder_path}/*.csa")
    games = []

    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
        moves = extract_moves_from_csa(lines)
        parsed_moves = [parse_move(line) for line in moves]
        games.append(parsed_moves)
    
    return games

def board_to_tensor(board):
    tensor = np.zeros((40, 9, 9), dtype=np.float32)

    for y in range(9):
        for x in range(9):
            piece = board[y][x]
            if piece != 0:
                if piece < 10:
                    tensor[piece - 1, y, x] = 1
                else:
                    tensor[piece - 11 + 20, y, x] = 1
    return tensor

class ShogiDataset(Dataset):
    def __init__(self, folder_path):
        self.games = load_csa_files(folder_path)
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        for game in self.games:
            board = self.init_board()

            for move in game:
                input_tensor = board_to_tensor(board)
                label = self.encode_move(move)

                samples.append((input_tensor, label))

                # 盤面更新（簡易実装）
                from_x, from_y = move['from']
                to_x, to_y = move['to']
                piece = move['piece']
                board[to_y][to_x] = piece
                board[from_y][from_x] = 0

        return samples

    def init_board(self):
        # 初期盤面セット
        # P1〜P9 行を処理しても良いし、固定で初期化しても良い
        board = np.zeros((9, 9), dtype=int)
        # 簡単のため、固定初期配置を書く（実際はファイルから読んでもOK）
        # ここは必要に応じて完成させましょう！
        return board

    def encode_move(self, move):
        # move をクラスラベルにエンコード
        # たとえば from (9x9) + to (9x9) で 81x81 = 6561 クラス
        from_x, from_y = move['from']
        to_x, to_y = move['to']
        label = from_y * 9 * 9 * 9 + from_x * 9 * 9 + to_y * 9 + to_x
        return label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_tensor, label = self.samples[idx]
        return torch.tensor(input_tensor), torch.tensor(label)


dataset = ShogiDataset(folder_path='path/to/csa_files')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for inputs, labels in dataloader:
    print(inputs.shape)  # torch.Size([32, 40, 9, 9])
    print(labels.shape)  # torch.Size([32])


