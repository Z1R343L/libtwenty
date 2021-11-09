#! /usr/bin/python3
"""[2048 lib]"""
import secrets
from copy import deepcopy

import numpy as np
from numpy.random import choice
from PIL import Image, ImageDraw, ImageFont
from ruamel.yaml import YAML

move_dict = {"up": 0, "right": 1, "down": 2, "left": 3}

image_size = 800
tile_size = int(image_size / 4)
tile_outline = 6
tile_radius = 20
font = ImageFont.truetype("assets/AGAALER.TTF", 48)

yaml = YAML()
with open("assets/t_colors.yaml", "r", encoding="utf-8") as file:
    t_colors = yaml.load(file)

t_range = list(t_colors.keys())


def font_color(tile: int) -> int:
    if tile in {2, 4}:
        return 0xFF656E77
    else:
        return 0xFFF1F6F8


def prep_tiles() -> dict[Image]:
    tiles = {}
    for t in t_range:
        t_im = Image.new("RGBA", (tile_size, tile_size), color=0x00000000)
        t_id = ImageDraw.Draw(t_im)
        t_id.rounded_rectangle(
            xy=[(0, 0), (tile_size, tile_size)],
            fill=t_colors[t],
            outline=0x00000000,
            width=tile_outline,
            radius=tile_radius,
        )
        if t != 0:
            tw, th = font.getsize(str(t))
            xt, yt = ((tile_size - tw) / 2), ((tile_size - th) / 2)
            t_id.text(xy=(xt, yt), text=str(t), font=font, fill=font_color(t))
        tiles[t] = t_im
    return tiles


tiles = prep_tiles()


def stack(board) -> None:
    for i in range(len(board)):
        for j in range(len(board)):
            k = i
            while board[k][j] == 0:
                if k == len(board) - 1:
                    break
                k += 1
            if k != i:
                board[i][j], board[k][j] = board[k][j], 0


def sum_up(board) -> None:
    for i in range(len(board) - 1):
        for j in range(len(board)):
            if board[i][j] != 0 and board[i][j] == board[i + 1][j]:
                board[i][j] += board[i + 1][j]
                board[i + 1][j] = 0


def spawn_tile(board) -> np.ndarray:
    zeroes_flatten = np.where(board == 0)
    zeroes_indices = [(x, y) for x, y in zip(zeroes_flatten[0], zeroes_flatten[1])]
    random_index = zeroes_indices[choice(len(zeroes_indices), 1)[0]]
    board[random_index] = secrets.choice([2, 2, 4])
    return board


class Board:
    def __init__(self, size: int = 4) -> None:
        """[2048 board]
        Args:
            size (int, optional): [board size]. Defaults to 4.
        """
        self.__dict__.clear()
        self.__board = np.zeros((size, size), int)
        self.__board = spawn_tile(board=self.__board)
        self.__board = spawn_tile(board=self.__board)
        self.__score = self.__board.sum()
        self.update_possible_moves()

    def board_string(self) -> str:
        return str(self.__board)

    def state_string(self, divider: str = "_") -> str:
        string_list = [str(i) for i in np.nditer(self.__board)]
        return divider.join(string_list)

    def render(self, quant: bool = True) -> Image:
        im = Image.new(
            "RGB",
            (image_size + (tile_outline * 2), image_size + (tile_outline * 2)),
            0x8193A4,
        )
        for x in range(4):
            for y in range(4):
                im_t = tiles[self.__board[x][y]]
                y1, x1 = tile_size * x, tile_size * y
                im.paste(im=im_t, box=(x1 + tile_outline, y1 + tile_outline), mask=im_t)
        if quant:
            im.convert("P", palette=Image.ADAPTIVE)
        return im

    def load(self, data: dict) -> None:
        self.__dict__.clear()
        self.__dict__.update(data)

    def dump(self) -> dict:
        return self.__dict__

    def completed(self) -> bool:
        return len(np.where(self.__board == 0)[0]) == 0

    def move(self, action: str, evaluate: bool = False) -> bool:
        board_copy = deepcopy(self.__board)
        rotated_board = np.rot90(board_copy, move_dict[action])
        stack(rotated_board)
        sum_up(rotated_board)
        stack(rotated_board)
        board_copy = np.rot90(rotated_board, len(self.__board) - move_dict[action])
        if np.array_equal(self.__board, board_copy, equal_nan=False):
            return False

        if not evaluate:
            self.__board = board_copy
            self.__board = spawn_tile(board=self.__board)
            self.calculate_score()
        return True

    def update_possible_moves(self):
        self.__possible_moves = self.possible_moves()

    def possible_moves(self) -> dict:
        res, n, over = {}, 0, False
        for direction in ["left", "right", "up", "down"]:
            res[direction] = self.move(action=direction, evaluate=True)
            if not res[direction]:
                n += 1
        if n == 4:
            over = True
        res["over"] = over
        return res

    def calculate_score(self) -> None:
        self.__score = int(self.__board.sum())

    def score(self) -> int:
        return int(self.__score)
