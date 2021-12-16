"""
[2048 lib]

"""
import secrets
from base64 import urlsafe_b64decode, urlsafe_b64encode
from copy import deepcopy
from os.path import abspath
from pathlib import Path

import numpy as np
from numpy.random import choice
from PIL import Image, ImageDraw, ImageFont
import srsly

move_dict = {"up": 0, "right": 1, "down": 2, "left": 3}

tile_size = 200
tile_outline = 6
tile_radius = 20
assets_path = Path(abspath(__file__)).parent / "assets"
font = ImageFont.truetype(str(assets_path / "AGAALER.TTF"), 52, encoding="unic")

t_colors = srsly.read_yaml(assets_path / "t_colors.yaml")

t_range = list(t_colors.keys())


def font_color(tile: int) -> int:
    if tile in {2, 4}:
        return 0xFF656E77
    else:
        return 0xFFF1F6F8


def prep_tiles() -> dict:
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
    for i in range(board.size):
        for j in range(board.size):
            k = i
            while board[k][j] == 0:
                if k == board.size - 1:
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
        """
        [2048 board]
        Args:
            size (int, optional): [board size]. Defaults to 4.
        """
        self.__dict__.clear()
        self.board = np.zeros((size, size), int)
        self.board = spawn_tile(board=self.board)
        self.board = spawn_tile(board=self.board)
        self.score = self.board.sum()
        self.update_possible_moves()
        self.size = size

    def board_string(self) -> str:
        """
        [returns the boards ndarray as str]

        Returns:
            str: [the boards ndarray as str]
        """
        return str(self.board)

    def state_string(self, divider: str = "_") -> str:
        """
        [returns the tile values as flattened str]

        Args:
            divider (str, optional): [str to divide the values with]. Defaults to "_".

        Returns:
            str: [flattened tile-values string]
        """
        string_list = [str(i) for i in np.nditer(self.board)]
        return divider.join(string_list)

    def render(self, quant: bool = False) -> Image:
        """
        [renders the board and returns the pillow image]

        Args:
            quant (bool, optional): [convert from RGB to P, slower, smaller files]. Defaults to False.

        Returns:
            Image: [pillow image object]
        """
        image_size = tile_size * self.size
        im = Image.new(
            "RGB",
            (image_size + (tile_outline * 2), image_size + (tile_outline * 2)),
            0x8193A4,
        )
        for x in range(self.size):
            for y in range(self.size):
                im_t = tiles[self.board[x][y]]
                y1, x1 = tile_size * x, tile_size * y
                im.paste(im=im_t, box=(x1 + tile_outline, y1 + tile_outline), mask=im_t)
        if quant:
            im.convert("P", palette=Image.ADAPTIVE)
        return im

    def load(self, data: dict) -> None:
        """
        [load existing board state from dump]

        Args:
            data (dict): [the dump]
        """
        self.__dict__.clear()
        self.__dict__.update(srsly.msgpack_loads(urlsafe_b64decode(data)))

    def dump(self) -> bytes:
        """
        [dump current board state]

        Returns:
            dict: [the dumped board state as bytes]
        """
        return urlsafe_b64encode(srsly.msgpack_dumps(self.__dict__))

    def move(self, action: str, evaluate: bool = False) -> bool:
        """
        [summary]

        Args:
            action (str): [direction to move, (up|down|left|right)]
            evaluate (bool, optional): [just try not proceed]. Defaults to False.

        Returns:
            bool: [if the move succeeded]
        """
        board_copy = deepcopy(self.board)
        rotated_board = np.rot90(board_copy, move_dict[action])
        stack(rotated_board)
        sum_up(rotated_board)
        stack(rotated_board)
        board_copy = np.rot90(rotated_board, len(self.board) - move_dict[action])
        if np.array_equal(self.board, board_copy, equal_nan=False):
            return False
        if not evaluate:
            self.board = board_copy
            self.board = spawn_tile(board=self.board)
            self.calculate_score()
            self.possible_moves = self.update_possible_moves()
        return True

    def update_possible_moves(self) -> dict:
        """
        [evaluates which move directions can succeeed]

        Returns:
            dict: [dict with the results]
        """
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
        self.score = int(self.board.sum())

    def score(self) -> int:
        """
        [returns the current score]

        Returns:
            int: [current score]
        """
        return int(self.score)
