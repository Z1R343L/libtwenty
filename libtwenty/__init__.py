"""
[2048 lib]

"""
import secrets
from copy import deepcopy
from os.path import abspath
from pathlib import Path
from typing import Optional, Literal, Union

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
    for i in range(0, len(board)):
        for j in range(0, len(board)):
            k = i
            while board[k][j] == 0:
                if k == len(board) - 1:
                    break
                k += 1
            if k != i:
                board[i][j], board[k][j] = board[k][j], 0


def sum_up(board) -> None:
    for i in range(0, len(board) - 1):
        for j in range(0, len(board)):
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
    def __init__(
        self,
        size: int = 4,
        state_string: Optional[str] = None
    ) -> None:
        """
        [2048 board]
        Args:
            size (int, optional): [board size]. Defaults to 4.
        """
        self.board = np.zeros((size, size), int)
        if state_string:
            self.from_state_string(state_string=state_string)
        self.board = spawn_tile(board=self.board)
        self.board = spawn_tile(board=self.board)
        self.score = self.board.sum()
        self.update_possible_moves()
        self.size = size

    def board_string(self) -> str:
        return str(self.board)

    def to_state_string(self) -> str:
        return '_'.join([str(t_range[i]) for i in np.nditer(self.board)])
    
    def from_state_string(self, state_string: str) -> None:
        row, current_row = [0] * 2
        for i in state_string.split('_'):
            self.board[row][current_row] = t_range[int(i)]
            if current_row == 3:
                current_row = 0
                row += 1
            else:
                current_row += 1



    def render(self) -> Image:

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
        return im


    def move(
        self, 
        action: Union[int, Literal[list(move_dict.keys())]],
        evaluate: bool = False
    ) -> bool:
        board_copy = deepcopy(self.board)
        rotated_board = np.rot90(board_copy, move_dict[action])
        stack(rotated_board)
        sum_up(rotated_board)
        stack(rotated_board)
        board_copy = np.rot90(rotated_board, 4 - move_dict[action])
        if np.array_equal(self.board, board_copy, equal_nan=False):
            return False
        if not evaluate:
            self.board = board_copy
            self.board = spawn_tile(board=self.board)
            self.calculate_score()
            self.possible_moves = self.update_possible_moves()
        return True

    def update_possible_moves(self) -> None:
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
        self.possible_moves = res

    def calculate_score(self) -> None:
        self.score = int(self.board.sum())

