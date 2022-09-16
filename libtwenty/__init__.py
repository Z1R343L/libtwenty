from __future__ import annotations

import itertools
import secrets
from copy import deepcopy
from os.path import abspath
from pathlib import Path
from typing import Optional, Union
from io import BytesIO

import numpy as np
from numpy.random import choice
from PIL import Image, ImageDraw, ImageFont

move_dict = {"up": 0, "right": 1, "down": 2, "left": 3}

assets_path = Path(abspath(__file__)).parent / "assets"

class Colors:
    def __init__(self) -> None:
        self.t_colors = {
            0: 0xFFA0B0BD,
            2: 0xFFDAE4EE,
            4: 0xFFCEE1EE,
            8: 0xFF7EB2F4,
            16: 0xFF6B97F6,
            32: 0xFF697EF7,
            64: 0xFF4861F7,
            128: 0xFF90D6EE,
            256: 0xFF84D3EE,
            512: 0xFF78D1EC,
            1024: 0xFF3FC5ED,
            2048: 0xFF2DC2ED
        }
    def gereedy_get(self, k: int = 0) -> int:
        if k not in self.t_colors:
            k = -1
        return self.t_colors[k]

colors = Colors()

class Tiles:
    def __init__(self, colors: Colors) -> None:
        self.t_range = list(colors.t_colors.keys())
        self.t_cache, self.f_cache = [{}] * 2
        
    def font_color(self, tile: int) -> int:
        return 0xFF656E77 if tile in {2, 4} else 0xFFF1F6F8
        
    def prep_font(self, font_size: int):
        font = ImageFont.truetype(str(assets_path / "AGAALER.TTF"), font_size, encoding="unic")
        self.f_cache[font_size] = font
        return font
        
    def build_tile(self, tile_size: int, t: int) -> Image.Image:
        t_im = Image.new("RGBA", (tile_size, tile_size), color=0x00000000)
        t_id = ImageDraw.Draw(t_im)
        t_id.rounded_rectangle(
            xy=[(0, 0), (tile_size, tile_size)],
            fill=Colors.gereedy_get(k=t),
            outline=0x00000000,
            width=tile_outline,
            radius=tile_radius,
        )
        if t != 0:
            tw, th = font.getsize(str(t))
            xt, yt = ((tile_size - tw) / 2), ((tile_size - th) / 2)
            t_id.text(xy=(xt, yt), text=str(t), font=font, fill=font_color(t))
        return t_im

    def prep_tiles(self, tile_size: int = 200, tile_outline: int = 6) -> dict:
        font_size = int((52 / 200) * tile_size)
        font = self.f_cache.get(font_size) or prep_font(font_size=font_size)
        tile_radius = tile_size / 10
        tiles = {build_tile(tile_size, t) for t in t_range}
        self.t_cache[tile_size] = tiles
        return tiles

def stack(board) -> None:
    for i, j in itertools.product(range(len(board)), range(len(board))):
        k = i
        while board[k][j] == 0 and k != len(board) - 1:
            k += 1
        if k != i:
            board[i][j], board[k][j] = board[k][j], 0

def sum_up(board) -> None:
    for i, j in itertools.product(range(len(board) - 1), range(len(board))):
        if board[i][j] != 0 and board[i][j] == board[i + 1][j]:
            board[i][j] += board[i + 1][j]
            board[i + 1][j] = 0

def spawn_tile(board) -> np.ndarray:
    zeroes_flatten = np.where(board == 0)
    zeroes_indices = list(zip(zeroes_flatten[0], zeroes_flatten[1]))
    random_index = zeroes_indices[choice(len(zeroes_indices), 1)[0]]
    board[random_index] = secrets.choice([2, 2, 4])
    return board

class Board:
    def __init__(
        self,
        size: int = 4,
        tile_size: int = 200,
        state_string: Optional[str] = None
    ) -> None:
        self.tile_size = tile_size
        self.score, self.possible_moves = [None] * 2
        self.size = size
        if state_string:
            self.from_state_string(state_string=state_string)
        else:
            self.board = np.zeros((self.size, self.size), int)
            self.board = spawn_tile(board=self.board)
            self.board = spawn_tile(board=self.board)
        self.calculate_score()
        self.update_possible_moves()

    def __str__(self) -> str:
        return str(self.board)

    def to_state_string(self) -> str:
        return ''.join(f'{t_range.index(i):02d}' for i in np.nditer(self.board))

    def from_state_string(self, state_string: str) -> None:
        self.board = np.reshape([t_range[int(state_string[i : i + 2])] for i in range(0, 32, 2)], (4, 4))

    def render(self, bytesio: bool = False) -> Union[Image.Image, BytesIO]:
        image_size = self.tile_size * self.size
        tile_outline = int((6 / 200) * self.tile_size)
        tiles = self.t_cache.get(self.tile_size) or prep_tiles(tile_size=self.tile_size, tile_outline=tile_outline)
        im = Image.new(
            "RGB",
            (image_size + (tile_outline * 2), image_size + (tile_outline * 2)),
            0x8193A4,
        )
        for x, y in itertools.product(range(self.size), range(self.size)):
            im_t = tiles[self.board[x][y]]
            y1, x1 = self.tile_size * x, self.tile_size * y
            im.paste(im=im_t, box=(x1 + tile_outline, y1 + tile_outline), mask=im_t)
        if bytesio:
            buffer = BytesIO()
            im.save(buffer, 'PNG')
            buffer.seek(0)
            return buffer
        return im


    def move(
        self, 
        action: Union[int, str],
        evaluate: bool = False
    ) -> Union[bool, 'Board']:
        if isinstance(action, str):
            action = move_dict[action]
        board_copy = deepcopy(self.board)
        rotated_board = np.rot90(board_copy, action)
        stack(rotated_board)
        sum_up(rotated_board)
        stack(rotated_board)
        board_copy = np.rot90(rotated_board, 4 - action)
        if np.array_equal(self.board, board_copy, equal_nan=False):
            return False
        if not evaluate:
            self.board = board_copy
            self.board = spawn_tile(board=self.board)
            self.calculate_score()
            self.update_possible_moves()
            return self
        return True

    def update_possible_moves(self) -> None:
        res, n, over = {}, 0, False
        for direction in list(move_dict.values()):
            res[direction] = self.move(action=direction, evaluate=True)
            if not res[direction]:
                n += 1
        if n == 4:
            over = True
        res["over"] = over
        self.possible_moves = res

    def calculate_score(self) -> None:
        self.score = int(self.board.sum())

