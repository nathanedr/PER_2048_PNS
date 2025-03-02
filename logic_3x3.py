import abc
import json
import random
import struct
import sys
import time
import typing
from functools import lru_cache
import numpy as np
from numba import njit
import gzip

N_3x3 = 3
ROW_MASK_3x3 = 0xFFF

row_left_table_3x3 = np.zeros(4096, dtype=np.int64)
row_right_table_3x3 = np.zeros(4096, dtype=np.int64)
score_table_3x3 = np.zeros(4096, dtype=np.float64)

def reverse_row_3x3(row):
    row = row & ROW_MASK_3x3
    return (((row >> 8) & 0xF) << 0) | (row & 0xF0) | ((row & 0xF) << 8)

def init_tables_3x3():
    global row_left_table_3x3, row_right_table_3x3, score_table_3x3
    for row in range(4096):
        line = [(row >> (4 * i)) & 0xF for i in range(3)]
        score = 0.0
        for rank in line:
            if rank >= 2:
                score += (rank - 1) * (1 << rank)
        score_table_3x3[row] = score

        new_line = line.copy()
        i = 0
        while i < (N_3x3 - 1):
            j = i + 1
            while j < N_3x3 and new_line[j] == 0:
                j += 1
            if j == N_3x3:
                break
            if new_line[i] == 0:
                new_line[i] = new_line[j]
                new_line[j] = 0
                i -= 1
            elif new_line[i] == new_line[j]:
                if new_line[i] != 0xF:
                    new_line[i] += 1
                new_line[j] = 0
            i += 1

        result = (new_line[0] << 0) | (new_line[1] << 4) | (new_line[2] << 8)
        rev_result = reverse_row_3x3(result)
        rev_row = reverse_row_3x3(row)
        row_left_table_3x3[row] = row ^ result
        row_right_table_3x3[rev_row] = rev_row ^ rev_result

init_tables_3x3()

@njit
def transpose_3x3(x):
    r0 = x & ROW_MASK_3x3
    r1 = (x >> 12) & ROW_MASK_3x3
    r2 = (x >> 24) & ROW_MASK_3x3
    new0 = (((r2 >> 0) & 0xF) << 8) | (((r1 >> 0) & 0xF) << 4) | ((r0 >> 0) & 0xF)
    new1 = (((r2 >> 4) & 0xF) << 8) | (((r1 >> 4) & 0xF) << 4) | ((r0 >> 4) & 0xF)
    new2 = (((r2 >> 8) & 0xF) << 8) | (((r1 >> 8) & 0xF) << 4) | ((r0 >> 8) & 0xF)
    return new0 | (new1 << 12) | (new2 << 24)

@njit
def execute_move_left_3x3(board, row_left_table):
    ret = board
    ret ^= row_left_table[board & ROW_MASK_3x3] << 0
    ret ^= row_left_table[(board >> 12) & ROW_MASK_3x3] << 12
    ret ^= row_left_table[(board >> 24) & ROW_MASK_3x3] << 24
    return ret

@njit
def execute_move_right_3x3(board, row_right_table):
    ret = board
    ret ^= row_right_table[board & ROW_MASK_3x3] << 0
    ret ^= row_right_table[(board >> 12) & ROW_MASK_3x3] << 12
    ret ^= row_right_table[(board >> 24) & ROW_MASK_3x3] << 24
    return ret

@njit
def execute_move_up_3x3(board, row_left_table):
    t = transpose_3x3(board)
    new_t = execute_move_left_3x3(t, row_left_table)
    return transpose_3x3(new_t)

@njit
def execute_move_down_3x3(board, row_right_table):
    t = transpose_3x3(board)
    new_t = execute_move_right_3x3(t, row_right_table)
    return transpose_3x3(new_t)

@njit
def count_empty_3x3(board):
    count = 0
    for i in range(N_3x3 * N_3x3):
        if ((board >> (4 * i)) & 0xF) == 0:
            count += 1
    return count

def draw_tile_3x3():
    return 1 if random.random() < 0.9 else 2

def insert_tile_rand_3x3(board, tile):
    empties = count_empty_3x3(board)
    if empties == 0:
        return board
    index = random.randint(0, empties - 1)
    tmp = board
    shift = 0
    while True:
        if (tmp & 0xF) == 0:
            if index == 0:
                break
            index -= 1
        tmp >>= 4
        shift += 4
    return board | (tile << shift)

def initial_board_3x3():
    board = draw_tile_3x3() << (4 * random.randint(0, (N_3x3 * N_3x3) - 1))
    return insert_tile_rand_3x3(board, draw_tile_3x3())

def execute_move_3x3(board, move):
    if move == 0:
        return execute_move_up_3x3(board, row_left_table_3x3)
    elif move == 1:
        return execute_move_down_3x3(board, row_right_table_3x3)
    elif move == 2:
        return execute_move_left_3x3(board, row_left_table_3x3)
    elif move == 3:
        return execute_move_right_3x3(board, row_right_table_3x3)
    return board

def get_max_rank_3x3(board):
    max_rank = 0
    b = board
    for _ in range(N_3x3 * N_3x3):
        rank = b & 0xF
        if rank > max_rank:
            max_rank = rank
        b >>= 4
    return max_rank

def score_helper_3x3(board, table):
    return (table[(board >> 0) & ROW_MASK_3x3] +
            table[(board >> 12) & ROW_MASK_3x3] +
            table[(board >> 24) & ROW_MASK_3x3])

def score_board_3x3(board):
    return score_helper_3x3(board, score_table_3x3)

def monotonicity_bonus_3x3(board):
    bonus = 0
    for i in range(N_3x3):
        row = []
        for j in range(N_3x3):
            index = i * N_3x3 + j
            tile = (board >> (4 * index)) & 0xF
            row.append(tile)
        if row == sorted(row) or row == sorted(row, reverse=True):
            bonus += row[-1] - row[0]
    for j in range(N_3x3):
        col = []
        for i in range(N_3x3):
            index = i * N_3x3 + j
            tile = (board >> (4 * index)) & 0xF
            col.append(tile)
        if col == sorted(col) or col == sorted(col, reverse=True):
            bonus += col[-1] - col[0]
    return bonus

def max_tile_in_corner_3x3(board):
    max_tile = get_max_rank_3x3(board)
    corners = [0, 2, 6, 8]
    for pos in corners:
        if ((board >> (4 * pos)) & 0xF) == max_tile:
            return True, max_tile
    return False, 0

def play_game_gui_3x3(board, action, win_rank, already_won=False):
    valid_moves = [move for move in range(4) if execute_move_3x3(board, move) != board]
    if action not in valid_moves:
        return board, 0, False, False, False

    next_board = execute_move_3x3(board, action)
    moved = True

    score = score_board_3x3(next_board) - score_board_3x3(board)
    if get_max_rank_3x3(next_board) >= win_rank and not already_won:
        return next_board, score, True, True, False

    next_board = insert_tile_rand_3x3(next_board, draw_tile_3x3())

    if not any(execute_move_3x3(next_board, move) != next_board for move in range(4)):
        return next_board, 0, moved, False, True

    return next_board, score, moved, False, False

def game_over_3x3(board):
    for move in range(4):
        if execute_move_3x3(board, move) != board:
            return False
    return True

def get_random_tile_insertions_3x3(board):
    outcomes = []
    empty_positions = []
    for pos in range(N_3x3 * N_3x3):
        if ((board >> (4 * pos)) & 0xF) == 0:
            empty_positions.append(pos)
    if not empty_positions:
        outcomes.append((board, 1.0))
        return outcomes

    prob_per_pos = 1.0 / len(empty_positions)
    for pos in empty_positions:
        outcomes.append((board | (1 << (4 * pos)), 0.9 * prob_per_pos))
        outcomes.append((board | (2 << (4 * pos)), 0.1 * prob_per_pos))
    return outcomes

class Board:
    def __init__(self, raw: int = 0):
        self.raw = int(raw)

    def at(self, i: int) -> int:
        return (self.raw >> (i << 2)) & 0x0f

    def set(self, i: int, t: int) -> None:
        self.raw = (self.raw & ~(0x0f << (i << 2))) | ((t & 0x0f) << (i << 2))

    def init(self) -> None:
        self.raw = 0
        self.popup()
        self.popup()

    def popup(self) -> None:
        space = [i for i in range(9) if self.at(i) == 0]
        if space:
            self.set(random.choice(space), 1 if random.random() < 0.9 else 2)

    def move(self, opcode: int) -> int:
        if opcode == 0:
            return self.move_up()
        elif opcode == 1:
            return self.move_right()
        elif opcode == 2:
            return self.move_down()
        elif opcode == 3:
            return self.move_left()
        return -1

    def move_left(self) -> int:
        prev = self.raw
        new_board = execute_move_left_3x3(self.raw, row_left_table_3x3)
        self.raw = new_board
        score = score_board_3x3(new_board) - score_board_3x3(prev) if new_board != prev else -1
        return score

    def move_right(self) -> int:
        prev = self.raw
        new_board = execute_move_right_3x3(self.raw, row_right_table_3x3)
        self.raw = new_board
        score = score_board_3x3(new_board) - score_board_3x3(prev) if new_board != prev else -1
        return score

    def move_up(self) -> int:
        prev = self.raw
        t = transpose_3x3(self.raw)
        new_t = execute_move_left_3x3(t, row_left_table_3x3)
        self.raw = transpose_3x3(new_t)
        score = score_board_3x3(self.raw) - score_board_3x3(prev) if self.raw != prev else -1
        return score

    def move_down(self) -> int:
        prev = self.raw
        t = transpose_3x3(self.raw)
        new_t = execute_move_right_3x3(t, row_right_table_3x3)
        self.raw = transpose_3x3(new_t)
        score = score_board_3x3(self.raw) - score_board_3x3(prev) if self.raw != prev else -1
        return score

    def transpose(self):
        new_raw = 0
        for i in range(3):
            for j in range(3):
                tile = self.at(i * 3 + j)
                new_raw |= (tile << ((j * 3 + i) * 4))
        self.raw = new_raw

    def mirror(self):
        new_raw = 0
        for i in range(3):
            new_raw |= (self.at(i * 3 + 2) << ((i * 3 + 0) * 4)) 
            new_raw |= (self.at(i * 3 + 1) << ((i * 3 + 1) * 4)) 
            new_raw |= (self.at(i * 3 + 0) << ((i * 3 + 2) * 4)) 
        self.raw = new_raw

    def rotate_clockwise(self):
        self.transpose()
        self.mirror()

    def __str__(self) -> str:
        state = '+' + '-' * 18 + '+\n'
        for i in range(0, 9, 3):
            state += ('|' + ''.join(f'{((1 << self.at(j)) & -2):6d}' for j in range(i, i + 3)) + '|\n')
        state += '+' + '-' * 18 + '+'
        return state
    
class Feature(abc.ABC):
    def __init__(self, length: int):
        self.weight = [0.0] * length

    @abc.abstractmethod
    def estimate(self, b: Board) -> float:
        pass

    @abc.abstractmethod
    def update(self, b: Board, u: float) -> float:
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass

class Pattern(Feature):
    def __init__(self, patt: list[int], iso: int = 8):
        super().__init__(1 << (len(patt) * 4))
        self.patt = patt
        self.isom = [None] * iso
        idx = Board(0xfedcba9876543210)
        for i in range(iso):
            temp = Board(idx.raw)
            if i >= 4:
                temp.mirror()
            for _ in range(i % 4):
                temp.rotate_clockwise()
            self.isom[i] = [temp.at(t) for t in patt]

    def estimate(self, b: Board) -> float:
        value = 0
        for iso in self.isom:
            index = sum(b.at(pos) << (4 * i) for i, pos in enumerate(iso))
            value += self.weight[index]
        return value

    def update(self, b: Board, u: float) -> float:
        adjust = u / len(self.isom)
        value = 0
        for iso in self.isom:
            index = sum(b.at(pos) << (4 * i) for i, pos in enumerate(iso))
            self.weight[index] += adjust
            value += self.weight[index]
        return value

    def name(self) -> str:
        patt_str = "".join([f"{p:x}" for p in self.patt])
        return f"{len(self.patt)}-tuple pattern {patt_str}"

    def read(self, input: typing.BinaryIO) -> None:
        size = struct.unpack('I', input.read(4))[0]
        name = input.read(size).decode('utf-8')
        if name != self.name():
            print(f"Error: unexpected feature: {name} ({self.name()} is expected)")
            sys.exit(1)
        size = struct.unpack('Q', input.read(8))[0]
        if size != len(self.weight):
            print(f"Error: unexpected feature size {size} for {self.name()} ({len(self.weight)} is expected)")
            sys.exit(1)
        self.weight = list(struct.unpack(f'{size}f', input.read(size * 4)))
        if len(self.weight) != size:
            print("Error: unexpected end of binary")
            sys.exit(1)

    def write(self, output: typing.BinaryIO) -> None:
        name = self.name().encode('utf-8')
        output.write(struct.pack('I', len(name)))
        output.write(name)
        size = len(self.weight)
        output.write(struct.pack('Q', size))
        output.write(struct.pack(f'{size}f', *self.weight))
        
    def mirror(self):
        new_raw = 0
        for i in range(3):
            for j in range(3):
                tile = self.at(i * 3 + j)
                new_raw |= tile << ((i * 3 + (2 - j)) * 4)
        self.raw = new_raw

    def rotate_clockwise(self):
        self.transpose()
        self.mirror()

    def transpose(self):
        new_raw = 0
        for i in range(3):
            for j in range(3):
                tile = self.at(i * 3 + j)
                new_raw |= tile << ((j * 3 + i) * 4)
        self.raw = new_raw

class Move:
    def __init__(self, board: Board = None, opcode: int = -1):
        self.before = None
        self.after = None
        self.opcode = opcode
        self.score = -1
        self.esti = -float('inf')
        if board is not None:
            self.assign(board)

    def assign(self, b: Board) -> bool:
        self.before = Board(b.raw)
        self.after = Board(b.raw)
        self.score = self.after.move(self.opcode)
        self.esti = self.score if self.score != -1 else -float('inf')
        return self.score != -1

    def is_valid(self) -> bool:
        return self.after.raw != self.before.raw and self.score != -1

    def state(self) -> Board:
        return self.before

    def afterstate(self) -> Board:
        return self.after

    def value(self) -> float:
        return self.esti

    def reward(self) -> int:
        return self.score

    def set_value(self, value: float) -> None:
        self.esti = value

class Learning:
    def __init__(self):
        self.feats = []
        self.scores = []
        self.maxtile = []

    def add_feature(self, feat: Feature) -> None:
        self.feats.append(feat)
        print(f"Added {feat.name()}, size = {len(feat.weight)}")

    def estimate(self, b: Board) -> float:
        return sum(feat.estimate(b) for feat in self.feats)

    def update(self, b: Board, u: float) -> float:
        adjust = u / len(self.feats)
        return sum(feat.update(b, adjust) for feat in self.feats)

    def select_best_move(self, b: Board) -> Move:
        best = Move(b)
        for opcode in range(4):
            mv = Move(b, opcode)
            if mv.is_valid():
                mv.set_value(mv.reward() + self.estimate(mv.afterstate()))
                if mv.value() > best.value():
                    best = mv
        return best

    def learn_from_episode(self, path: list[Move], alpha: float = 0.1) -> None:
        target = 0
        path.pop()
        while path:
            mv = path.pop()
            error = target - self.estimate(mv.afterstate())
            target = mv.reward() + self.update(mv.afterstate(), alpha * error)

    def make_statistic(self, n: int, b: Board, score: int, unit: int = 1000) -> None:
        self.scores.append(score)
        self.maxtile.append(get_max_rank_3x3(b.raw))
        if n % unit == 0:
            avg_score = sum(self.scores) / len(self.scores)
            max_score = max(self.scores)
            print(f"{n}\tavg = {avg_score}\tmax = {max_score}")
            stat = [self.maxtile.count(i) for i in range(16)]
            coef = 100 / unit
            for t in range(1, 16):
                if stat[t]:
                    accu = sum(stat[t:])
                    tile = (1 << t) & -2
                    print(f"\t{tile}\t{accu * coef:.1f}%")
            self.scores.clear()
            self.maxtile.clear()

    def load(self, path: str) -> None:
        try:
            with open(path, 'rb') as input:
                size = struct.unpack('Q', input.read(8))[0]
                if size != len(self.feats):
                    print(f"Error: unexpected feature count: {size} ({len(self.feats)} is expected)")
                    sys.exit(1)
                for feat in self.feats:
                    feat.read(input)
                    print(f"{feat.name()} is loaded from {path}")
        except FileNotFoundError:
            pass

    def save(self, path: str) -> None:
        """
        Sauvegarde la table de poids dans un fichier binaire.
        """
        try:
            with open(path, 'wb') as output:
                output.write(struct.pack('Q', len(self.feats)))
                for feat in self.feats:
                    feat.write(output)
                    print(f"{feat.name()} is saved to {path}")
        except FileNotFoundError:
            pass

    def load_gzip(self, path: str) -> None:
        try:
            with gzip.open(path, 'rb') as input:
                size = struct.unpack('Q', input.read(8))[0]
                if size != len(self.feats):
                    print(f"Error: unexpected feature count: {size} ({len(self.feats)} is expected)")
                    sys.exit(1)
                for feat in self.feats:
                    feat.read(input)
                    print(f"{feat.name()} is loaded from {path}")
        except FileNotFoundError:
            pass

    def save_gzip(self, path: str) -> None:
        try:
            with gzip.open(path, 'wb') as output:
                output.write(struct.pack('Q', len(self.feats)))
                for feat in self.feats:
                    feat.write(output)
                    print(f"{feat.name()} is saved to {path}")
        except FileNotFoundError:
            pass