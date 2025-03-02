import abc
import random
import struct
import sys
import typing
import numpy as np
from numba import njit
import gzip

N_4x4 = 4
ROW_MASK_4x4 = 0xFFFF

row_left_table_4x4 = np.zeros(65536, dtype=np.int64)
row_right_table_4x4 = np.zeros(65536, dtype=np.int64)
score_table_4x4 = np.zeros(65536, dtype=np.float64)

def reverse_row_4x4(row):
    row = row & ROW_MASK_4x4
    return ((row & 0xF) << 12) | ((row & 0xF0) << 4) | ((row & 0xF00) >> 4) | ((row & 0xF000) >> 12)

def init_tables_4x4():
    global row_left_table_4x4, row_right_table_4x4, score_table_4x4
    for row in range(65536):
        line = [(row >> (4 * i)) & 0xF for i in range(N_4x4)]
        score = 0.0
        for rank in line:
            if rank >= 2:
                score += (rank - 1) * (1 << rank)
        score_table_4x4[row] = score
        new_line = line.copy()
        i = 0
        while i < (N_4x4 - 1):
            j = i + 1
            while j < N_4x4 and new_line[j] == 0:
                j += 1
            if j == N_4x4:
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
        result = (new_line[0] << 0) | (new_line[1] << 4) | (new_line[2] << 8) | (new_line[3] << 12)
        rev_result = reverse_row_4x4(result)
        rev_row = reverse_row_4x4(row)
        row_left_table_4x4[row] = row ^ result
        row_right_table_4x4[rev_row] = rev_row ^ rev_result

init_tables_4x4()

@njit
def transpose_4x4(x):
    r0 = x & ROW_MASK_4x4
    r1 = (x >> 16) & ROW_MASK_4x4
    r2 = (x >> 32) & ROW_MASK_4x4
    r3 = (x >> 48) & ROW_MASK_4x4
    new0 = ((r0 & 0xF)      ) | (((r1 & 0xF)      ) << 4) | (((r2 & 0xF)      ) << 8) | (((r3 & 0xF)      ) << 12)
    new1 = (((r0 >> 4) & 0xF)) | (((r1 >> 4) & 0xF) << 4) | (((r2 >> 4) & 0xF) << 8) | (((r3 >> 4) & 0xF) << 12)
    new2 = (((r0 >> 8) & 0xF)) | (((r1 >> 8) & 0xF) << 4) | (((r2 >> 8) & 0xF) << 8) | (((r3 >> 8) & 0xF) << 12)
    new3 = (((r0 >> 12) & 0xF))| (((r1 >> 12) & 0xF)<< 4) | (((r2 >> 12) & 0xF)<< 8) | (((r3 >> 12) & 0xF)<< 12)
    return new0 | (new1 << 16) | (new2 << 32) | (new3 << 48)

@njit
def execute_move_left_4x4(board, row_left_table):
    ret = board
    ret ^= row_left_table[board & ROW_MASK_4x4] << 0
    ret ^= row_left_table[(board >> 16) & ROW_MASK_4x4] << 16
    ret ^= row_left_table[(board >> 32) & ROW_MASK_4x4] << 32
    ret ^= row_left_table[(board >> 48) & ROW_MASK_4x4] << 48
    return ret

@njit
def execute_move_right_4x4(board, row_right_table):
    ret = board
    ret ^= row_right_table[board & ROW_MASK_4x4] << 0
    ret ^= row_right_table[(board >> 16) & ROW_MASK_4x4] << 16
    ret ^= row_right_table[(board >> 32) & ROW_MASK_4x4] << 32
    ret ^= row_right_table[(board >> 48) & ROW_MASK_4x4] << 48
    return ret

@njit
def count_empty_4x4(board):
    count = 0
    for i in range(N_4x4 * N_4x4):
        if ((board >> (4 * i)) & 0xF) == 0:
            count += 1
    return count

def score_board_4x4(board):
    return (score_table_4x4[(board >> 0) & ROW_MASK_4x4] +
            score_table_4x4[(board >> 16) & ROW_MASK_4x4] +
            score_table_4x4[(board >> 32) & ROW_MASK_4x4] +
            score_table_4x4[(board >> 48) & ROW_MASK_4x4])

def get_max_rank_4x4(board):
    max_rank = 0
    for i in range(N_4x4 * N_4x4):
        rank = (board >> (4 * i)) & 0xF
        if rank > max_rank:
            max_rank = rank
    return max_rank

class Board_4x4:
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
        space = [i for i in range(N_4x4 * N_4x4) if self.at(i) == 0]
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
        new_board = execute_move_left_4x4(self.raw, row_left_table_4x4)
        self.raw = new_board
        score = score_board_4x4(new_board) - score_board_4x4(prev) if new_board != prev else -1
        return score

    def move_right(self) -> int:
        prev = self.raw
        new_board = execute_move_right_4x4(self.raw, row_right_table_4x4)
        self.raw = new_board
        score = score_board_4x4(new_board) - score_board_4x4(prev) if new_board != prev else -1
        return score

    def move_up(self) -> int:
        prev = self.raw
        t = transpose_4x4(self.raw)
        new_t = execute_move_left_4x4(t, row_left_table_4x4)
        self.raw = transpose_4x4(new_t)
        score = score_board_4x4(self.raw) - score_board_4x4(prev) if self.raw != prev else -1
        return score

    def move_down(self) -> int:
        prev = self.raw
        t = transpose_4x4(self.raw)
        new_t = execute_move_right_4x4(t, row_right_table_4x4)
        self.raw = transpose_4x4(new_t)
        score = score_board_4x4(self.raw) - score_board_4x4(prev) if self.raw != prev else -1
        return score

    def transpose(self):
        new_raw = 0
        for i in range(N_4x4):
            for j in range(N_4x4):
                tile = self.at(i * N_4x4 + j)
                new_raw |= (tile << ((j * N_4x4 + i) * 4))
        self.raw = new_raw

    def mirror(self):
        new_raw = 0
        for i in range(N_4x4):
            for j in range(N_4x4):
                new_raw |= self.at(i * N_4x4 + (N_4x4 - 1 - j)) << ((i * N_4x4 + j) * 4)
        self.raw = new_raw

    def rotate_clockwise(self):
        self.transpose()
        self.mirror()

    def __str__(self) -> str:
        state = '+' + '-' * (6 * N_4x4) + '+\n'
        for i in range(0, N_4x4 * N_4x4, N_4x4):
            row_str = ''.join(f'{(0 if self.at(j)==0 else (1 << self.at(j))):6d}' for j in range(i, i + N_4x4))
            state += '|' + row_str + '|\n'
        state += '+' + '-' * (6 * N_4x4) + '+'
        return state

class Feature_4x4(abc.ABC):
    def __init__(self, length: int):
        self.weight = [0.0] * length

    @abc.abstractmethod
    def estimate(self, b: Board_4x4) -> float:
        pass

    @abc.abstractmethod
    def update(self, b: Board_4x4, u: float) -> float:
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass

class Pattern_4x4(Feature_4x4):
    def __init__(self, patt: list[int], iso: int = 8):
        super().__init__(1 << (len(patt) * 4))
        self.patt = patt
        self.isom = [None] * iso
        idx = Board_4x4(0xFEDCBA9876543210)
        for i in range(iso):
            temp = Board_4x4(idx.raw)
            if i >= 4:
                temp.mirror()
            for _ in range(i % 4):
                temp.rotate_clockwise()
            self.isom[i] = [temp.at(t) for t in patt]

    def estimate(self, b: Board_4x4) -> float:
        value = 0
        for iso in self.isom:
            index = sum(b.at(pos) << (4 * i) for i, pos in enumerate(iso))
            value += self.weight[index]
        return value

    def update(self, b: Board_4x4, u: float) -> float:
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
        
class Move_4x4:
    def __init__(self, board: Board_4x4 = None, opcode: int = -1):
        self.before = None
        self.after = None
        self.opcode = opcode
        self.score = -1
        self.esti = -float('inf')
        if board is not None:
            self.assign(board)

    def assign(self, b: Board_4x4) -> bool:
        self.before = Board_4x4(b.raw)
        self.after = Board_4x4(b.raw)
        self.score = self.after.move(self.opcode)
        self.esti = self.score if self.score != -1 else -float('inf')
        return self.score != -1

    def is_valid(self) -> bool:
        return self.after.raw != self.before.raw and self.score != -1

    def state(self) -> Board_4x4:
        return self.before

    def afterstate(self) -> Board_4x4:
        return self.after

    def value(self) -> float:
        return self.esti

    def reward(self) -> int:
        return self.score

    def set_value(self, value: float) -> None:
        self.esti = value

class Learning_4x4:
    def __init__(self):
        self.feats = []
        self.scores = []
        self.maxtile = []

    def add_feature(self, feat: Feature_4x4) -> None:
        self.feats.append(feat)
        print(f"Added {feat.name()}, size = {len(feat.weight)}")

    def estimate(self, b: Board_4x4) -> float:
        return sum(feat.estimate(b) for feat in self.feats)

    def update(self, b: Board_4x4, u: float) -> float:
        adjust = u / len(self.feats)
        return sum(feat.update(b, adjust) for feat in self.feats)

    def select_best_move(self, b: Board_4x4) -> Move_4x4:
        best = Move_4x4(b)
        for opcode in range(4):
            mv = Move_4x4(b, opcode)
            if mv.is_valid():
                mv.set_value(mv.reward() + self.estimate(mv.afterstate()))
                if mv.value() > best.value():
                    best = mv
        return best

    def learn_from_episode(self, path: list, alpha: float = 0.1) -> None:
        target = 0
        path.pop()
        while path:
            mv = path.pop()
            error = target - self.estimate(mv.afterstate())
            target = mv.reward() + self.update(mv.afterstate(), alpha * error)

    def make_statistic(self, n: int, b: Board_4x4, score: int, unit: int = 1000) -> None:
        self.scores.append(score)
        self.maxtile.append(get_max_rank_4x4(b.raw))
        if n % unit == 0:
            avg_score = sum(self.scores) / len(self.scores)
            max_score = max(self.scores)
            print(f"{n}\tavg = {avg_score}\tmax = {max_score}")
            stat = [self.maxtile.count(i) for i in range(16)]
            coef = 100 / unit
            for t in range(1, 16):
                if stat[t]:
                    accu = sum(stat[t:])
                    tile = (1 << t) if t > 0 else 0
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