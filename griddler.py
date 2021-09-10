import numpy as np
import math
from itertools import combinations
from enum import Enum


class State(Enum):
    GRAY = 0
    BLACK = 1
    WHITE = 2


class Dimension(Enum):
    ROW = 0
    COL = 1


PRINT_DICT = {State.GRAY.value: '___', State.BLACK.value: 'XXX', State.WHITE.value: '   '}


class Griddler:
    def __init__(self, blx_x, blx_y):
        self.blx = {Dimension.ROW: blx_x, Dimension.COL: blx_y}
        self.n = len(blx_x)
        self.m = len(blx_y)
        self.data = np.array([[State.GRAY.value for _ in range(self.m)] for _ in range(self.n)])
        self.max_combination = 1000
        self.jobs = {}

    def solve(self):
        if sum(sum(self.blx[Dimension.ROW], [])) != sum(sum(self.blx[Dimension.COL], [])):
            raise Exception("Sum mismatch")

        while (self.data == State.GRAY.value).any():
            print(f"Starting. max_combinations={self.max_combination}")
            self.jobs = self.init_jobs()

            while len(self.jobs) > 0:
                max_value = max(self.jobs.values())
                if max_value <= 0:
                    break
                max_key = [k for k, v in self.jobs.items() if v == max_value][0]
                self.jobs[max_key] = 0
                (dim, idx) = max_key

                if dim == Dimension.ROW:
                    self.solve_line(self.data[idx, :], self.blx[Dimension.ROW][idx], Dimension.COL)
                else:
                    self.solve_line(self.data[:, idx], self.blx[Dimension.COL][idx], Dimension.ROW)

            self.print()
            self.max_combination *= 2

    def init_jobs(self):
        jobs = {}
        for i in range(self.n):
            jobs[(Dimension.ROW, i)] = 1
        for j in range(self.m):
            jobs[(Dimension.COL, j)] = 1
        return jobs

    def print(self):
        for i in range(self.n):
            print(''.join(PRINT_DICT.get(self.data[i, j]) for j in range(self.m)))
        print()

    def update(self, old_line, new_value, a_idx, n_idx, dim):
        if n_idx < 0 or a_idx + n_idx > len(old_line):
            raise Exception("Invalid")
        if n_idx == 0:
            return
        diffs = list(np.nonzero(old_line[a_idx:a_idx + n_idx] != new_value)[0])
        if len(diffs) > 0:
            old_line[a_idx:a_idx + n_idx] = new_value
            for idx in diffs:
                self.jobs[(dim, a_idx + idx)] += 1

    def update_soft(self, old_line, agg_line, new_line, a_idx, n_idx, first_time):
        for idx in range(n_idx):
            if old_line[a_idx+idx] != State.GRAY.value and old_line[a_idx+idx] != new_line[idx]:
                return True  # still first time

        for idx in range(n_idx):
            if first_time:
                agg_line[idx] = new_line[idx]
            if new_line[idx] != agg_line[idx]:
                agg_line[idx] = State.GRAY.value
        return False

    def solve_line(self, line_data, line_blx, dim):
        grayed = np.nonzero(line_data == State.GRAY.value)
        if len(grayed[0]) == 0:
            return

        (a_idx, a_blx) = self.pfx(line_data, line_blx, dim)
        (z_idx, z_blx) = self.sfx(line_data, line_blx, dim)
        if a_blx >= z_blx:
            if a_idx < z_idx:
                self.update(line_data, State.WHITE.value, a_idx, z_idx - a_idx, dim)
            return
        if a_idx >= z_idx:
            return

        self.freedom_solve(line_data, line_blx, a_idx, z_idx, a_blx, z_blx, dim)
        self.brute_force(line_data, line_blx, a_idx, z_idx, a_blx, z_blx, dim)

    def pfx(self, line_data, line_blx, dim):
        return self.pfx_internal(line_data, line_blx, 0, 0, dim)

    def sfx(self, line_data, line_blx, dim):
        (z_idx_inv, z_blx_inv) = self.pfx_internal(np.flip(line_data), list(reversed(line_blx)), 0, 0, dim)
        return len(line_data) - z_idx_inv, len(line_blx) - z_blx_inv

    def pfx_internal(self, line_data, line_blx, a_idx, a_blx, dim):
        n_idx = len(line_data) - a_idx
        n_blx = len(line_blx) - a_blx

        if n_idx == 0:  # no line data
            return a_idx, a_blx
    
        if n_blx == 0:  # no more blx
            self.update(line_data, State.WHITE.value, a_idx, n_idx, dim)
            return a_idx+n_idx-1, a_blx+n_blx-1

        current_blx = line_blx[a_blx]

        if line_data[a_idx] == State.BLACK.value:
            self.update(line_data, State.BLACK.value, a_idx, current_blx, dim)
            a_idx += current_blx
            a_blx += 1
            if len(line_data) > a_idx:
                self.update(line_data, State.WHITE.value, a_idx, 1, dim)
                a_idx += 1
            return self.pfx_internal(line_data, line_blx, a_idx, a_blx, dim)
        elif line_data[a_idx] == State.WHITE.value:
            maybe_black = np.nonzero(line_data[a_idx:] != State.WHITE.value)
            if len(maybe_black[0]) > 0:
                a_idx = maybe_black[0][0] + a_idx
                return self.pfx_internal(line_data, line_blx, a_idx, a_blx, dim)
            else:
                raise Exception("Shouldn't get here")
        else:
            blacked = np.nonzero(line_data[a_idx:a_idx + current_blx] == State.BLACK.value)
            if len(blacked[0]) > 0 and 0 <= blacked[0][0] < current_blx:
                first_blacked = blacked[0][0]
                overflow = current_blx - first_blacked
                if overflow > 0:
                    self.update(line_data, State.BLACK.value, a_idx + first_blacked, overflow, dim)
            return a_idx, a_blx

    def freedom_solve(self, line_data, line_blx, a_idx, z_idx, a_blx, z_blx, dim):
        n_idx = z_idx - a_idx
        freedom = n_idx - sum(blx + 1 for blx in line_blx[a_blx:z_blx]) + 1
        if freedom < max(line_blx[a_blx:z_blx]):
            idx = a_idx
            for blx in line_blx[a_blx:z_blx]:
                blacks = blx - freedom
                if blacks > 0:
                    self.update(line_data, State.BLACK.value, idx + freedom, blacks, dim)
                idx = idx + blx + 1

    def brute_force(self, line_data, line_blx, a_idx, z_idx, a_blx, z_blx, dim):
        n_idx = z_idx - a_idx
        n_blx = z_blx - a_blx
        freedom = n_idx - sum(blx + 1 for blx in line_blx[a_blx:z_blx]) + 1
        possibilities = math.comb(freedom + n_blx, n_blx)
        if possibilities > self.max_combination:
            return

        agg_line = np.zeros(n_idx)
        first_time = True
        for combo in combinations(range(freedom + n_blx), n_blx):
            new_line = np.zeros(n_idx)
            idx = 0
            blx = a_blx
            for i in range(freedom + n_blx):
                if i in combo:
                    closed_blx = np.array([State.BLACK.value for _ in range(line_blx[blx])])
                    if blx < z_blx - 1:
                        closed_blx = np.concatenate((closed_blx, np.array([State.WHITE.value])))
                    new_line[idx:idx+len(closed_blx)] = closed_blx
                    idx += len(closed_blx)
                    blx += 1
                else:
                    new_line[idx] = State.WHITE.value
                    idx += 1
            first_time = self.update_soft(line_data, agg_line, new_line, a_idx, n_idx, first_time) and first_time
        if first_time:
            raise Exception("Dead end")

        return self.update(line_data, agg_line, a_idx, n_idx, dim)


def test():
    blx_x = [[1, 1], [1, 16], [19], [9], [2, 4],
             [3, 10], [13], [16], [1, 19], [24],
             [2, 3, 3, 8], [3, 3, 3, 1, 7], [1, 4, 2, 2, 7], [1, 3, 7], [1, 1, 1, 7],
             [4, 1, 7], [1, 8], [1, 2, 7], [2, 6], [3, 1, 5],
             [1, 1, 5], [2, 11], [5, 6, 1], [8, 1], [3, 1],
             [3, 1], [4, 5], [4, 2, 2], [3, 3, 1], [4, 2]]
    blx_y = [[1, 1], [1, 1, 3, 3], [2, 1, 1, 1, 5], [1, 1, 1, 1, 6], [1, 3, 4, 3, 5],
             [4, 4, 6, 4], [1, 3, 2, 1, 1, 1, 3], [3, 2, 2, 1, 1, 1], [3, 5, 1], [3, 5, 1],
             [3, 4, 1], [3, 3, 2, 1], [3, 7, 1], [2, 6, 2], [2, 5, 3, 1, 4],
             [2, 5, 2, 1, 1, 1], [2, 5, 4, 1, 1, 2], [2, 6, 3, 1, 1], [2, 17, 2], [2, 17, 2],
             [2, 17, 1], [2, 17, 1], [2, 17, 1], [3, 14, 7], [1, 10, 2]]
    Griddler(blx_x, blx_y).solve()


test()
