import time
import numpy as np
import math
import random
from itertools import combinations
import copy


class Griddler:
    def __init__(self, blx_x, blx_y):
        self.blx_x = blx_x
        self.blx_y = blx_y
        if sum(sum(self.blx_x, [])) != sum(sum(self.blx_y, [])):
            raise Exception("Sum mismatch")

        self.n = len(blx_x)
        self.m = len(blx_y)
        self.data = np.zeros([self.n, self.m])
        self.max_combination = 10
        self.jobs = {}
        self.bf_jobs = {}
        self.future_bf_jobs = {}
        self.empty_jobs = self.empty_jobs(self.n, self.m)
        self.init_jobs(True)
        self.key = (-1, -1)
        self.use_bf = False

    def print(self):
        for i in range(self.n):
            print(''.join({0: '___', 1: 'XXX', 2: '   '}.get(self.data[i, j]) for j in range(self.m)))
        print()

    @staticmethod
    def empty_jobs(n, m):
        res = {}
        for i in range(n):
            res[(0, i)] = 0

        for j in range(m):
            res[(1, j)] = 0
        return res

    def init_jobs(self, first_time):
        self.bf_jobs = self.future_bf_jobs
        self.future_bf_jobs = copy.deepcopy(self.empty_jobs)

        if first_time:
            for i in range(self.n):
                self.jobs[(0, i)] = 1
                self.future_bf_jobs[(0, i)] = 0
                self.bf_jobs[(0, i)] = 0

            for j in range(self.m):
                self.jobs[(1, j)] = 1
                self.future_bf_jobs[(1, j)] = 0
                self.bf_jobs[(1, j)] = 0

    def solve(self):
        while (self.data == 0).any():
            print(f"Starting. max_combinations={self.max_combination}")
            self.solve_loop()
            # self.print()
            self.max_combination *= 1.5

    def solve_loop(self):
        self.init_jobs(False)
        while True:
            self.use_bf = False
            self.key = max(self.jobs, key=self.jobs.get)
            if self.jobs[self.key] > 0:
                self.jobs[self.key] = 0
            else:
                self.use_bf = True
                self.key = max(self.bf_jobs, key=self.bf_jobs.get)
                if self.bf_jobs[self.key] > 0:
                    self.bf_jobs[self.key] = 0
                else:
                    return

            (is_col, idx) = self.key

            if is_col:
                self.solve_line(self.data[:, idx], self.blx_y[idx])
            else:
                self.solve_line(self.data[idx, :], self.blx_x[idx])

    def solve_line(self, line_data, line_blx):
        if (line_data != 0).all():
            return

        (a_idx, a_blx) = self.pfx(line_data, line_blx)
        (z_idx, z_blx) = self.sfx(line_data, line_blx)
        if a_blx >= z_blx:
            if a_idx < z_idx:
                self.update(line_data, 2, a_idx, z_idx - a_idx)
            return
        if a_idx >= z_idx:
            return

        if not line_data[a_idx:z_idx].any():
            self.use_bf = False

        self.freedom_solve(line_data, line_blx, a_idx, z_idx, a_blx, z_blx)

        if self.use_bf:
            located_blx = self.locate_blx(line_data, line_blx, a_idx, z_idx, a_blx, z_blx)
            if len(located_blx) > 0:
                for idx, blx in located_blx.items():
                    self.bf(line_data, line_blx, a_idx, idx - 1, a_blx, blx)
                    self.bf(line_data, line_blx, idx + line_blx[blx], z_idx, blx + 1, z_blx)
            else:
                self.bf(line_data, line_blx, a_idx, z_idx, a_blx, z_blx)

    def update(self, old_line, new_value, a_idx, n_idx):
        if n_idx < 0 or a_idx + n_idx > len(old_line):
            raise Exception("Invalid")
        if n_idx == 0:
            return
        diffs = list(np.nonzero(old_line[a_idx:a_idx + n_idx] != new_value)[0])
        if len(diffs) > 0:
            old_line[a_idx:a_idx + n_idx] = new_value
            # self.bf_jobs[self.key] += 1
            for idx in diffs:
                self.jobs[(1 - self.key[0], a_idx + idx)] += 1
                self.bf_jobs[(1 - self.key[0], a_idx + idx)] += 1

    def pfx(self, line_data, line_blx):
        return self.pfx_internal(line_data, line_blx, 0, 0)

    def pfx_internal(self, line_data, line_blx, a_idx, a_blx):
        n_idx = len(line_data) - a_idx
        n_blx = len(line_blx) - a_blx

        if n_idx == 0:  # no line data
            return a_idx, a_blx

        if n_blx == 0:  # no more blx
            self.update(line_data, 2, a_idx, n_idx)
            return a_idx+n_idx, a_blx+n_blx

        current_blx = line_blx[a_blx]

        if line_data[a_idx] == 1:
            self.update(line_data, 1, a_idx, current_blx)
            a_idx += current_blx
            a_blx += 1
            if len(line_data) > a_idx:
                self.update(line_data, 2, a_idx, 1)
                a_idx += 1
            return self.pfx_internal(line_data, line_blx, a_idx, a_blx)
        elif line_data[a_idx] == 2:
            maybe_black = np.nonzero(line_data[a_idx:] != 2)
            if len(maybe_black[0]) > 0:
                a_idx = maybe_black[0][0] + a_idx
                return self.pfx_internal(line_data, line_blx, a_idx, a_blx)
            else:
                raise Exception("Shouldn't get here")
        else:
            blacked = np.nonzero(line_data[a_idx:a_idx + current_blx] == 1)
            if len(blacked[0]) > 0 and 0 <= blacked[0][0] < current_blx:
                first_blacked = blacked[0][0]
                overflow = current_blx - first_blacked
                if overflow > 0:
                    self.update(line_data, 1, a_idx + first_blacked, overflow)
            return a_idx, a_blx

    def sfx(self, line_data, line_blx):
        return self.sfx_internal(line_data, line_blx, len(line_data), len(line_blx))

    def sfx_internal(self, line_data, line_blx, z_idx, z_blx):
        n_idx = z_idx
        n_blx = z_blx

        if n_idx == 0:  # no line data
            return z_idx, z_blx

        if n_blx == 0:  # no more blx
            self.update(line_data, 2, z_idx - n_idx, n_idx)
            return z_idx - n_idx, z_blx - n_blx

        current_blx = line_blx[z_blx - 1]

        if line_data[z_idx - 1] == 1:
            self.update(line_data, 1, z_idx - current_blx, current_blx)
            z_idx -= current_blx
            z_blx -= 1
            if 0 < z_idx:
                self.update(line_data, 2, z_idx - 1, 1)
                z_idx -= 1
            return self.sfx_internal(line_data, line_blx, z_idx, z_blx)
        elif line_data[z_idx - 1] == 2:
            maybe_black = np.nonzero(line_data[:z_idx] != 2)
            if len(maybe_black[0]) > 0:
                z_idx = maybe_black[0][-1] + 1
                return self.sfx_internal(line_data, line_blx, z_idx, z_blx)
            else:
                raise Exception("Shouldn't get here")
        else:
            blacked = np.nonzero(line_data[z_idx - current_blx:z_idx] == 1)
            if len(blacked[0]) > 0 and 0 <= blacked[0][-1] < current_blx:
                last_blacked = blacked[0][-1]
                if last_blacked > 0:
                    self.update(line_data, 1, z_idx - current_blx, last_blacked + 1)
            return z_idx, z_blx

    def freedom_solve(self, line_data, line_blx, a_idx, z_idx, a_blx, z_blx):
        n_idx = z_idx - a_idx
        freedom = n_idx - sum(blx + 1 for blx in line_blx[a_blx:z_blx]) + 1
        if freedom < max(line_blx[a_blx:z_blx]):
            idx = a_idx
            for blx in line_blx[a_blx:z_blx]:
                blacks = blx - freedom
                if blacks > 0:
                    self.update(line_data, 1, idx + freedom, blacks)
                idx = idx + blx + 1

    @staticmethod
    def locate_blx(line_data, line_blx, a_idx, z_idx, a_blx, z_blx):
        empty = False
        white = False
        blacks = 0
        suspects = {}
        located_blocks = {}
        for idx in range(a_idx, z_idx):
            if line_data[idx] == 0:
                empty = True
                white = False
                blacks = 0
            elif empty and not white and line_data[idx] == 2:
                empty = False
                white = True
                blacks = 0
            elif white and blacks == 0 and line_data[idx] == 1:
                blacks = 1
                white = False
                empty = False
            elif blacks > 0 and line_data[idx] == 1:
                blacks += 1
                white = False
                empty = False
            elif blacks > 0 and line_data[idx] == 2:
                suspects[idx - blacks] = blacks
                blacks = 0
                empty = False
                white = True

        if len(suspects) > 0:
            for idx, blx in suspects.items():
                if blx in line_blx[a_blx:z_blx]:
                    if line_blx[a_blx:z_blx].count(blx) == 1:
                        blx_idx = line_blx[a_blx:z_blx].index(blx)
                        located_blocks[idx] = a_blx + blx_idx
        return located_blocks

    def bf(self, line_data, line_blx, a_idx, z_idx, a_blx, z_blx):
        n_idx = z_idx - a_idx
        n_blx = z_blx - a_blx
        freedom = n_idx - sum(blx + 1 for blx in line_blx[a_blx:z_blx]) + 1
        possibilities = math.comb(freedom + n_blx, n_blx)
        if possibilities > self.max_combination:
            self.future_bf_jobs[self.key] += 1
            return

        agg_line = np.zeros(n_idx)
        first_time = True
        list_of_combinations = combinations(range(freedom + n_blx), n_blx)
        list_of_combinations = list(list_of_combinations)
        random.shuffle(list_of_combinations)
        for combo in list_of_combinations:
            new_line = np.zeros(n_idx + 1)
            idx = 0
            blx = a_blx
            for i in range(freedom + n_blx):
                if i in combo:
                    blx_len = line_blx[blx]
                    new_line[idx:idx + blx_len].fill(1)
                    idx += blx_len
                    blx += 1
                new_line[idx] = 2
                idx += 1
            new_line = new_line[:-1]
            first_time = self.update_soft(line_data, agg_line, new_line, a_idx, n_idx, first_time) and first_time
            if not first_time and (agg_line == line_data[a_idx:z_idx]).all():
                return
        if first_time:
            raise Exception("Dead end")

        self.update(line_data, agg_line, a_idx, n_idx)

    @staticmethod
    def update_soft(old_line, agg_line, new_line, a_idx, n_idx, first_time):
        if np.logical_and(old_line[a_idx:a_idx + n_idx] != 0, old_line[a_idx:a_idx + n_idx] != new_line).any():
            return True  # still first time

        if first_time:
            agg_line[:] = new_line
        else:
            agg_line[new_line != agg_line] = 0
        return False


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
    t = time.time()
    Griddler(blx_x, blx_y).solve()
    print(time.time() - t)


test()
