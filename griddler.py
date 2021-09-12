import time
import numpy as np
import math
import random
from itertools import combinations


class Griddler:
    def __init__(self, blx_x, blx_y):
        self.blx_x = blx_x
        self.blx_y = blx_y
        self.n = len(blx_x)
        self.m = len(blx_y)
        self.data = np.zeros([self.n, self.m])
        self.sums = [np.zeros(self.n), np.zeros(self.m)]
        self.max_combination = 10
        self.jobs = {}

    def solve(self):
        if sum(sum(self.blx_x, [])) != sum(sum(self.blx_y, [])):
            raise Exception("Sum mismatch")

        while (self.data == 0).any():
            print(f"Starting. max_combinations={self.max_combination}")
            self.solve_loop()
            # self.print()
            self.max_combination *= 10

    def solve_loop(self):
        self.jobs = self.init_jobs()
        while len(self.jobs) > 0:
            max_key = max(self.jobs, key=self.jobs.get)
            if self.jobs[max_key] <= 0:
                break

            self.jobs[max_key] = 0
            (is_col, idx, use_brute_force) = max_key

            if is_col:
                self.solve_line(self.data[:, idx], self.blx_y[idx], 0, use_brute_force)
            else:
                self.solve_line(self.data[idx, :], self.blx_x[idx], 1, use_brute_force)

    def init_jobs(self):
        jobs = {}
        for i in range(self.n):
            jobs[(0, i, False)] = 100 / self.weight(self.sums[0][i])
            jobs[(0, i, True)] = 1 / self.weight(self.sums[0][i])

        for j in range(self.m):
            jobs[(1, j, False)] = 100 / self.weight(self.sums[1][j])
            jobs[(1, j, True)] = 1 / self.weight(self.sums[1][j])
        return jobs

    @staticmethod
    def weight(s):
        return math.exp(s)

    def print(self):
        for i in range(self.n):
            print(''.join({0: '___', 1: 'XXX', 2: '   '}.get(self.data[i, j]) for j in range(self.m)))
        print()

    def update(self, old_line, new_value, a_idx, n_idx, other_is_row):
        if n_idx < 0 or a_idx + n_idx > len(old_line):
            raise Exception("Invalid")
        if n_idx == 0:
            return
        diffs = list(np.nonzero(old_line[a_idx:a_idx + n_idx] != new_value)[0])
        if len(diffs) > 0:
            old_line[a_idx:a_idx + n_idx] = new_value
            for idx in diffs:
                self.sums[other_is_row][a_idx + idx] += 1
                self.jobs[(other_is_row, a_idx + idx, False)] += 100 / self.weight(self.sums[other_is_row][a_idx + idx])
                self.jobs[(other_is_row, a_idx + idx, True)] += 1 / self.weight(self.sums[other_is_row][a_idx + idx])

    def update_soft(self, old_line, agg_line, new_line, a_idx, n_idx, first_time):
        if np.logical_and(old_line[a_idx:a_idx+n_idx] != 0, old_line[a_idx:a_idx+n_idx] != new_line[:n_idx]).any():
            return True  # still first time

        if first_time:
            agg_line[:] = new_line[:]
        else:
            agg_line[new_line != agg_line] = 0
        return False

    def solve_line(self, line_data, line_blx, other_is_row, use_brute_force):
        grayed = np.nonzero(line_data == 0)
        if len(grayed[0]) == 0:
            return

        (a_idx, a_blx) = self.pfx(line_data, line_blx, other_is_row)
        (z_idx, z_blx) = self.sfx(line_data, line_blx, other_is_row)
        if a_blx >= z_blx:
            if a_idx < z_idx:
                self.update(line_data, 2, a_idx, z_idx - a_idx, other_is_row)
            return
        if a_idx >= z_idx:
            return

        self.freedom_solve(line_data, line_blx, a_idx, z_idx, a_blx, z_blx, other_is_row)
        if use_brute_force:
            self.brute_force(line_data, line_blx, a_idx, z_idx, a_blx, z_blx, other_is_row)

    def pfx(self, line_data, line_blx, other_is_row):
        return self.pfx_internal(line_data, line_blx, 0, 0, other_is_row)

    def sfx(self, line_data, line_blx, other_is_row):
        (z_idx_inv, z_blx_inv) = self.pfx_internal(np.flip(line_data), list(reversed(line_blx)), 0, 0, other_is_row)
        return len(line_data) - z_idx_inv, len(line_blx) - z_blx_inv

    def pfx_internal(self, line_data, line_blx, a_idx, a_blx, other_is_row):
        n_idx = len(line_data) - a_idx
        n_blx = len(line_blx) - a_blx

        if n_idx == 0:  # no line data
            return a_idx, a_blx
    
        if n_blx == 0:  # no more blx
            self.update(line_data, 2, a_idx, n_idx, other_is_row)
            return a_idx+n_idx-1, a_blx+n_blx-1

        current_blx = line_blx[a_blx]

        if line_data[a_idx] == 1:
            self.update(line_data, 1, a_idx, current_blx, other_is_row)
            a_idx += current_blx
            a_blx += 1
            if len(line_data) > a_idx:
                self.update(line_data, 2, a_idx, 1, other_is_row)
                a_idx += 1
            return self.pfx_internal(line_data, line_blx, a_idx, a_blx, other_is_row)
        elif line_data[a_idx] == 2:
            maybe_black = np.nonzero(line_data[a_idx:] != 2)
            if len(maybe_black[0]) > 0:
                a_idx = maybe_black[0][0] + a_idx
                return self.pfx_internal(line_data, line_blx, a_idx, a_blx, other_is_row)
            else:
                raise Exception("Shouldn't get here")
        else:
            blacked = np.nonzero(line_data[a_idx:a_idx + current_blx] == 1)
            if len(blacked[0]) > 0 and 0 <= blacked[0][0] < current_blx:
                first_blacked = blacked[0][0]
                overflow = current_blx - first_blacked
                if overflow > 0:
                    self.update(line_data, 1, a_idx + first_blacked, overflow, other_is_row)
            return a_idx, a_blx

    def freedom_solve(self, line_data, line_blx, a_idx, z_idx, a_blx, z_blx, other_is_row):
        n_idx = z_idx - a_idx
        freedom = n_idx - sum(blx + 1 for blx in line_blx[a_blx:z_blx]) + 1
        if freedom < max(line_blx[a_blx:z_blx]):
            idx = a_idx
            for blx in line_blx[a_blx:z_blx]:
                blacks = blx - freedom
                if blacks > 0:
                    self.update(line_data, 1, idx + freedom, blacks, other_is_row)
                idx = idx + blx + 1

    def brute_force(self, line_data, line_blx, a_idx, z_idx, a_blx, z_blx, other_is_row):
        n_idx = z_idx - a_idx
        n_blx = z_blx - a_blx
        freedom = n_idx - sum(blx + 1 for blx in line_blx[a_blx:z_blx]) + 1
        possibilities = math.comb(freedom + n_blx, n_blx)
        if possibilities > self.max_combination:
            return

        agg_line = np.zeros(n_idx)
        first_time = True
        list_of_combinations = combinations(range(freedom + n_blx), n_blx)
        list_of_combinations = list(list_of_combinations)
        random.shuffle(list_of_combinations)
        for combo in list_of_combinations:
            new_line = np.zeros(n_idx)
            idx = 0
            blx = a_blx
            for i in range(freedom + n_blx):
                if i in combo:
                    closed_blx = np.array([1 for _ in range(line_blx[blx])])
                    if blx < z_blx - 1:
                        closed_blx = np.concatenate((closed_blx, np.array([2])))
                    new_line[idx:idx+len(closed_blx)] = closed_blx
                    idx += len(closed_blx)
                    blx += 1
                else:
                    new_line[idx] = 2
                    idx += 1
            first_time = self.update_soft(line_data, agg_line, new_line, a_idx, n_idx, first_time) and first_time
            if not first_time and not agg_line.any():
                break
        if first_time:
            raise Exception("Dead end")

        return self.update(line_data, agg_line, a_idx, n_idx, other_is_row)


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
