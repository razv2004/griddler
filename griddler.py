import time
import numpy as np
import math
import random
from itertools import combinations


class Griddler:
    def __init__(self, blx_x, blx_y):
        self.blx_x = blx_x
        self.blx_y = blx_y
        if sum(sum(self.blx_x, [])) != sum(sum(self.blx_y, [])):
            raise Exception(f"Sum mismatch sum_x={sum(sum(self.blx_x, []))} sum_y={sum(sum(self.blx_y, []))}")

        self.n = len(blx_x)
        self.m = len(blx_y)
        self.data = np.zeros([self.n, self.m])
        self.jobs = self.init_jobs()
        self.bf_jobs = {}
        self.inspect_jobs = {}
        self.key = (-1, -1)
        self.updated = 0
        self.bf_level = 0

    def print(self):
        for i in range(self.n):
            # print(''.join({0: '___', 1: 'XXX', 2: '   '}.get(self.data[i, j]) for j in range(self.m)))
            print(''.join({0: '_', 1: 'X', 2: ' '}.get(self.data[i, j]) for j in range(self.m)))
        print()

    def init_jobs(self):
        jobs = {}
        for i in range(self.n):
            jobs[(0, i)] = 1
        for j in range(self.m):
            jobs[(1, j)] = 1
        return jobs

    def solve(self):
        while True:
            self.updated = 0
            start_time = time.time()
            self.bf_level = 0
            if len(self.jobs) > 0:
                self.key = max(self.jobs, key=self.jobs.get)
                del self.jobs[self.key]
            else:
                self.bf_level = 1
                if len(self.inspect_jobs) > 0:
                    self.key = max(self.inspect_jobs, key=self.inspect_jobs.get)
                    del self.inspect_jobs[self.key]
                else:
                    self.bf_level = 2
                    if len(self.bf_jobs) > 0:
                        self.key = min(self.bf_jobs, key=self.bf_jobs.get)
                        del self.bf_jobs[self.key]
                    else:
                        self.print()
                        return

            (is_col, idx) = self.key

            if is_col:
                self.solve_line(self.data[:, idx], self.blx_y[idx])
            else:
                self.solve_line(self.data[idx, :], self.blx_x[idx])

            time_elapsed = time.time() - start_time
            if self.updated > 0 or time_elapsed > 10:
                print(f"key={self.key}, level={self.bf_level}, time={time_elapsed}, grayed={np.sum(self.data == 0)}")
                print(f"jobs={self.jobs}, inspect={self.inspect_jobs}, bf={self.bf_jobs}")
                self.print()

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
            self.bf_level = 0

        self.freedom_solve(line_data, line_blx, a_idx, z_idx, a_blx, z_blx)

        if self.bf_level > 0:
            located_blx = self.locate_blx(line_data, line_blx, a_idx, z_idx, a_blx, z_blx)
            if len(located_blx) > 0:
                for idx, blx in located_blx.items():
                    self.bf(line_data, line_blx, a_idx, idx - 1, a_blx, blx)
                    self.bf(line_data, line_blx, idx + line_blx[blx], z_idx, blx + 1, z_blx)
            else:
                self.bf(line_data, line_blx, a_idx, z_idx, a_blx, z_blx)

    def update(self, old_line, new_value, a_idx, n_idx):
        if n_idx < 0 or a_idx + n_idx > len(old_line):
            self.print()
            raise Exception(f"Invalid. key={self.key}, old_line={old_line}, value={new_value}, a={a_idx}, n={n_idx}")
        if n_idx == 0:
            return
        diffs = list(np.nonzero(old_line[a_idx:a_idx + n_idx] != new_value)[0])
        if len(diffs) == 0:
            return

        self.updated += len(diffs)
        old_line[a_idx:a_idx + n_idx] = new_value
        # self.bf_jobs[self.key] += 1
        for idx in diffs:
            other_key = (1 - self.key[0], a_idx + idx)
            if other_key in self.jobs.keys():
                self.jobs[other_key] += 1
            else:
                self.jobs[other_key] = 1
            if other_key in self.inspect_jobs.keys():
                self.inspect_jobs[other_key] += 1
            else:
                self.inspect_jobs[other_key] = 1

    def pfx(self, line_data, line_blx):
        return self.pfx_internal(line_data, line_blx, 0, 0)

    def pfx_internal(self, line_data, line_blx, a_idx, a_blx):
        n_idx = len(line_data) - a_idx
        n_blx = len(line_blx) - a_blx

        if n_idx == 0:  # no line data
            return a_idx, a_blx

        if n_blx == 0:  # no more blx
            self.update(line_data, 2, a_idx, n_idx)
            return a_idx + n_idx, a_blx + n_blx

        current_blx = line_blx[a_blx]
        if current_blx == n_idx:
            self.update(line_data, 1, a_idx, n_idx)
        elif current_blx > n_idx:
            raise Exception("Invalid state")

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
            if len(maybe_black[0]) <= 0:
                raise Exception("Shouldn't get here")
            a_idx = maybe_black[0][0] + a_idx
            return self.pfx_internal(line_data, line_blx, a_idx, a_blx)
        else:
            if line_data[a_idx + current_blx] == 1:
                non_blacked = np.nonzero(line_data[a_idx + current_blx:] != 1)
                if len(non_blacked) == 0 or len(non_blacked[0]) == 0:
                    if n_blx > 1:
                        raise Exception("State error")
                    first_non_blacked = n_idx
                else:
                    first_non_blacked = current_blx + non_blacked[0][0]
                self.update(line_data, 2, a_idx, first_non_blacked - current_blx)
                return self.pfx_internal(line_data, line_blx, a_idx + first_non_blacked - current_blx, a_blx)

            blacked = np.nonzero(line_data[a_idx:a_idx + current_blx] == 1)
            if len(blacked[0]) > 0 and 0 <= blacked[0][0] < current_blx:
                if n_blx > 1:
                    next_blx = line_blx[a_blx + 1]
                else:
                    next_blx = n_idx
                whited = np.nonzero(line_data[a_idx + current_blx:a_idx + current_blx + 1 + next_blx] == 2)
                if len(whited[0]) > 0:
                    first_whited = whited[0][0]
                    self.freedom_solve(line_data, line_blx, a_idx, a_idx + current_blx + first_whited, a_blx, a_blx + 1)
                    return self.pfx_internal(line_data, line_blx, a_idx + current_blx + first_whited + 1, a_blx + 1)
                else:
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
        if current_blx == n_idx:
            self.update(line_data, 1, 0, n_idx)
        elif current_blx > n_idx:
            raise Exception("Invalid state")

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
            if len(maybe_black[0]) <= 0:
                raise Exception("Shouldn't get here")
            z_idx = maybe_black[0][-1] + 1
            return self.sfx_internal(line_data, line_blx, z_idx, z_blx)
        else:
            if line_data[z_idx - current_blx - 1] == 1:
                non_blacked = np.nonzero(line_data[:z_idx - current_blx] != 1)
                if len(non_blacked) == 0 or len(non_blacked[0]) == 0:
                    if n_blx > 1:
                        raise Exception("State error")
                    last_non_black = 0
                else:
                    last_non_black = max(non_blacked[0]) + 1
                self.update(line_data, 2, last_non_black + current_blx, z_idx - last_non_black - current_blx)
                return self.sfx_internal(line_data, line_blx, last_non_black + current_blx, z_blx)

            blacked = np.nonzero(line_data[z_idx - current_blx:z_idx] == 1)

            if len(blacked[0]) > 0 and 0 <= blacked[0][-1] < current_blx:
                if n_blx > 1:
                    previous_blx = line_blx[z_blx - 2]
                else:
                    previous_blx = 0
                whited = np.nonzero(line_data[z_idx - current_blx - previous_blx - 1:z_idx - current_blx] == 2)
                if len(whited[0]) > 0:
                    last_whited = whited[0][-1]
                    split_point = z_idx - current_blx - previous_blx + last_whited
                    self.freedom_solve(line_data, line_blx, split_point, z_idx, z_blx - 1, z_blx)
                    return self.sfx_internal(line_data, line_blx, split_point - 1, z_blx - 1)

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
        if self.bf_level == 1 and possibilities > 100:
            if self.key in self.bf_jobs.keys():
                possibilities = max(possibilities, self.bf_jobs[self.key])
            self.bf_jobs[self.key] = possibilities
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


def test(choice):
    blx_x = blx_y = []
    if choice == "man":
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

    if choice == "cactus":
        blx_x = [[1], [3, 3], [3, 5], [4, 2, 2], [11, 2],
                 [2, 7, 4], [5, 3, 4], [6, 1, 2, 1], [2, 5, 6], [4, 5, 4],
                 [4, 8, 1, 2], [4, 5, 4, 4], [2, 6, 7, 6], [7, 6, 2, 6], [1, 4, 10, 2, 3],
                 [4, 1, 10, 5], [5, 4, 5, 4], [5, 8, 3, 1], [2, 7], [5, 3],
                 [7], [6], [2, 2, 3], [7, 1], [9],
                 [3, 3], [5], [1, 2], [4], [4],
                 [8], [1, 1], [1, 1], [8], [2, 6, 2]]
        blx_y = [[3], [2, 2], [9], [4, 5], [1, 7, 2],
                 [5, 9], [7, 2, 1], [4, 2, 1, 3, 2], [4, 10, 1], [4, 8, 3, 2, 1],
                 [10, 12, 1, 1], [5, 9, 10, 1, 2], [2, 15, 2, 8, 2], [6, 3, 15, 3, 2], [11, 11, 5, 2],
                 [1, 10, 8, 2], [7, 1, 4, 1, 2], [2, 1, 1, 2, 1, 1], [5, 3, 2, 1], [1, 6, 1],
                 [9, 2], [6, 6], [3, 8, 1], [10], [1, 1]]

    if choice == "cable":
        blx_x = [[2, 2], [2, 3, 3], [1, 1, 3, 3], [4, 4, 3], [1, 5, 2, 1],
                 [4, 4, 3, 3], [7, 1, 1, 3, 5], [4, 7, 2, 1, 2, 4, 1], [2, 7, 6, 4, 5, 2], [1, 1, 4, 2, 2, 3, 2, 7, 1],
                 [1, 4, 2, 1, 3, 8, 2, 1], [3, 6, 4, 3, 2, 5, 1], [1, 6, 3, 3, 2, 4], [2, 10, 3, 2, 4], [1, 6, 2, 4, 1, 2, 4],
                 [3, 2, 2, 4, 2, 3, 4], [3, 2, 1, 3, 3, 3, 7], [3, 1, 1, 4, 4, 2, 3, 6], [3, 9, 4, 4, 12], [3, 3, 4, 2, 6, 6, 12],
                 [3, 3, 2, 2, 7, 20], [1, 3, 1, 6, 30], [3, 1, 6, 30], [2, 2, 2, 2, 26, 4], [3, 1, 2, 30, 3],
                 [1, 1, 2, 5, 6, 19, 1], [2, 2, 4, 2, 13, 3], [5, 1, 2, 5, 13, 4], [7, 1, 2, 9, 16], [5, 5, 16, 11],
                 [5, 2, 3, 2, 9, 13], [5, 8, 3, 6, 8, 1], [3, 6, 2, 4, 3, 3, 2, 3], [2, 6, 2, 4, 4, 6, 1, 2, 4], [8, 3, 3, 1, 4, 6, 1, 2, 2, 1],
                 [6, 2, 1, 1, 1, 1, 3, 6, 2, 1, 2, 2], [9, 3, 2, 1, 3, 4, 2, 1, 2, 2], [5, 1, 2, 3, 1, 1, 3, 1, 1, 2, 1, 4, 1], [4, 1, 2, 2, 1, 3, 1, 3, 1, 2, 1, 5, 3], [4, 2, 1, 1, 1, 1, 3, 3, 1, 2, 1, 5, 2],
                 [4, 1, 1, 1, 1, 1, 3, 5, 2, 1, 5, 2], [4, 1, 1, 1, 1, 3, 6, 2, 1, 5, 2], [4, 1, 1, 1, 1, 4, 6, 2, 1, 4, 2], [4, 1, 1, 1, 4, 6, 2, 1, 7], [2, 1, 1, 1, 3, 4, 6, 2, 1, 6, 1],
                 [1, 1, 1, 11, 6, 2, 1, 3, 2, 3], [1, 1, 1, 3, 1, 5, 1, 1, 2, 6], [1, 1, 3, 1, 10, 1, 7], [1, 2, 3, 7, 13, 1, 7], [1, 2, 9, 16, 5],
                 [1, 1, 9, 16, 2], [2, 4, 9, 16, 3], [7, 9, 16, 3, 1], [8, 8, 16, 1], [8, 8, 16, 2, 1],
                 [8, 8, 16, 3, 2], [8, 5, 2, 12, 6], [8, 2, 2, 4, 6, 6], [8, 5, 9, 1, 3], [6, 6, 10, 5],
                 [3, 2, 3, 11, 4, 5], [1, 5, 1, 20, 1, 10], [1, 3, 19, 1, 4, 6], [2, 18, 1, 1, 3, 6, 1], [17, 2, 3, 2, 5, 1],
                 [5, 2, 1, 3, 8, 2], [7, 9], [6, 3, 4, 6], [2, 15, 1, 1, 2], [7, 10, 1, 4, 2],
                 [1, 14, 9], [1, 5, 9, 1, 5, 2, 1], [6, 30], [3, 3, 13, 7, 5], [26, 2, 5]]

        blx_y = [[2, 2, 14], [1, 1, 15], [2, 2, 4, 10], [1, 1, 4, 27], [2, 1, 5, 5, 12],
                 [1, 1, 4, 4, 11, 9, 1], [2, 2, 4, 4, 2, 1, 2, 9, 1, 1], [1, 1, 3, 4, 1, 1, 1, 1, 1, 9, 2, 1], [2, 2, 3, 3, 2, 1, 1, 1, 9, 2, 1], [1, 1, 2, 3, 1, 1, 1, 9, 3, 1],
                 [2, 2, 2, 4, 1, 13, 6, 2, 2], [1, 1, 3, 1, 4, 3, 1, 1, 4, 2], [7, 2, 2, 4, 1, 1, 2, 1, 2, 9, 3, 2], [1, 2, 1, 3, 4, 2, 1, 1, 2, 1, 10, 2, 3], [4, 3, 2, 3, 3, 3, 2, 1, 1, 9, 2, 3, ],
                 [4, 9, 7, 1, 1, 1, 2, 1, 9, 2, 5], [2, 4, 10, 1, 1, 2, 1, 1, 10, 2, 5], [2, 4, 4, 3, 1, 1, 1, 1, 1, 2, 2, 8, 2, 5, 1, 1], [2, 8, 2, 2, 2, 1, 1, 12, 2, 6, 3], [2, 3, 8, 2, 4, 13, 9, 2, 6, 3],
                 [3, 5, 9, 4, 1, 1, 1, 5, 5], [2, 3, 2, 9, 1, 1, 1, 5, 1, 1], [4, 2, 6, 14, 9, 1, 5, 1, 1], [4, 3, 2, 3, 14, 9, 2, 5, 2], [8, 2, 2, 14, 9, 2, 4, 3],
                 [3, 2, 1, 5, 3, 3, 4, 9, 2, 4, 1, 3], [1, 2, 1, 2, 6, 3, 10, 1, 4, 1, 2, 1], [1, 2, 1, 6, 2, 6, 5, 10, 1, 4, 1, 4], [5, 2, 7, 2, 5, 8, 9, 5, 7], [4, 2, 7, 2, 5, 8, 9, 2, 2, 7],
                 [1, 2, 3, 8, 1, 4, 8, 9, 2, 2, 6], [6, 1, 8, 2, 3, 7, 9, 2, 2, 1, 2, 3], [2, 2, 9, 2, 3, 10, 10, 1, 1, 1, 2, 5], [1, 2, 4, 5, 2, 10, 3, 6, 5], [2, 1, 3, 5, 1, 13, 10, 3, 1, 9],
                 [1, 2, 3, 6, 1, 2, 12, 9, 2, 10], [2, 1, 2, 8, 1, 4, 9, 2, 9], [1, 2, 3, 12, 28, 2, 7], [1, 2, 3, 13, 14], [1, 1, 3, 14, 1, 1, 11],
                 [1, 2, 2, 14, 2, 1, 2, 6, 1], [1, 2, 12, 2, 2, 1, 3, 3], [2, 2, 3, 11, 3, 1, 2, 2, 3], [1, 2, 16, 4, 2, 1, 1, 2, 3], [2, 2, 16, 6, 2, 2, 2, 2, 1, 3],
                 [1, 4, 15, 7, 1, 3, 2, 2, 5], [3, 2, 14, 7, 2, 1, 2, 1, 9], [4, 3, 14, 4, 1, 2, 1, 9, 4], [6, 3, 11, 3, 4, 2, 3, 13, 4, 1], [5, 2, 8, 1, 2, 3, 4, 2, 3, 21, 1],
                 [10, 7, 2, 1, 2, 4, 1, 3, 5, 7, 2, 3], [2, 15, 2, 1, 3, 1, 1, 3, 2, 1, 4, 9], [3, 14, 2, 3, 1, 3, 4, 1, 2, 2, 5], [2, 12, 1, 2, 2, 2, 3, 2, 1, 5, 2], [2, 12, 1, 5, 2, 4, 2, 7, 1, 1]]

    if choice == "dog":
        blx_x = [[6], [8], [4, 9], [8, 10], [9, 3, 4],
                 [11, 4, 3], [12, 3, 3], [13, 4, 4], [14, 4, 4], [15, 4, 5],
                 [15, 4, 5], [17, 5, 5], [6, 8, 4, 6], [5, 8, 4, 6], [5, 8, 4, 7],
                 [4, 9, 4, 7], [4, 9, 5, 5], [4, 9, 5, 5], [4, 10, 7, 5], [4, 3, 5, 14, 5],
                 [4, 2, 24, 2, 6], [4, 2, 25, 7], [4, 1, 8, 11, 7], [4, 7, 2, 4, 8], [4, 7, 4, 4, 4, 5],
                 [5, 7, 4, 4, 6, 4, 4], [5, 3, 4, 2, 7, 5, 4], [5, 3, 4, 8, 5, 4], [6, 3, 2, 4, 2, 6, 6, 2], [12, 4, 5, 2, 3, 4, 7, 2],
                 [17, 4, 3, 4, 2, 2, 9], [15, 4, 2, 3, 3, 9], [5, 8, 4, 2, 3, 4, 9], [3, 8, 3, 2, 3, 6, 9], [2, 8, 4, 3, 2, 7, 8],
                 [2, 8, 3, 4, 4, 6, 6], [2, 8, 3, 3, 11, 4, 5], [15, 5, 13, 4, 4], [10, 5, 22, 3], [9, 5, 19, 3],
                 [8, 3, 17, 4, 3], [7, 2, 2, 17, 3, 2, 3], [7, 6, 16, 2, 2, 2], [7, 3, 4, 16, 6, 3], [8, 4, 3, 16, 7, 1, 4],
                 [8, 9, 17, 6, 2, 5], [9, 8, 5, 3, 3, 6], [10, 4, 2, 2, 7, 3, 7, 2, 5], [11, 4, 2, 2, 8, 2, 6], [12, 6, 1, 2, 2, 7, 2, 9],
                 [5, 6, 6, 2, 2, 2, 9, 11], [5, 6, 6, 2, 2, 5, 2, 9, 2, 17], [6, 5, 5, 3, 2, 5, 4, 6, 2, 2, 4, 17], [5, 5, 4, 2, 2, 2, 1, 3, 6, 2, 2, 4, 22], [6, 3, 4, 4, 2, 2, 8, 2, 1, 4, 25],
                 [7, 5, 2, 3, 8, 2, 1, 4, 28], [7, 6, 3, 4, 6, 2, 2, 5, 34], [7, 7, 8, 6, 2, 2, 5, 36], [7, 6, 6, 2, 2, 2, 2, 6, 35], [6, 6, 2, 2, 2, 3, 2, 6, 26, 8],
                 [6, 5, 2, 2, 4, 2, 7, 26, 8], [6, 4, 3, 4, 2, 7, 27, 11], [3, 4, 4, 5, 5, 2, 7, 27, 12], [3, 3, 3, 7, 5, 2, 7, 28, 11], [4, 3, 4, 9, 6, 2, 5, 30, 12],
                 [5, 3, 3, 10, 6, 2, 5, 4, 26, 12], [6, 3, 2, 6, 3, 6, 3, 5, 4, 28, 13], [7, 3, 3, 3, 2, 3, 5, 3, 4, 3, 27, 12], [8, 2, 4, 3, 2, 4, 5, 3, 4, 1, 27, 13], [9, 2, 3, 2, 3, 7, 6, 3, 4, 1, 26, 13],
                 [3, 5, 2, 13, 10, 3, 4, 1, 27, 13], [4, 9, 11, 10, 4, 5, 2, 27, 12], [4, 16, 28, 5, 2, 28, 11], [5, 13, 30, 5, 1, 28, 9], [5, 11, 31, 5, 2, 29, 10],
                 [6, 43, 5, 3, 28, 11], [7, 43, 5, 4, 29, 13], [8, 11, 7, 18, 5, 4, 7, 20, 12], [10, 10, 7, 18, 4, 4, 7, 7, 11, 13], [10, 9, 7, 18, 3, 4, 7, 7, 11, 15],
                 [5, 8, 7, 18, 1, 3, 6, 6, 11, 14], [4, 7, 6, 18, 4, 7, 6, 11, 14], [3, 4, 6, 5, 18, 4, 6, 6, 10, 15], [3, 4, 6, 6, 17, 4, 5, 6, 9, 12], [3, 5, 5, 5, 18, 3, 3, 4, 5, 7, 12],
                 [3, 6, 6, 4, 27, 3, 4, 11], [2, 13, 3, 30, 3, 2, 2, 11], [3, 13, 2, 31, 3, 3, 3, 11], [2, 13, 1, 5, 25, 3, 3, 5, 11], [2, 12, 4, 25, 9, 6, 7, 11],
                 [2, 13, 5, 24, 9, 7, 10, 11], [3, 19, 20, 9, 20, 11], [4, 16, 22, 8, 22, 11], [4, 12, 24, 8, 23, 12], [5, 11, 2, 26, 7, 24, 12],
                 [7, 10, 32, 6, 25, 13], [8, 9, 15, 13, 6, 13, 3, 6, 15], [8, 8, 15, 10, 6, 14, 5, 5, 16], [9, 8, 13, 14, 30, 5, 17], [19, 11, 5, 9, 31, 7, 17],
                 [20, 8, 7, 13, 40, 13], [22, 4, 9, 13, 41, 11], [24, 11, 12, 43, 9], [17, 26, 11, 44, 8], [16, 26, 11, 55, 7],
                 [16, 26, 10, 57, 6], [15, 25, 10, 9, 48, 5], [15, 23, 10, 6, 30, 18, 5], [15, 23, 10, 2, 31, 18, 4], [14, 21, 10, 3, 31, 17, 4],
                 [14, 18, 9, 3, 31, 14, 4], [14, 2, 8, 3, 32, 10, 4], [13, 2, 7, 2, 10, 20, 8, 4], [12, 2, 6, 3, 9, 21, 6, 4], [12, 2, 6, 2, 4, 21, 5, 3],
                 [5, 6, 2, 5, 2, 21, 4, 3], [5, 6, 2, 5, 2, 22, 4, 3], [5, 5, 2, 5, 3, 21, 4, 4], [6, 4, 8, 3, 19, 5, 4], [6, 2, 7, 2, 17, 5, 4],
                 [7, 2, 7, 2, 16, 5, 3], [7, 2, 7, 3, 13, 3, 4], [8, 2, 7, 3, 12, 4, 2], [7, 5, 6, 2, 11, 4, 3], [6, 4, 5, 2, 10, 6, 4],
                 [5, 5, 4, 3, 10, 6, 4], [5, 5, 4, 3, 10, 5, 3], [5, 5, 3, 2, 10, 5, 3], [5, 5, 3, 2, 10, 3, 3], [6, 4, 3, 3, 10, 3, 2, 2],
                 [2, 4, 4, 3, 3, 10, 6, 1, 2], [2, 5, 2, 4, 2, 10, 4, 2, 2], [11, 2, 3, 3, 10, 3, 3, 2], [16, 3, 3, 10, 7, 2], [17, 4, 3, 11, 6, 2],
                 [17, 4, 2, 12, 8, 2], [18, 4, 2, 17, 8, 2], [7, 2, 5, 10, 2, 17, 7, 2], [2, 2, 1, 8, 12, 2, 18, 7, 2], [3, 2, 2, 12, 2, 9, 2, 17, 7, 2],
                 [27, 2, 14, 17, 7, 2], [41, 2, 2, 10, 16, 7, 3], [48, 3, 10, 12, 7, 2], [47, 1, 7, 4, 5, 2], [47, 2, 6, 3, 6, 3],
                 [46, 2, 2, 3, 3, 9, 2], [46, 2, 2, 2, 3, 10, 2], [48, 1, 3, 2, 3, 7, 1], [48, 1, 5, 6, 4, 2, 2, 1], [69, 3, 2, 2, 1],
                 [69, 3, 2, 1, 2], [70, 14], [69, 11]]

        blx_y = [[20, 8], [26, 13], [29, 16], [33, 17], [12, 13, 4, 13],
                 [11, 5, 17, 5, 13], [10, 3, 20, 16, 21], [10, 3, 26, 24, 22], [10, 59, 8, 13], [10, 42, 35, 2, 5, 13],
                 [20, 23, 10, 6, 8, 22, 3, 5, 14], [18, 27, 9, 7, 5, 22, 31], [15, 4, 11, 10, 9, 6, 4, 7, 46, 13], [13, 4, 11, 9, 8, 6, 3, 9, 43, 13], [11, 4, 10, 2, 8, 5, 6, 2, 11, 42, 14],
                 [17, 5, 1, 4, 7, 4, 4, 7, 2, 12, 32, 6, 14], [15, 3, 2, 5, 5, 7, 4, 6, 11, 29, 21], [13, 4, 6, 10, 12, 11, 17, 6, 20], [10, 9, 2, 3, 15, 11, 11, 18, 3, 4, 20], [6, 9, 3, 3, 19, 11, 13, 19, 8, 20],
                 [3, 9, 3, 6, 11, 5, 47, 30], [3, 8, 5, 5, 9, 3, 47, 10, 15], [2, 6, 6, 4, 4, 2, 59, 15], [2, 3, 6, 1, 3, 2, 52, 15], [2, 2, 3, 3, 2, 40, 14],
                 [2, 1, 6, 3, 9, 14, 14], [2, 18, 2, 6, 12, 13], [2, 2, 19, 3, 3, 4, 6, 2, 7, 13], [2, 4, 2, 13, 6, 4, 3, 7, 3, 6, 12], [3, 4, 11, 1, 5, 5, 4, 8, 3, 3, 6, 12],
                 [4, 2, 10, 2, 5, 4, 2, 13, 3, 4, 6, 12], [4, 3, 10, 2, 4, 3, 4, 15, 4, 6, 6, 12], [4, 6, 11, 1, 4, 2, 4, 14, 5, 7, 7, 12], [4, 17, 1, 2, 5, 12, 4, 6, 6, 12], [4, 3, 16, 1, 10, 12, 3, 7, 7, 12],
                 [5, 3, 11, 1, 4, 25, 4, 8, 7, 12], [7, 4, 4, 10, 2, 4, 2, 5, 4, 7, 4, 8, 7, 12], [12, 4, 5, 9, 2, 2, 3, 4, 1, 5, 6, 10, 8, 12], [16, 5, 5, 10, 8, 4, 1, 5, 21, 8, 12], [15, 2, 5, 5, 10, 3, 2, 3, 2, 29, 8, 12],
                 [17, 2, 5, 4, 11, 4, 28, 8, 12], [11, 3, 2, 4, 4, 4, 3, 2, 3, 28, 9, 12], [6, 2, 3, 3, 3, 2, 2, 2, 3, 27, 10, 11], [4, 4, 2, 3, 3, 6, 2, 26, 10, 11], [4, 6, 3, 5, 2, 2, 4, 27, 11, 11],
                 [4, 2, 9, 2, 6, 3, 7, 27, 13, 11], [4, 4, 9, 2, 2, 3, 14, 26, 13, 3, 6], [5, 8, 3, 9, 1, 1, 3, 11, 26, 13, 3, 8], [23, 9, 5, 14, 28, 13, 3, 3, 4], [22, 9, 2, 15, 47, 2, 3, 4],
                 [21, 9, 12, 37, 11, 2, 1, 4], [21, 12, 6, 39, 9, 5, 3, 8], [27, 2, 41, 8, 11, 4, 4], [82, 36, 2, 5], [5, 24, 11, 18, 37, 6],
                 [3, 11, 20, 14, 16], [4, 3, 14, 42, 16], [4, 26, 40, 9, 5], [4, 18, 7, 32, 7, 4], [7, 7, 4, 24, 6, 4],
                 [17, 7, 4, 20, 6, 5], [16, 6, 15, 13], [17, 2, 4, 13, 13], [3, 13, 9, 12, 15, 5], [5, 25, 11, 12, 5],
                 [6, 22, 10, 10, 5], [8, 19, 10, 4], [8, 11, 10, 4], [11, 10, 4], [18, 2, 5, 5, 3, 10, 3],
                 [63, 2], [17, 38], [16, 32], [14, 3, 19], [16, 19],
                 [17, 22], [17, 9, 9], [19, 7, 10], [23, 4, 3, 10], [33, 11, 2],
                 [33, 19], [32, 22], [31, 24], [30, 25], [29, 27],
                 [27, 2, 29], [23, 3, 29], [23, 4, 28], [31, 26], [31, 23],
                 [29, 22], [28, 6, 14], [27, 6, 15], [26, 25], [3, 19, 28],
                 [3, 17, 29], [3, 15, 7, 21], [3, 18, 7, 21, 4], [4, 20, 7, 22, 6], [5, 20, 7, 22, 6],
                 [4, 19, 7, 21, 7], [5, 19, 7, 23, 7], [6, 18, 31, 7], [5, 16, 30, 8], [6, 15, 35, 9],
                 [8, 14, 47], [8, 10, 45], [8, 7, 44], [10, 43], [12, 42],
                 [16, 5, 33], [19, 6, 32], [21, 6, 30], [21, 7, 28], [19, 6, 7, 8],
                 [20, 7, 3], [20, 7], [20, 3, 8], [20, 4, 9], [5, 16, 5, 9],
                 [3, 25, 10], [25, 11], [25, 13], [23, 14], [23, 14],
                 [23, 16], [21, 6, 6], [19, 5, 7], [18, 4, 8], [17, 2, 2, 6],
                 [16, 7], [14, 7], [12, 9], [12, 3, 5, 3], [13, 7, 6],
                 [17, 3, 6, 8], [15, 2, 16, 2], [12, 2, 15, 2], [11, 17, 5], [8, 3, 18],
                 [7, 13, 2], [4, 8, 4, 2], [4, 4, 3, 2], [5, 8], [5, 3, 2],
                 [5, 1, 2], [6, 1], [6, 2], [7]]

    if choice == "horse":
        blx_x = [[4, 5, 5, 5], [4, 5, 5, 5], [5, 8, 3, 5, 5], [2, 17, 5, 5], [10, 8, 1, 5, 5],
                 [9, 4, 4, 1, 5], [11, 2, 16, 3, 2, 2, 5], [1, 1, 2, 3, 3, 2, 3, 4, 1, 3, 8, 5], [1, 6, 2, 3, 1, 6, 5, 12, 5], [3, 4, 2, 1, 2, 3, 9, 8, 4, 5],
                 [1, 4, 2, 1, 2, 3, 2, 4, 3, 5, 2, 5], [1, 4, 2, 1, 2, 2, 2, 2, 2, 5, 1, 4, 5, 5], [1, 3, 2, 1, 5, 2, 2, 1, 3, 4, 5, 5], [1, 2, 2, 1, 1, 4, 2, 2, 1, 5, 4, 5], [1, 2, 4, 1, 3, 7, 3, 8, 1, 2, 5],
                 [4, 3, 1, 7, 3, 6, 2, 3, 7, 6, 3], [4, 4, 6, 1, 3, 5, 5, 1, 8, 8, 5, 2], [4, 3, 4, 3, 3, 5, 7, 10, 7, 5, 2], [2, 9, 2, 2, 2, 7, 1, 9, 5, 1, 4, 2], [1, 2, 8, 4, 7, 3, 8, 4, 3, 3, 2],
                 [1, 14, 5, 8, 1, 9, 3, 4, 2], [1, 1, 11, 3, 1, 1, 8, 10, 3, 5, 5], [1, 2, 6, 1, 3, 1, 7, 1, 8, 4, 5, 5], [1, 2, 1, 2, 1, 5, 2, 5, 3, 7, 4, 5, 5], [2, 8, 2, 6, 3, 4, 1, 7, 5, 5, 5],
                 [4, 7, 3, 7, 5, 3, 6, 6, 5, 5], [4, 2, 3, 3, 6, 9, 6, 2, 5, 5, 5], [4, 7, 4, 1, 5, 10, 8, 3, 5, 5], [2, 3, 1, 2, 4, 9, 2, 1, 4, 3, 5, 5], [1, 9, 3, 7, 6, 4, 2, 4, 2, 5, 5],
                 [1, 6, 4, 1, 4, 4, 3, 4, 1, 5, 5], [1, 6, 2, 6, 2, 2, 8, 5, 3, 1, 5, 5], [3, 3, 4, 1, 7, 1, 2, 5, 2, 3, 5, 5], [1, 4, 3, 2, 3, 9, 7, 5, 5, 2, 5, 5], [1, 8, 4, 1, 9, 7, 1, 7, 2, 5, 5],
                 [3, 7, 1, 5, 7, 5, 2, 11, 2, 5, 5], [1, 1, 5, 3, 2, 1, 1, 7, 3, 1, 1, 1, 1, 2, 5, 5], [1, 3, 3, 5, 1, 3, 5, 4, 3, 12, 2, 5, 5], [5, 1, 3, 3, 2, 1, 3, 5, 4, 4, 6, 2, 5, 5], [1, 3, 7, 2, 4, 3, 3, 15, 2, 3, 1, 3, 5],
                 [1, 1, 1, 6, 1, 2, 1, 1, 3, 3, 15, 2, 2, 1, 1, 2, 5], [3, 1, 5, 2, 3, 3, 1, 14, 3, 2, 2, 5], [1, 4, 3, 6, 3, 1, 1, 8, 4, 3, 4, 5, 5], [1, 4, 2, 2, 6, 4, 3, 4, 10, 4, 5, 4, 5], [3, 4, 1, 3, 5, 4, 7, 5, 5, 5],
                 [1, 9, 2, 4, 5, 11, 6, 6, 5], [1, 9, 2, 1, 1, 2, 4, 1, 10, 4, 2, 2, 3, 5], [1, 3, 1, 3, 2, 5, 1, 1, 5, 2, 3, 3, 2, 5], [2, 3, 3, 2, 2, 1, 1, 1, 1, 4, 10, 4, 3, 5], [5, 2, 2, 1, 3, 6, 2, 1, 1, 1, 3, 5, 4, 4, 2, 5],
                 [5, 1, 2, 2, 3, 2, 1, 1, 1, 1, 1, 2, 4, 5, 5, 1, 5], [3, 1, 2, 3, 5, 1, 2, 1, 1, 1, 5, 6, 4, 5], [3, 4, 3, 1, 2, 1, 3, 1, 1, 1, 3, 3, 1, 5], [3, 1, 2, 2, 2, 6, 1, 1, 4, 1, 1, 7, 10, 5], [2, 5, 1, 3, 1, 4, 1, 21, 5],
                 [2, 1, 2, 1, 2, 3, 2, 4, 1, 5, 24, 2], [1, 2, 2, 3, 1, 1, 1, 5, 1, 5, 15, 9, 2], [1, 2, 1, 1, 3, 3, 5, 1, 4, 1, 12, 2, 4], [1, 1, 1, 1, 3, 2, 1, 1, 1, 4, 1, 3, 3, 10, 5, 5], [1, 3, 4, 5, 3, 3, 1, 4, 1, 9, 1, 5, 4],
                 [5, 1, 2, 1, 1, 2, 1, 5, 7, 3, 5, 3], [12, 5, 3, 3, 1, 1, 4, 1, 11, 5, 1, 2], [12, 1, 5, 1, 1, 1, 3, 3, 9, 1, 5, 2], [12, 5, 5, 3, 1, 4, 1, 9, 2, 5, 5], [12, 1, 7, 1, 2, 3, 8, 2, 5, 5],
                 [12, 5, 7, 1, 6, 4, 1, 5], [1, 10, 1, 6, 1, 3, 4, 5], [1, 10, 6, 4, 4, 5, 3, 2, 3, 5], [1, 10, 9, 3, 2, 2, 5, 3, 5, 5], [1, 10, 3, 8, 1, 1, 1, 5, 3, 6, 5],
                 [1, 10, 5, 3, 3, 2, 5, 9, 1, 5], [3, 12, 4, 2, 1, 2, 5, 6, 4, 5], [1, 15, 6, 2, 5, 1, 5, 5], [1, 1, 10, 2, 4, 5, 5, 5, 5], [23, 4, 5, 5, 5, 5]]

        blx_y = [[1], [3], [2, 1, 1, 1, 2], [22, 3, 1, 3, 26], [1, 1, 4, 4, 1, 2, 1, 3, 1, 2, 1, 7, 6, 1, 1],
                 [1, 9, 3, 3, 1, 1, 1, 1, 1, 1, 5, 2, 12, 1], [3, 8, 2, 2, 7, 11, 6, 1, 1, 11, 3], [1, 5, 3, 1, 4, 1, 1, 6, 2, 1, 1, 13, 1], [1, 4, 3, 2, 2, 3, 4, 4, 3, 1, 1, 12, 1], [1, 2, 5, 2, 4, 3, 4, 1, 1, 2, 4, 2, 2, 12, 1],
                 [1, 2, 7, 2, 3, 1, 3, 5, 2, 2, 2, 2, 4, 1, 12, 1], [3, 2, 1, 7, 2, 1, 3, 4, 6, 2, 1, 2, 2, 12, 1], [2, 2, 1, 1, 5, 2, 1, 1, 4, 4, 2, 4, 1, 3, 14], [1, 2, 1, 1, 7, 2, 1, 1, 2, 2, 6, 2, 3, 4, 2, 14], [2, 1, 1, 8, 2, 1, 1, 5, 1, 5, 3, 2, 1, 14],
                 [2, 1, 9, 1, 2, 3, 3, 4, 1, 2, 1, 4], [1, 1, 11, 1, 1, 5, 2, 1, 2, 2, 4], [2, 6, 4, 1, 2, 6, 5, 3, 2, 3, 1, 4], [1, 4, 2, 3, 1, 1, 8, 5, 3, 2, 4], [2, 2, 1, 1, 2, 20, 2, 1, 5, 3, 3],
                 [2, 2, 1, 1, 1, 2, 3, 3, 3, 3, 2, 1, 3, 1, 4, 1, 3], [1, 2, 5, 1, 2, 1, 1, 1, 2, 1, 2, 3, 1, 1, 1, 4, 2], [2, 1, 1, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 1, 1, 1, 5, 1], [1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 6, 1], [1, 1, 7, 3, 5, 2, 3, 7, 1, 3, 1, 1, 1, 6, 1],
                 [1, 3, 1, 5, 4, 3, 3, 5, 3, 1, 1, 1, 7, 2, 2, 1], [7, 2, 5, 3, 4, 1, 1, 2, 3, 7, 2, 1], [9, 6, 3, 1, 1, 2, 2, 1, 1, 1, 7, 5], [7, 8, 3, 1, 1, 2, 1, 1, 1, 3, 6, 5], [5, 6, 9, 1, 3, 1, 1, 1, 1, 2, 1, 1, 6, 1, 1],
                 [3, 7, 4, 9, 5, 1, 1, 1, 4, 3, 6, 4], [4, 2, 4, 10, 6, 2, 6, 1, 1, 1, 1, 1, 3], [4, 2, 7, 1, 1, 1, 1, 1, 1, 5, 2, 8, 3, 3, 2], [4, 2, 8, 1, 1, 1, 1, 1, 1, 3, 1, 8, 1, 1, 2, 1], [1, 3, 2, 5, 8, 12, 6, 2],
                 [1, 3, 2, 4, 1, 2, 7, 1, 12, 1, 11, 1], [2, 3, 2, 1, 5, 6, 1, 12, 1, 1, 4, 4], [2, 3, 3, 7, 5, 1, 3, 5, 2, 1, 1, 13, 2, 1], [3, 4, 3, 8, 5, 1, 2, 2, 4, 2, 1, 1, 12, 3], [2, 5, 4, 9, 4, 1, 3, 1, 5, 2, 1, 1, 12],
                 [1, 3, 2, 3, 10, 4, 2, 4, 5, 2, 1, 1, 1, 4, 3, 2, 8], [1, 2, 2, 2, 14, 4, 6, 3, 1, 1, 1, 2, 1, 1, 1, 1, 8], [3, 2, 2, 1, 3, 3, 5, 2, 9, 1, 4, 1, 1, 1, 3, 3, 1, 8], [3, 3, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 5, 1, 5, 1, 1, 1, 1, 1, 1, 2, 8], [3, 4, 1, 3, 3, 3, 1, 3, 2, 1, 5, 1, 6, 1, 4, 3, 3, 8],
                 [4, 3, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 1, 3, 1, 7, 13], [4, 2, 1, 3, 3, 4, 1, 2, 2, 1, 3, 1, 2, 1, 13], [4, 1, 15, 3, 2, 1, 2, 2, 7, 13], [3, 1, 14, 3, 3, 2, 1, 3, 6, 12], [3, 1, 15, 3, 3, 3, 2, 2, 4, 12],
                 [1, 3, 14, 2, 4, 6, 7, 8, 4, 3], [1, 3, 1, 16, 5, 4, 2, 2, 7, 9, 3, 6, 2], [1, 3, 2, 16, 6, 3, 3, 7, 3, 6, 3, 7, 2], [1, 2, 2, 15, 1, 6, 11, 4, 6, 3, 8, 2], [1, 1, 2, 3, 5, 3, 7, 7, 5, 6, 2, 3, 2, 2],
                 [1, 3, 7, 7, 6, 5, 2, 2, 2], [1, 2, 15, 5, 3, 6, 4, 2], [2, 14, 3, 5, 3], [3, 11, 3, 4, 2], [2, 4, 3, 4, 4],
                 [5, 3, 5, 3, 18, 2, 2, 1, 3, 3, 9, 3, 3], [5, 3, 4, 2, 19, 2, 2, 4, 3, 8, 2, 4], [5, 2, 3, 3, 21, 4, 4, 3, 7, 4, 4], [5, 4, 3, 2, 22, 4, 4, 3, 7, 3, 4], [5, 4, 2, 2, 23, 5, 3, 3, 7, 2, 5],
                 [3, 2, 7, 2, 1], [2, 3, 6, 2], [3, 3, 2], [3, 2], [4, 2],
                 [15, 4, 34, 3, 14], [15, 3, 34, 4, 13], [16, 34, 3, 12], [57, 4, 12], [57, 4, 12]]

    t = time.time()
    Griddler(blx_x, blx_y).solve()
    print(time.time() - t)


# test("man")
# test("cactus")
# test("cable")
test("horse")
# test("dog")
