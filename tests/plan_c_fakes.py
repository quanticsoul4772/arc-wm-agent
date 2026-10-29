"""Fakes for the Plan C self-test (scripts/mechanism_selftest.py) and its tests.

`BlockGame` is test_agent.FakeGame with a 2x2 block instead of a single cell, so a move changes 4 cells, more
than the gate's change floor, and a correct `predict` can be accepted. UP at the top edge changes nothing (the
no-op path). From level 2 on, the block is drawn mirrored, so a model carried over from level 1 misses (the
grant-revoke path).
"""
from __future__ import annotations

from tests.test_agent import FakeGame

BLOCK = 8


class BlockGame(FakeGame):
    def _frame(self):
        row = self.row if self.level < 2 else 6 - self.row
        grid = [[BLOCK if (r in (row, row + 1) and c in (3, 4)) else 0 for c in range(8)] for r in range(8)]
        return {"grid": grid, "step": self.action_num, "level": self.level}


# A correct model of BlockGame on level 1, written as a model cell would be (BLOCK renders as "R").
BLOCK_PREDICT = '''
def predict(ascii, action):
    rows = ascii.splitlines()
    marked = [i for i, r in enumerate(rows) if "R" in r]
    if not marked:
        return ascii
    top, bottom = marked[0], marked[-1]
    block, blank = rows[top], rows[top].replace("R", "W")
    if action == "UP" and top > 0:
        rows[top - 1], rows[bottom] = block, blank
    elif action == "DOWN" and bottom < len(rows) - 1:
        rows[bottom + 1], rows[top] = block, blank
    return "\\n".join(rows)
'''
