"""Object segmentation of a colour grid.

`segment(rows)` splits a grid into 4-connected regions of equal colour and
describes each one for the model:

- `id`: position in reading order of the region's first (top-most, then
  left-most) cell;
- `color`: the colour character;
- `hash`: a position-independent signature of colour plus shape (the region's
  cells shifted so its bounding box starts at (0, 0));
- `pixels`: cell count;
- `boundary`: the clockwise outer contour, traced cell to cell with
  8-neighbour moves from the first cell and reduced to the points where the
  direction of travel changes, as `[row, col]` pairs;
- `children`: ids of the regions this one directly encloses (a region is
  enclosed by another when it cannot reach the grid edge without crossing it;
  the innermost encloser is the parent).

`adjacency_list` holds every `[i, j]` (i < j) pair of regions that touch along
an edge. Only the standard library is used, so the sandbox child can import it.
"""
from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Sequence
from typing import Any

ASCII_ALPHABET = "WwgGcBMPRbSYOrNp"

# Clockwise order (rows grow downwards): W, NW, N, NE, E, SE, S, SW.
_RING = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))
_STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def cell_char(value: Any, color_chars: str = ASCII_ALPHABET) -> str:
    """Colour character for a grid value; strings are taken as already rendered."""
    if isinstance(value, str):
        return value
    index = int(value)
    index = 0 if index < 0 else (len(color_chars) - 1 if index >= len(color_chars) else index)
    return color_chars[index]


def to_char_rows(rows: Sequence[Sequence[Any]] | str, color_chars: str = ASCII_ALPHABET) -> list[str]:
    """Normalise a numeric grid, a list of row strings, or an ascii board to row strings."""
    if isinstance(rows, str):
        return rows.split("\n") if rows else []
    return ["".join(cell_char(v, color_chars) for v in row) for row in rows]


def shape_hash(color: str, cells: list[tuple[int, int]]) -> str:
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    normalised = sorted((r - top, c - left) for r, c in cells)
    return hashlib.sha1(repr((color, normalised)).encode()).hexdigest()[:16]


def _label(grid: list[str]) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    height = len(grid)
    labels = [[-1] * len(row) for row in grid]
    regions: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(len(grid[r])):
            if labels[r][c] != -1:
                continue
            ident = len(regions)
            colour = grid[r][c]
            labels[r][c] = ident
            cells = []
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in _STEPS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < len(grid[nr]) and labels[nr][nc] == -1 and grid[nr][nc] == colour:
                        labels[nr][nc] = ident
                        queue.append((nr, nc))
            cells.sort()
            regions.append(cells)
    return labels, regions


def _trace(cells: list[tuple[int, int]], member: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Moore-neighbour contour walk, clockwise from the first cell in reading order."""
    start = cells[0]
    if len(cells) == 1:
        return [start]
    path = [start]
    current = start
    came_from = 0  # index in _RING of the cell we last checked before moving (start: west)
    first_move: tuple[tuple[int, int], tuple[int, int]] | None = None
    limit = 4 * len(cells) + 8
    for _ in range(limit):
        moved = False
        for k in range(1, 9):
            index = (came_from + k) % 8
            dr, dc = _RING[index]
            candidate = (current[0] + dr, current[1] + dc)
            if candidate in member:
                move = (current, candidate)
                if first_move is None:
                    first_move = move
                elif move == first_move:
                    return path[:-1]
                # The empty cell checked just before `candidate` becomes the backtrack
                # point; the next scan around `candidate` starts right after it.
                before = _RING[(index - 1) % 8]
                back = (current[0] + before[0] - candidate[0], current[1] + before[1] - candidate[1])
                came_from = _RING.index(back)
                current = candidate
                path.append(current)
                moved = True
                break
        if not moved:
            return path
    return path


def _corners(path: list[tuple[int, int]]) -> list[list[int]]:
    if len(path) < 3:
        return [list(p) for p in path]
    count = len(path)
    out = []
    for i in range(count):
        prev_p, here, next_p = path[i - 1], path[i], path[(i + 1) % count]
        incoming = (here[0] - prev_p[0], here[1] - prev_p[1])
        outgoing = (next_p[0] - here[0], next_p[1] - here[1])
        if incoming != outgoing:
            out.append([here[0], here[1]])
    return out


def _holes(cells: list[tuple[int, int]], member: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Cells inside the region's bounding box that cannot reach the outside
    through 4-neighbour moves without crossing the region."""
    top = min(r for r, _ in cells) - 1
    bottom = max(r for r, _ in cells) + 1
    left = min(c for _, c in cells) - 1
    right = max(c for _, c in cells) + 1
    outside: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r in range(top, bottom + 1):
        for c in (left, right):
            if (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))
    for c in range(left, right + 1):
        for r in (top, bottom):
            if (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS4:
            nxt = (r + dr, c + dc)
            if top <= nxt[0] <= bottom and left <= nxt[1] <= right and nxt not in outside and nxt not in member:
                outside.add(nxt)
                queue.append(nxt)
    inside = set()
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if (r, c) not in outside and (r, c) not in member:
                inside.add((r, c))
    return inside


def segment(rows: Sequence[Sequence[Any]] | str, color_chars: str = ASCII_ALPHABET) -> dict[str, list]:
    grid = to_char_rows(rows, color_chars)
    labels, regions = _label(grid)
    members = [set(cells) for cells in regions]

    # Enclosure: which regions lie in the holes of which.
    enclosers: list[list[int]] = [[] for _ in regions]
    for ident, cells in enumerate(regions):
        rows_span = cells[-1][0] - cells[0][0]
        cols = [c for _, c in cells]
        if len(cells) < 8 or rows_span < 2 or max(cols) - min(cols) < 2:
            continue
        inside = _holes(cells, members[ident])
        if not inside:
            continue
        for other in sorted({labels[r][c] for r, c in inside}):
            enclosers[other].append(ident)
    depth = [len(e) for e in enclosers]
    children: list[list[int]] = [[] for _ in regions]
    for ident, outer in enumerate(enclosers):
        if outer:
            parent = max(outer, key=lambda o: (depth[o], -o))
            children[parent].append(ident)

    adjacency: set[tuple[int, int]] = set()
    for r, row in enumerate(labels):
        for c, here in enumerate(row):
            if c + 1 < len(row) and row[c + 1] != here:
                adjacency.add((min(here, row[c + 1]), max(here, row[c + 1])))
            if r + 1 < len(labels) and c < len(labels[r + 1]) and labels[r + 1][c] != here:
                below = labels[r + 1][c]
                adjacency.add((min(here, below), max(here, below)))

    nodes = []
    for ident, cells in enumerate(regions):
        colour = grid[cells[0][0]][cells[0][1]]
        nodes.append(
            {
                "id": ident,
                "color": colour,
                "hash": shape_hash(colour, cells),
                "pixels": len(cells),
                "boundary": _corners(_trace(cells, members[ident])),
                "children": sorted(children[ident]),
            }
        )
    return {"nodes": nodes, "adjacency_list": [[a, b] for a, b in sorted(adjacency)]}
