from wmagent.segmentation import ASCII_ALPHABET, segment, shape_hash


def _seg(text: str) -> dict:
    return segment("\n".join(line.strip() for line in text.strip().splitlines()))


def test_regions_are_four_connected_and_in_reading_order():
    seg = _seg(
        """
        BBR
        BRB
        RBB
        """
    )
    nodes = seg["nodes"]
    # Diagonal R cells are separate regions; the B cells form two regions either side.
    assert [(n["id"], n["color"], n["pixels"]) for n in nodes] == [
        (0, "B", 3), (1, "R", 1), (2, "R", 1), (3, "B", 3), (4, "R", 1),
    ]


def test_hash_ignores_position_and_depends_on_colour_and_shape():
    a = _seg("BBBB\nBRRB\nBBBB")["nodes"][1]
    b = _seg("BBBB\nBBRR\nBBBB")["nodes"][1]
    c = _seg("BBBB\nBYYB\nBBBB")["nodes"][1]
    d = _seg("BRBB\nBRBB\nBBBB")["nodes"][1]
    assert a["hash"] == b["hash"]
    assert a["hash"] != c["hash"]
    assert a["hash"] != d["hash"]
    assert len(a["hash"]) == 16


def test_hash_matches_recorded_values():
    # Values printed by the model in recorded runs.
    assert shape_hash("G", [(5, 9)]) == "cc53199b01fcac5a"
    n_block = [(r, c) for r in range(5) for c in range(5) if (r, c) not in {(0, 0), (0, 4), (4, 0), (4, 4)}]
    assert shape_hash("N", [(r + 1, c + 13) for r, c in n_block]) == "027ce8803e516794"


def test_boundary_is_clockwise_corner_list():
    rect = _seg("BBBBB\nBRRRB\nBRRRB\nBBBBB")["nodes"][1]
    assert rect["boundary"] == [[1, 1], [1, 3], [2, 3], [2, 1]]
    line = _seg("BBBB\nBYBB\nBYBB\nBBBB")["nodes"][1]
    assert line["boundary"] == [[1, 1], [2, 1]]
    dot = _seg("BB\nBG")["nodes"][1]
    assert dot["boundary"] == [[1, 1]]


def test_boundary_of_recorded_irregular_shape():
    # The 6-cell shape the model traced in a recorded bp35 frame (offset to the origin).
    grid = [
        "SBBbBB",
        "SBbbYB",
        "SBbbYB",
        "SBBbBB",
    ]
    nodes = segment(grid)["nodes"]
    shape = next(n for n in nodes if n["color"] == "b")
    assert shape["pixels"] == 6
    assert shape["boundary"] == [[0, 3], [3, 3], [2, 2], [1, 2]]
    yellow = next(n for n in nodes if n["color"] == "Y")
    assert yellow["boundary"] == [[1, 4], [2, 4]]


def test_children_are_directly_enclosed_regions():
    seg = _seg(
        """
        WWWWWWWWW
        WBBBBBBBW
        WBRRRRRBW
        WBRGGGRBW
        WBRGYGRBW
        WBRGGGRBW
        WBRRRRRBW
        WBBBBBBBW
        WWWWWWWWW
        """
    )
    by_colour = {n["color"]: n for n in seg["nodes"]}
    assert by_colour["W"]["children"] == [by_colour["B"]["id"]]
    assert by_colour["B"]["children"] == [by_colour["R"]["id"]]
    assert by_colour["R"]["children"] == [by_colour["G"]["id"]]
    assert by_colour["G"]["children"] == [by_colour["Y"]["id"]]
    assert by_colour["Y"]["children"] == []


def test_open_shape_encloses_nothing_and_border_cells_are_never_enclosed():
    seg = _seg(
        """
        RRRRR
        RBBBR
        RBYBB
        RBBBR
        RRRRR
        """
    )
    red = seg["nodes"][0]
    assert red["color"] == "R"
    # The B ring leaks to the right edge through (2, 4), so R encloses nothing.
    assert red["children"] == []
    blue = next(n for n in seg["nodes"] if n["color"] == "B")
    yellow = next(n for n in seg["nodes"] if n["color"] == "Y")
    assert blue["children"] == [yellow["id"]]


def test_adjacency_lists_edge_contacts_once():
    seg = _seg("RRB\nGGB")
    assert seg["adjacency_list"] == [[0, 1], [0, 2], [1, 2]]
    diagonal = _seg("RB\nBR")
    # Diagonal contact does not count; each R touches each B along an edge.
    assert diagonal["adjacency_list"] == [[0, 1], [0, 2], [1, 3], [2, 3]]


def test_numeric_grid_and_ascii_agree_and_values_are_clamped():
    grid = [[5, 5, 9], [5, 20, -3]]
    seg = segment(grid)
    assert [n["color"] for n in seg["nodes"]] == ["B", "b", "p", "W"]
    rows = ["".join(ASCII_ALPHABET[max(0, min(15, v))] for v in row) for row in grid]
    assert segment("\n".join(rows)) == seg
    assert segment(rows) == seg
