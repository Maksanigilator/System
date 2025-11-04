from typing import List, Tuple, Dict, Set


def _parse_edges(csv_edges: str) -> List[Tuple[int, int]]:

    text = csv_edges.strip()
    # Support literals like "1,2\n1,3" passed via CLI quotes
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    lines: List[str] = [line.strip() for line in text.splitlines() if line.strip()]
    edges: List[Tuple[int, int]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 2:
            raise ValueError(f"Invalid edge line: '{line}'")
        try:
            parent = int(parts[0])
            child = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Non-integer vertex id in line: '{line}'") from exc
        edges.append((parent, child))
    return edges


def _build_index_map(edges: List[Tuple[int, int]], root_id: int) -> Tuple[Dict[int, int], List[int]]:

    vertices: Set[int] = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    vertices.add(root_id)

    ordered_vertices: List[int] = sorted(vertices)
    index_by_vertex: Dict[int, int] = {v: i for i, v in enumerate(ordered_vertices)}
    return index_by_vertex, ordered_vertices


def _empty_bool_matrix(n: int) -> List[List[bool]]:

    return [[False for _ in range(n)] for _ in range(n)]


def _transpose_bool_matrix(m: List[List[bool]]) -> List[List[bool]]:

    n = len(m)
    return [[m[j][i] for j in range(n)] for i in range(n)]


def _reachability(adjacency: List[List[int]]) -> List[List[bool]]:

    n = len(adjacency)
    reach: List[List[bool]] = [[bool(adjacency[i][j]) for j in range(n)] for i in range(n)]

    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                row_k = reach[k]
                row_i = reach[i]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True
    return reach


def main(s: str, e: str) -> Tuple[List[List[bool]], List[List[bool]], List[List[bool]], List[List[bool]], List[List[bool]]]:

    edges: List[Tuple[int, int]] = _parse_edges(s)
    root_id: int = int(e)

    index_by_vertex, ordered_vertices = _build_index_map(edges, root_id)
    n: int = len(ordered_vertices)

    r1: List[List[bool]] = _empty_bool_matrix(n)
    children_by_parent: Dict[int, List[int]] = {}

    for u, v in edges:
        iu = index_by_vertex[u]
        iv = index_by_vertex[v]
        r1[iu][iv] = True
        children_by_parent.setdefault(u, []).append(v)

    r2: List[List[bool]] = _transpose_bool_matrix(r1)

    reach: List[List[bool]] = _reachability(r1)

    r3: List[List[bool]] = _empty_bool_matrix(n)
    for i in range(n):
        for j in range(n):
            if reach[i][j] and not r1[i][j]:
                r3[i][j] = True

    r4: List[List[bool]] = _transpose_bool_matrix(r3)

    r5: List[List[bool]] = _empty_bool_matrix(n)
    for parent, children in children_by_parent.items():
        idx_children = [index_by_vertex[c] for c in children]
        for i in range(len(idx_children)):
            for j in range(len(idx_children)):
                if i == j:
                    continue
                r5[idx_children[i]][idx_children[j]] = True

    return r1, r2, r3, r4, r5


def _format_matrix(matrix: List[List[bool]], headers: List[int]) -> str:

    n = len(matrix)
    head = [" "] + [str(h) for h in headers]
    lines: List[str] = ["\t".join(head)]
    for i in range(n):
        row_vals = ["1" if cell else "0" for cell in matrix[i]]
        lines.append("\t".join([str(headers[i])] + row_vals))
    return "\n".join(lines)


def print_result(s: str, e: str) -> None:

    edges = _parse_edges(s)
    _, ordered_vertices = _build_index_map(edges, int(e))
    r1, r2, r3, r4, r5 = main(s, e)

    labels = [
        ("r1 — непосредственное управление", r1),
        ("r2 — непосредственное подчинение", r2),
        ("r3 — опосредованное управление", r3),
        ("r4 — опосредованное подчинение", r4),
        ("r5 — соподчинение (один родитель)", r5),
    ]

    for title, m in labels:
        print(title)
        print(_format_matrix(m, ordered_vertices))
        print()


if __name__ == "__main__":

    import sys
    from pathlib import Path

    if len(sys.argv) >= 3:
        path = Path(sys.argv[1])
        root = sys.argv[2]
        if path.exists():
            content = path.read_text(encoding="utf-8")
        else:
            content = sys.argv[1]
        print_result(content, root)
    else:
        try:
            default_path = Path("task0/task2.csv")
            content = default_path.read_text(encoding="utf-8")
            print_result(content, "1")
        except Exception as exc:
            print("Usage: python -m task1.task <edges.csv|\"1,2\\n1,3...\"> <root_id>")
            print(f"Error: {exc}")

