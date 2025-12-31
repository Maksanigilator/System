import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Matrix = List[List[int]]


def _is_matrix(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if not value:
        return False
    if not all(isinstance(row, list) for row in value):
        return False
    n = len(value)
    return all(len(row) == n and all(cell in (0, 1, True, False) for cell in row) for row in value)


def _flatten_clusters(data: Sequence[Any]) -> List[Any]:
    """Return flat list of labels preserving first occurrence order."""
    seen = set()
    flat: List[Any] = []
    for item in data:
        if isinstance(item, (list, tuple)):
            for sub in item:
                if sub not in seen:
                    flat.append(sub)
                    seen.add(sub)
        else:
            if item not in seen:
                flat.append(item)
                seen.add(item)
    return flat


def _matrix_from_clusters(clusters: Sequence[Any]) -> Tuple[Matrix, List[str]]:
    """
    Build relation matrix from cluster order.
    clusters: e.g. ["a", ["b", "c"], "d"] means a < {b, c} < d.
    Returns (matrix, labels_in_order).
    """
    # Normalize clusters into list of lists
    norm_clusters: List[List[Any]] = []
    for item in clusters:
        if isinstance(item, (list, tuple)):
            norm_clusters.append(list(item))
        else:
            norm_clusters.append([item])

    labels: List[str] = [str(v) for v in _flatten_clusters(norm_clusters)]
    index: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    cluster_by_label: Dict[str, int] = {}
    for c_idx, cluster in enumerate(norm_clusters):
        for value in cluster:
            cluster_by_label[str(value)] = c_idx

    n = len(labels)
    m: Matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            ci = cluster_by_label[li]
            cj = cluster_by_label[lj]
            if ci < cj:
                # lj is to the right (better) than li -> li <= lj
                m[i][j] = 1
            elif ci == cj:
                # same cluster, objects are indistinguishable
                m[i][j] = 1
            else:
                # ci > cj -> li is to the right of lj, so m[i][j] = 0
                m[i][j] = 0
    return m, labels


def _parse_ranking(json_str: str) -> Tuple[Matrix, List[str]]:
    """
    Parse ranking JSON. Supports two shapes:
      1) Square 0/1 matrix: [[1,1,...], ...]
      2) Cluster order: ["a", ["b","c"], "d"] or {"ranking": [...]}
    Returns (matrix, labels_as_strings).
    """
    data = json.loads(json_str)
    if isinstance(data, dict):
        data = data.get("ranking") or data.get("clusters") or data.get("data")
    if data is None:
        raise ValueError("Ranking JSON is empty or missing 'ranking'/'clusters' key")

    if _is_matrix(data):
        matrix: Matrix = [[1 if cell else 0 for cell in row] for row in data]  # type: ignore
        labels = [str(i + 1) for i in range(len(matrix))]
        return matrix, labels

    if isinstance(data, list):
        return _matrix_from_clusters(data)

    raise ValueError("Unsupported ranking format; expected matrix or cluster list")


def _transpose(m: Matrix) -> Matrix:
    return [list(row) for row in zip(*m)]


def _hadamard(a: Matrix, b: Matrix) -> Matrix:
    n = len(a)
    return [[1 if a[i][j] and b[i][j] else 0 for j in range(n)] for i in range(n)]


def _or(a: Matrix, b: Matrix) -> Matrix:
    n = len(a)
    return [[1 if a[i][j] or b[i][j] else 0 for j in range(n)] for i in range(n)]


def _warshall_closure(m: Matrix) -> Matrix:
    n = len(m)
    closure = [[1 if i == j or m[i][j] else 0 for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            if closure[i][k]:
                row_i = closure[i]
                row_k = closure[k]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = 1
    return closure


def _components(eq: Matrix) -> List[List[int]]:
    n = len(eq)
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: List[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for j in range(n):
                if not seen[j] and eq[v][j] and eq[j][v]:
                    seen[j] = True
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


def _order_clusters(clusters: List[List[int]], c: Matrix) -> List[List[int]]:
    """
    Order clusters using matrix C.
    If cluster A outranks B (exists a in A, b in B: c[a][b]==1 and c[b][a]==0) add edge B->A.
    Topologically sort; if cycle, keep initial order.
    """
    k = len(clusters)
    edges: List[Tuple[int, int]] = []
    indeg = [0] * k

    def outranks(idx_a: int, idx_b: int) -> bool:
        for a in clusters[idx_a]:
            for b in clusters[idx_b]:
                if c[a][b] and not c[b][a]:
                    return True
        return False

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            a_over_b = outranks(i, j)
            b_over_a = outranks(j, i)
            if a_over_b and not b_over_a:
                edges.append((j, i))  # j -> i (i is better)

    for u, v in edges:
        indeg[v] += 1

    # Kahn's algorithm
    queue: List[int] = [i for i in range(k) if indeg[i] == 0]
    ordered: List[int] = []
    while queue:
        u = queue.pop(0)
        ordered.append(u)
        for x, y in edges:
            if x == u:
                indeg[y] -= 1
                if indeg[y] == 0:
                    queue.append(y)

    if len(ordered) != k:
        # Cycle detected; fallback to original order
        ordered = list(range(k))

    return [clusters[i] for i in ordered]


def main(ranking_a_json: str, ranking_b_json: str) -> str:
    """
    Build contradiction core and consensus cluster ranking.
    Args:
        ranking_a_json: JSON string with ranking A
        ranking_b_json: JSON string with ranking B
    Returns:
        JSON string with fields:
          core: list of [obj_i, obj_j] pairs in contradiction
          consensus: ordered list of clusters (each cluster is list of labels)
    """
    ya, labels_a = _parse_ranking(ranking_a_json)
    yb, labels_b = _parse_ranking(ranking_b_json)
    if labels_a != labels_b:
        raise ValueError("Rankings must refer to the same set of objects in the same order")

    n = len(ya)
    if n != len(yb):
        raise ValueError("Matrices must be the same size")

    ya_t = _transpose(ya)
    yb_t = _transpose(yb)

    # Agreement matrices
    y_ab = _hadamard(ya, yb)
    y_ab_t = _hadamard(ya_t, yb_t)

    # Matrix to detect contradictions: 1 where at least one orientation is consistent
    p = _or(y_ab, y_ab_t)
    core: List[Tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if p[i][j] == 0 and p[j][i] == 0:
                core.append((labels_a[i], labels_a[j]))

    # Consensus order matrix
    c = [row[:] for row in y_ab]
    for a, b in core:
        ia = labels_a.index(a)
        ib = labels_a.index(b)
        c[ia][ib] = 1
        c[ib][ia] = 1

    e = _hadamard(c, _transpose(c))
    e_closure = _warshall_closure(e)

    comps_idx = _components(e_closure)
    ordered_clusters_idx = _order_clusters(comps_idx, c)
    ordered_clusters_labels: List[List[str]] = [
        [labels_a[i] for i in cluster] for cluster in ordered_clusters_idx
    ]

    result = {
        "core": [[a, b] for a, b in core],
        "consensus": ordered_clusters_labels,
    }
    return json.dumps(result, ensure_ascii=False)


def print_result(ranking_a_json: str, ranking_b_json: str) -> None:
    print(main(ranking_a_json, ranking_b_json))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) >= 3:
        path_a = Path(sys.argv[1])
        path_b = Path(sys.argv[2])
        if path_a.exists():
            content_a = path_a.read_text(encoding="utf-8")
        else:
            content_a = sys.argv[1]
        if path_b.exists():
            content_b = path_b.read_text(encoding="utf-8")
        else:
            content_b = sys.argv[2]
        print_result(content_a, content_b)
    else:
        example_a = json.dumps(["A", ["B", "C"], "D"])
        example_b = json.dumps([["A", "B"], "C", "D"])
        print(
            "Usage: python -m task3.task <ranking_a.json or path> <ranking_b.json or path>"
        )
        print("Example output with demo rankings:")
        print_result(example_a, example_b)

