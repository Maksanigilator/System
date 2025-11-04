from typing import List, Tuple
import math


def _count_true(matrix: List[List[bool]]) -> int:

    return sum(1 for row in matrix for cell in row if cell)


def _probabilities(counts: List[int]) -> List[float]:

    total = float(sum(counts))
    if total == 0.0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def _entropy_base2(probabilities: List[float]) -> float:

    h = 0.0
    for p in probabilities:
        if p > 0.0:
            h -= p * math.log(p, 2)
    return h


def _get_builder():

    try:
        from task1.task import main as build_predicate_matrices
        return build_predicate_matrices
    except Exception:
        # Fallback when executed as a standalone script
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from task1.task import main as build_predicate_matrices
        return build_predicate_matrices


def main(s: str, e: str) -> Tuple[float, float]:

    # Reuse predicate matrices from task1
    build_predicate_matrices = _get_builder()

    r1, r2, r3, r4, r5 = build_predicate_matrices(s, e)

    counts = [
        _count_true(r1),  # direct management
        _count_true(r2),  # direct subordination
        _count_true(r3),  # indirect management
        _count_true(r4),  # indirect subordination
        _count_true(r5),  # co-subordination (same parent)
    ]

    ps = _probabilities(counts)
    h = _entropy_base2(ps)

    # Normalized by maximum entropy over 5 categories
    h_max = math.log(5, 2)
    h_norm = 0.0 if h_max == 0.0 else h / h_max

    return h, h_norm


def print_result(s: str, e: str) -> None:

    build_predicate_matrices = _get_builder()

    r1, r2, r3, r4, r5 = build_predicate_matrices(s, e)
    counts = [
        _count_true(r1),
        _count_true(r2),
        _count_true(r3),
        _count_true(r4),
        _count_true(r5),
    ]
    ps = _probabilities(counts)
    h, h_norm = main(s, e)

    labels = ["r1", "r2", "r3", "r4", "r5"]
    print("Counts per relation:")
    for name, c, p in zip(labels, counts, ps):
        print(f"  {name}: {c} ({p:.6f})")
    print(f"Entropy (bits): {h:.6f}")
    print(f"Normalized entropy: {h_norm:.6f}")


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
            print("Usage: python -m task2.task <edges.csv|\"1,2\\n1,3...\"> <root_id>")
            print(f"Error: {exc}")


