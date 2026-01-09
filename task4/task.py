import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Point = Tuple[float, float]
MembershipSets = Dict[str, List[Point]]


def _parse_data(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _load_membership_sets(json_str: str, key_name: str) -> MembershipSets:
    data = _parse_data(json_str)
    if not isinstance(data, dict):
        raise ValueError("Membership description must be a JSON object")
    sets = data.get(key_name)
    if sets is None and len(data) == 1:
        sets = next(iter(data.values()))
    if sets is None:
        raise ValueError(f"Key '{key_name}' not found in membership description")
    result: MembershipSets = {}
    for item in sets:
        term = item.get("id")
        points = item.get("points")
        if term is None or points is None:
            raise ValueError("Each membership function must contain 'id' and 'points'")
        pts: List[Point] = [(float(x), float(y)) for x, y in points]
        pts.sort(key=lambda p: p[0])
        result[str(term)] = pts
    return result


def _membership(points: List[Point], x: float) -> float:
    if not points:
        return 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 == x1:
            if x == x0:
                return max(y0, y1)
            continue
        if (x0 <= x <= x1) or (x1 <= x <= x0):
            t = (x - x0) / (x1 - x0)
            value = y0 + t * (y1 - y0)
            return max(0.0, min(1.0, value))
    return 0.0


def _fuzzify(value: float, sets: MembershipSets) -> Dict[str, float]:
    return {name: _membership(points, value) for name, points in sets.items()}


def _parse_rules(json_str: str) -> List[Tuple[str, str]]:
    data = _parse_data(json_str)
    if not isinstance(data, list):
        raise ValueError("Rule mapping must be a list of pairs")
    rules: List[Tuple[str, str]] = []
    for pair in data:
        if not (isinstance(pair, list) or isinstance(pair, tuple)) or len(pair) != 2:
            raise ValueError("Each rule must be a pair [temperature_term, control_term]")
        rules.append((str(pair[0]), str(pair[1])))
    return rules


def _domain(all_sets: Iterable[List[Point]]) -> Tuple[float, float]:
    xs: List[float] = []
    for pts in all_sets:
        xs.extend(x for x, _ in pts)
    if not xs:
        return 0.0, 1.0
    return min(xs), max(xs)


def _aggregate_output(
    temperature_membership: Dict[str, float],
    control_sets: MembershipSets,
    rules: List[Tuple[str, str]],
    samples: int = 400,
) -> Tuple[List[float], List[float]]:
    xmin, xmax = _domain(control_sets.values())
    if xmax == xmin:
        xmax = xmin + 1.0
    step = (xmax - xmin) / samples
    xs = [xmin + i * step for i in range(samples + 1)]

    def mu_out(x: float) -> float:
        mu_values: List[float] = []
        for temp_term, control_term in rules:
            alpha = temperature_membership.get(temp_term, 0.0)
            pts = control_sets.get(control_term)
            if pts is None or alpha <= 0.0:
                continue
            mu_values.append(min(alpha, _membership(pts, x)))
        return max(mu_values) if mu_values else 0.0

    mus = [mu_out(x) for x in xs]
    return xs, mus


def _centroid(xs: List[float], mus: List[float]) -> float:
    if not xs or not mus or len(xs) != len(mus):
        return 0.0
    area = 0.0
    moment = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        m0, m1 = mus[i], mus[i + 1]
        dx = x1 - x0
        trapezoid_area = (m0 + m1) * dx / 2.0
        trapezoid_moment = (m0 * x0 + m1 * x1) * dx / 2.0
        area += trapezoid_area
        moment += trapezoid_moment
    if area == 0.0:
        return (xs[0] + xs[-1]) / 2.0
    return moment / area


def main(
    temperature_sets_json: str,
    control_sets_json: str,
    rule_mapping_json: str,
    temperature_value: float,
) -> float:
    temperature_sets = _load_membership_sets(temperature_sets_json, "температура")
    try:
        control_sets = _load_membership_sets(control_sets_json, "уровень нагрева")
    except ValueError:
        control_sets = _load_membership_sets(control_sets_json, "температура")
    rules = _parse_rules(rule_mapping_json)

    temperature_membership = _fuzzify(float(temperature_value), temperature_sets)
    xs, mus = _aggregate_output(temperature_membership, control_sets, rules)
    return _centroid(xs, mus)


def print_result(
    temperature_sets_json: str,
    control_sets_json: str,
    rule_mapping_json: str,
    temperature_value: float,
) -> None:
    result = main(
        temperature_sets_json,
        control_sets_json,
        rule_mapping_json,
        temperature_value,
    )
    print(f"{result:.6f}")


if __name__ == "__main__":
    import sys

    def _read_arg_or_path(arg: str) -> str:
        path = Path(arg)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return arg

    if len(sys.argv) >= 5:
        temp_sets_arg, control_sets_arg, mapping_arg, temp_value_arg = sys.argv[1:5]
        print_result(
            _read_arg_or_path(temp_sets_arg),
            _read_arg_or_path(control_sets_arg),
            _read_arg_or_path(mapping_arg),
            float(temp_value_arg),
        )
    else:
        print(
            "Usage: python task4/task.py "
            "<temp_sets.json|path> <control_sets.json|path> "
            "<rules.json|path> <temperature_value>"
        )

