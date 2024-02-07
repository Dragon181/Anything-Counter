import pytest

from anything_counter.anything_counter.models import Line, Point
from anything_counter.utils.functions import is_intersecting, ccw


def test_ccw():
    a = Point(0, 0)
    b = Point(1, 1)
    c = Point(2, 2)
    expected_result = 0
    assert ccw(a, b, c) == expected_result


@pytest.mark.parametrize(
    'line1, line2, expected_result', [
        (Line(Point(0, 0), Point(2, 2)), Line(Point(1, 0), Point(1, 2)), -1),
        (Line(Point(1, 0), Point(1, 2)), Line(Point(0, 0), Point(2, 2)), -1),
        (Line(Point(0, 0), Point(1, 1)), Line(Point(2, 2), Point(3, 3)), None)
    ])
def test_is_intersecting(line1, line2, expected_result):
    assert is_intersecting(line1, line2) == expected_result
