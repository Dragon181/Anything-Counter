from typing import Optional

from anything_counter.anything_counter.models import Line, Point


def intersection_over_union(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def ccw(a: Point, b: Point, c: Point) -> int:
    return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)


def is_intersecting(line1: Line, line2: Line) -> Optional[int]:

    o1 = ccw(line1.start, line1.end, line2.start)
    o2 = ccw(line1.start, line1.end, line2.end)
    o3 = ccw(line2.start, line2.end, line1.start)
    o4 = ccw(line2.start, line2.end, line1.end)

    if o1 * o2 < 0 and o3 * o4 < 0:
        direction = 1 if o1 * o2 > 0 else -1
        return direction
    return None
