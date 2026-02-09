#[must_use]
pub const fn extension_name() -> &'static str {
    "rtree"
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    #[must_use]
    pub const fn contains_point(self, point: Point) -> bool {
        point.x >= self.min_x
            && point.x <= self.max_x
            && point.y >= self.min_y
            && point.y <= self.max_y
    }

    #[must_use]
    pub const fn contains_box(self, other: Self) -> bool {
        other.min_x >= self.min_x
            && other.max_x <= self.max_x
            && other.min_y >= self.min_y
            && other.max_y <= self.max_y
    }
}

#[must_use]
pub fn geopoly_bbox(vertices: &[Point]) -> Option<BoundingBox> {
    let first = *vertices.first()?;
    let mut bounds = BoundingBox {
        min_x: first.x,
        min_y: first.y,
        max_x: first.x,
        max_y: first.y,
    };

    for vertex in vertices.iter().skip(1) {
        bounds.min_x = bounds.min_x.min(vertex.x);
        bounds.min_y = bounds.min_y.min(vertex.y);
        bounds.max_x = bounds.max_x.max(vertex.x);
        bounds.max_y = bounds.max_y.max(vertex.y);
    }

    Some(bounds)
}

#[must_use]
pub fn geopoly_group_bbox(polygons: &[&[Point]]) -> Option<BoundingBox> {
    let mut iter = polygons.iter().filter_map(|polygon| geopoly_bbox(polygon));
    let mut bounds = iter.next()?;

    for next in iter {
        bounds.min_x = bounds.min_x.min(next.min_x);
        bounds.min_y = bounds.min_y.min(next.min_y);
        bounds.max_x = bounds.max_x.max(next.max_x);
        bounds.max_y = bounds.max_y.max(next.max_y);
    }

    Some(bounds)
}

#[must_use]
pub fn geopoly_area(vertices: &[Point]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }

    let mut twice_area = 0.0;
    for index in 0..vertices.len() {
        let current = vertices[index];
        let next = vertices[(index + 1) % vertices.len()];
        twice_area += current.x.mul_add(next.y, -(next.x * current.y));
    }

    twice_area.abs() * 0.5
}

#[must_use]
pub fn geopoly_contains_point(vertices: &[Point], point: Point) -> bool {
    if vertices.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut previous = vertices[vertices.len() - 1];

    for &current in vertices {
        if point_on_segment(previous, current, point) {
            return true;
        }

        let crosses_scanline = (current.y > point.y) != (previous.y > point.y);
        if crosses_scanline {
            let intersection_x = ((previous.x - current.x) * (point.y - current.y)
                / (previous.y - current.y))
                + current.x;
            if point.x < intersection_x {
                inside = !inside;
            }
        }

        previous = current;
    }

    inside
}

#[must_use]
pub fn geopoly_overlap(lhs: &[Point], rhs: &[Point]) -> bool {
    if lhs.len() < 3 || rhs.len() < 3 {
        return false;
    }

    for lhs_index in 0..lhs.len() {
        let lhs_start = lhs[lhs_index];
        let lhs_end = lhs[(lhs_index + 1) % lhs.len()];
        for rhs_index in 0..rhs.len() {
            let rhs_start = rhs[rhs_index];
            let rhs_end = rhs[(rhs_index + 1) % rhs.len()];
            if segments_intersect(lhs_start, lhs_end, rhs_start, rhs_end) {
                return true;
            }
        }
    }

    geopoly_contains_point(lhs, rhs[0]) || geopoly_contains_point(rhs, lhs[0])
}

#[must_use]
pub fn geopoly_within(inner: &[Point], outer: &[Point]) -> bool {
    if inner.len() < 3 || outer.len() < 3 {
        return false;
    }

    let Some(inner_bbox) = geopoly_bbox(inner) else {
        return false;
    };
    let Some(outer_bbox) = geopoly_bbox(outer) else {
        return false;
    };
    if !outer_bbox.contains_box(inner_bbox) {
        return false;
    }

    inner
        .iter()
        .copied()
        .all(|point| geopoly_contains_point(outer, point))
}

#[must_use]
pub fn geopoly_ccw(vertices: &[Point]) -> Vec<Point> {
    if vertices.len() < 3 {
        return vertices.to_vec();
    }

    let mut normalized = vertices.to_vec();
    if signed_twice_area(&normalized) < 0.0 {
        normalized.reverse();
    }
    normalized
}

#[must_use]
pub fn geopoly_regular(center_x: f64, center_y: f64, radius: f64, sides: usize) -> Vec<Point> {
    if sides < 3 || !radius.is_finite() || radius <= 0.0 {
        return Vec::new();
    }

    let Ok(sides_u32) = u32::try_from(sides) else {
        return Vec::new();
    };
    let Ok(capacity) = usize::try_from(sides_u32) else {
        return Vec::new();
    };

    let step = std::f64::consts::TAU / f64::from(sides_u32);
    let mut vertices = Vec::with_capacity(capacity);
    for index in 0..sides_u32 {
        let angle = step * f64::from(index);
        vertices.push(Point::new(
            center_x + (radius * angle.cos()),
            center_y + (radius * angle.sin()),
        ));
    }

    vertices
}

#[must_use]
pub fn geopoly_xform(
    vertices: &[Point],
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
) -> Vec<Point> {
    vertices
        .iter()
        .copied()
        .map(|vertex| {
            Point::new(
                a.mul_add(vertex.x, b.mul_add(vertex.y, e)),
                c.mul_add(vertex.x, d.mul_add(vertex.y, f)),
            )
        })
        .collect()
}

fn segments_intersect(a_start: Point, a_end: Point, b_start: Point, b_end: Point) -> bool {
    let o1 = orientation(a_start, a_end, b_start);
    let o2 = orientation(a_start, a_end, b_end);
    let o3 = orientation(b_start, b_end, a_start);
    let o4 = orientation(b_start, b_end, a_end);

    if o1 != o2 && o3 != o4 {
        return true;
    }

    if o1 == 0 && point_on_segment(a_start, a_end, b_start) {
        return true;
    }
    if o2 == 0 && point_on_segment(a_start, a_end, b_end) {
        return true;
    }
    if o3 == 0 && point_on_segment(b_start, b_end, a_start) {
        return true;
    }
    if o4 == 0 && point_on_segment(b_start, b_end, a_end) {
        return true;
    }

    false
}

fn orientation(start: Point, end: Point, probe: Point) -> i8 {
    let cross =
        (end.y - start.y).mul_add(probe.x - end.x, -((end.x - start.x) * (probe.y - end.y)));
    if cross > f64::EPSILON {
        1
    } else if cross < -f64::EPSILON {
        -1
    } else {
        0
    }
}

fn point_on_segment(start: Point, end: Point, point: Point) -> bool {
    if orientation(start, end, point) != 0 {
        return false;
    }

    point.x >= start.x.min(end.x)
        && point.x <= start.x.max(end.x)
        && point.y >= start.y.min(end.y)
        && point.y <= start.y.max(end.y)
}

fn signed_twice_area(vertices: &[Point]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }

    let mut twice_area = 0.0;
    for index in 0..vertices.len() {
        let current = vertices[index];
        let next = vertices[(index + 1) % vertices.len()];
        twice_area += current.x.mul_add(next.y, -(next.x * current.y));
    }
    twice_area
}

#[cfg(test)]
mod tests {
    use super::{
        Point, extension_name, geopoly_area, geopoly_bbox, geopoly_contains_point,
        geopoly_ccw, geopoly_group_bbox, geopoly_overlap, geopoly_regular, geopoly_within,
        geopoly_xform, signed_twice_area,
    };

    fn approx_eq(left: f64, right: f64) -> bool {
        (left - right).abs() < 1e-9
    }

    fn square(x0: f64, y0: f64, size: f64) -> [Point; 4] {
        [
            Point::new(x0, y0),
            Point::new(x0 + size, y0),
            Point::new(x0 + size, y0 + size),
            Point::new(x0, y0 + size),
        ]
    }

    #[test]
    fn test_extension_name_matches_crate_suffix() {
        let expected = env!("CARGO_PKG_NAME")
            .strip_prefix("fsqlite-ext-")
            .expect("extension crates should use fsqlite-ext-* naming");
        assert_eq!(extension_name(), expected);
    }

    #[test]
    fn test_geopoly_area_triangle() {
        let triangle = [
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(0.0, 3.0),
        ];
        assert!(approx_eq(geopoly_area(&triangle), 6.0));
    }

    #[test]
    fn test_geopoly_area_square() {
        let unit_square = square(0.0, 0.0, 1.0);
        assert!(approx_eq(geopoly_area(&unit_square), 1.0));
    }

    #[test]
    fn test_geopoly_contains_point_inside_and_outside() {
        let polygon = square(0.0, 0.0, 10.0);
        assert!(geopoly_contains_point(&polygon, Point::new(5.0, 5.0)));
        assert!(!geopoly_contains_point(&polygon, Point::new(11.0, 5.0)));
    }

    #[test]
    fn test_geopoly_overlap_true() {
        let lhs = square(0.0, 0.0, 4.0);
        let rhs = square(2.0, 2.0, 4.0);
        assert!(geopoly_overlap(&lhs, &rhs));
    }

    #[test]
    fn test_geopoly_overlap_false() {
        let lhs = square(0.0, 0.0, 2.0);
        let rhs = square(3.0, 3.0, 2.0);
        assert!(!geopoly_overlap(&lhs, &rhs));
    }

    #[test]
    fn test_geopoly_within_contained_and_not_contained() {
        let outer = square(0.0, 0.0, 10.0);
        let inner = square(2.0, 2.0, 3.0);
        let outside = square(-1.0, -1.0, 3.0);
        assert!(geopoly_within(&inner, &outer));
        assert!(!geopoly_within(&outside, &outer));
    }

    #[test]
    fn test_geopoly_bbox_and_group_bbox() {
        let first = square(0.0, 0.0, 1.0);
        let second = square(5.0, -2.0, 2.0);
        let bounds = geopoly_bbox(&second).expect("square should produce bbox");
        assert!(approx_eq(bounds.min_x, 5.0));
        assert!(approx_eq(bounds.min_y, -2.0));
        assert!(approx_eq(bounds.max_x, 7.0));
        assert!(approx_eq(bounds.max_y, 0.0));

        let grouped = geopoly_group_bbox(&[&first, &second]).expect("grouped bbox");
        assert!(approx_eq(grouped.min_x, 0.0));
        assert!(approx_eq(grouped.min_y, -2.0));
        assert!(approx_eq(grouped.max_x, 7.0));
        assert!(approx_eq(grouped.max_y, 1.0));
    }

    #[test]
    fn test_geopoly_regular_hexagon() {
        let hexagon = geopoly_regular(0.0, 0.0, 2.0, 6);
        assert_eq!(hexagon.len(), 6);
        for vertex in &hexagon {
            let distance = (vertex.x * vertex.x + vertex.y * vertex.y).sqrt();
            assert!(approx_eq(distance, 2.0));
        }
    }

    #[test]
    fn test_geopoly_ccw_winding() {
        let clockwise = [
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(1.0, 0.0),
        ];
        assert!(signed_twice_area(&clockwise) < 0.0);
        let ccw = geopoly_ccw(&clockwise);
        assert!(signed_twice_area(&ccw) > 0.0);
    }

    #[test]
    fn test_geopoly_xform_translate() {
        let polygon = square(1.0, 2.0, 1.0);
        let translated = geopoly_xform(&polygon, 1.0, 0.0, 0.0, 1.0, 5.0, -3.0);
        assert!(approx_eq(translated[0].x, 6.0));
        assert!(approx_eq(translated[0].y, -1.0));
    }
}
