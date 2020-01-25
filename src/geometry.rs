use image::{ImageBuffer, Rgb};
use indicatif::ParallelProgressIterator;
use nalgebra_glm::{self as glm, Mat4, Vec2, Vec3};
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::convert::TryInto;
use std::error::Error;
use std::io::{stdin, Cursor, Read};
use std::sync::Mutex;
use stl_io::IndexedMesh;

#[derive(Debug, Clone)]
pub struct Triangle {
    normal: Vec3,
    vertices: [Vec3; 3],
}

impl Triangle {
    pub fn new(normal: Vec3, vertices: [Vec3; 3]) -> Triangle {
        Triangle { normal, vertices }
    }

    pub fn contains(&self, p: Vec2) -> bool {
        let p0 = self.vertices[0];
        let p1 = self.vertices[1];
        let p2 = self.vertices[2];

        let area =
            0.5 * (-p1.y * p2.x + p0.y * (-p1.x + p2.x) + p0.x * (p1.y - p2.y) + p1.x * p2.y);
        let sign = if area < 0.0 { -1.0 } else { 1.0 };
        let s = (p0.y * p2.x - p0.x * p2.y + (p2.y - p0.y) * p.x + (p0.x - p2.x) * p.y) * sign;
        let t = (p0.x * p1.y - p0.y * p1.x + (p0.y - p1.y) * p.x + (p1.x - p0.x) * p.y) * sign;
        s > 0.0 && t > 0.0 && (s + t) < 2.0 * area * sign
    }

    pub fn xy_bounding_box(&self) -> Rect {
        Rect::covering(self.vertices.iter().map(|v| v.xy()))
    }
}

impl core::ops::Mul<Mat4> for &Triangle {
    type Output = Triangle;

    fn mul(self, rhs: Mat4) -> Triangle {
        let mul = |vec: Vec3| {
            let mut vec4 = glm::vec3_to_vec4(&vec);
            vec4.w = 1.;
            glm::vec4_to_vec3(&(rhs * vec4))
        };
        Triangle::new(
            mul(self.normal),
            [
                mul(self.vertices[0]),
                mul(self.vertices[1]),
                mul(self.vertices[2]),
            ],
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    top: f32,
    bottom: f32,
    left: f32,
    right: f32,
}

impl Rect {
    fn covering(mut vecs: impl Iterator<Item = Vec2>) -> Rect {
        let mut rect = match vecs.next() {
            None => Rect::default(),
            Some(v) => Rect {
                top: v.y,
                bottom: v.y,
                left: v.x,
                right: v.x,
            },
        };
        for vec in vecs {
            rect.cover(vec);
        }
        rect
    }

    fn cover(&mut self, vec: Vec2) {
        if vec.x > self.right {
            self.right = vec.x;
        }
        if vec.x < self.left {
            self.left = vec.x;
        }

        if vec.y > self.top {
            self.top = vec.y;
        }
        if vec.y < self.bottom {
            self.bottom = vec.y;
        }
    }

    fn contains(&self, vec: &Vec2) -> bool {
        self.axis(Axis::X).contains(vec.x) && self.axis(Axis::Y).contains(vec.y)
    }

    pub fn axis(&self, axis: Axis) -> LineSegment {
        match axis {
            Axis::X => LineSegment::new(self.left, self.right),
            Axis::Y => LineSegment::new(self.bottom, self.top),
        }
    }

    pub fn center(&self) -> Vec2 {
        glm::vec2((self.right + self.left) / 2., (self.top + self.bottom) / 2.)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    begin: f32,
    end: f32,
}

impl LineSegment {
    fn new(begin: f32, end: f32) -> Self {
        debug_assert!(begin <= end);
        LineSegment { begin, end }
    }

    pub fn contains(&self, scalar: f32) -> bool {
        self.begin <= scalar && self.end >= scalar
    }

    pub fn intersects(&self, begin: f32, end: f32) -> bool {
        debug_assert!(begin <= end);
        if end < self.begin {
            return false;
        }
        if begin > self.end {
            return false;
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Axis {
    X,
    Y,
}

impl Axis {
    pub fn from_point(self, point: Vec2) -> f32 {
        match self {
            Self::X => point.x,
            Self::Y => point.y,
        }
    }
}
