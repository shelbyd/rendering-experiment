use image::{ImageBuffer, Rgb};
use indicatif::ParallelProgressIterator;
use nalgebra_glm::{self as glm, Mat4, Vec2, Vec3};
use rayon::prelude::*;
use std::convert::TryInto;
use std::error::Error;
use std::io::{stdin, Cursor, Read};
use std::sync::Mutex;
use stl_io::IndexedMesh;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

type Result<T> = core::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let stdin = full_stdin()?;
    let mesh = parse_mesh(&stdin)?;
    let scene = Scene::new(vec![PlacedMesh::new(
        mesh,
        glm::translation(&glm::vec3(0., 0., -5.)) * glm::scaling(&glm::vec3(0.1, 0.1, 0.1)),
    )]);
    let camera = Camera::new(
        glm::look_at(
            &glm::vec3(0., 2., 0.),
            &glm::vec3(0., 0., -5.),
            &glm::vec3(0., 1., 0.),
        ),
        glm::perspective(
            WIDTH as f32 / HEIGHT as f32, // aspect
            glm::half_pi(),               // fovy
            0.01,                         // near clipping plane
            10000.,                       // far clipping plane
        ),
    );
    let image = camera.render(&scene);
    image.save_with_format("/dev/stdout", image::ImageFormat::PNG)?;
    Ok(())
}

fn full_stdin() -> Result<Vec<u8>> {
    let mut vec = Vec::new();
    stdin().lock().read_to_end(&mut vec)?;
    Ok(vec)
}

fn parse_mesh(bytes: &[u8]) -> Result<TriangleMesh> {
    let mut reader = Cursor::new(bytes);
    let mesh = stl_io::read_stl(&mut reader)?;
    mesh.validate()?;
    Ok(TriangleMesh::from(mesh))
}

struct TriangleMesh {
    triangles: Vec<Triangle>,
}

impl TriangleMesh {
    fn triangles(&self) -> impl Iterator<Item = &Triangle> {
        self.triangles.iter()
    }
}

impl From<IndexedMesh> for TriangleMesh {
    fn from(indexed: IndexedMesh) -> Self {
        let triangles = indexed
            .faces
            .iter()
            .map(|face| {
                Triangle::new(
                    Vec3::from(face.normal),
                    [
                        Vec3::from(indexed.vertices[face.vertices[0]]),
                        Vec3::from(indexed.vertices[face.vertices[1]]),
                        Vec3::from(indexed.vertices[face.vertices[2]]),
                    ],
                )
            })
            .collect();

        TriangleMesh { triangles }
    }
}

#[derive(Debug)]
struct Triangle {
    normal: Vec3,
    vertices: [Vec3; 3],
}

impl Triangle {
    fn new(normal: Vec3, vertices: [Vec3; 3]) -> Triangle {
        Triangle { normal, vertices }
    }

    fn contains(&self, p: &Vec2) -> bool {
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

    fn xy_bounding_box(&self) -> Rect {
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
struct Rect {
    top: f32,
    bottom: f32,
    left: f32,
    right: f32,
}

impl Rect {
    fn covering(vecs: impl Iterator<Item = Vec2>) -> Rect {
        let mut rect = Rect::default();
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
        let horizontal = self.left <= vec.x || self.right >= vec.x;
        let vertical = self.bottom <= vec.y || self.top >= vec.y;
        horizontal && vertical
    }
}

struct Scene {
    meshes: Vec<PlacedMesh>,
}

impl Scene {
    fn new(meshes: Vec<PlacedMesh>) -> Self {
        Scene { meshes }
    }

    fn triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
        self.meshes
            .iter()
            .flat_map(|mesh| mesh.mesh.triangles().map(move |t| t * mesh.transform))
    }
}

struct PlacedMesh {
    mesh: TriangleMesh,
    transform: Mat4,
}

impl PlacedMesh {
    fn new(mesh: TriangleMesh, transform: Mat4) -> Self {
        PlacedMesh { mesh, transform }
    }
}

struct Camera {
    transform: Mat4,
    projection: Mat4,
}

impl Camera {
    fn new(transform: Mat4, projection: Mat4) -> Self {
        Camera {
            transform,
            projection,
        }
    }

    fn render(&self, scene: &Scene) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let world_to_projection = self.projection * self.transform;
        let projection_triangles: Vec<_> = scene
            .triangles()
            .map(|t| &t * world_to_projection)
            .collect();

        let buffer = Mutex::new(ImageBuffer::from_pixel(WIDTH, HEIGHT, Rgb([0, 0, 0])));

        let height = buffer.lock().unwrap().height();
        (0..height).into_par_iter().progress().for_each(|y| {
            let mut row_buffer = Vec::with_capacity(WIDTH.try_into().unwrap());
            row_buffer.resize(WIDTH as usize, Rgb([0, 0, 0]));

            for (x, pixel) in row_buffer.iter_mut().enumerate() {
                *pixel = self.render_point(x.try_into().unwrap(), y, &projection_triangles);
            }

            let mut buffer = buffer.lock().unwrap();
            for (x, pixel) in row_buffer.into_iter().enumerate() {
                buffer.put_pixel(x.try_into().unwrap(), y, pixel);
            }
        });

        buffer.into_inner().unwrap()
    }

    fn render_point(&self, x: u32, y: u32, projection_triangles: &[Triangle]) -> Rgb<u8> {
        let point = {
            let window_coords = Vec2::new(x as f32 / WIDTH as f32, y as f32 / HEIGHT as f32);
            let negative_one_to_one = (window_coords - Vec2::new(0.5, 0.5)) * 2.;
            negative_one_to_one
        };

        let contains = projection_triangles.iter().any(|t| t.contains(&point));

        let value = if contains { 255 } else { 0 };
        Rgb([value, value, value])
    }
}
