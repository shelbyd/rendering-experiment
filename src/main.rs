use image::{ImageBuffer, Rgb};
use nalgebra_glm::{self as glm, Mat4, Vec2, Vec3};
use std::error::Error;
use std::io::{stdin, Cursor, Read};
use stl_io::IndexedMesh;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

const SCALE_FACTOR: f32 = 32.0;

type Result<T> = core::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let stdin = full_stdin()?;
    let mesh = parse_mesh(&stdin)?;
    let scene = Scene::new(vec![PlacedMesh::new(
        mesh,
        glm::translation(&glm::vec3(0., 0., -5.)),
    )]);
    let image = render(&scene)?;
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

fn render(scene: &Scene) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    Ok(ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
        let point = {
            let window_coords = Vec2::new(x as f32, y as f32);
            let centered = window_coords - Vec2::new((WIDTH / 2) as f32, (HEIGHT / 2) as f32);
            centered / SCALE_FACTOR
        };
        let contains = scene.triangles().any(|t| t.contains(&point));
        let value = if contains { 255 } else { 0 };
        Rgb([value, value, value])
    }))
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
            .map(|face| Triangle {
                normal: Vec3::from(face.normal),
                vertices: [
                    Vec3::from(indexed.vertices[face.vertices[0]]),
                    Vec3::from(indexed.vertices[face.vertices[1]]),
                    Vec3::from(indexed.vertices[face.vertices[2]]),
                ],
            })
            .collect();

        TriangleMesh { triangles }
    }
}

struct Triangle {
    normal: Vec3,
    vertices: [Vec3; 3],
}

impl Triangle {
    fn contains(&self, p: &Vec2) -> bool {
        let p0 = self.vertices[0].xz();
        let p1 = self.vertices[1].xz();
        let p2 = self.vertices[2].xz();

        let A = 0.5 * (-p1.y * p2.x + p0.y * (-p1.x + p2.x) + p0.x * (p1.y - p2.y) + p1.x * p2.y);
        let sign = if A < 0.0 { -1.0 } else { 1.0 };
        let s = (p0.y * p2.x - p0.x * p2.y + (p2.y - p0.y) * p.x + (p0.x - p2.x) * p.y) * sign;
        let t = (p0.x * p1.y - p0.y * p1.x + (p0.y - p1.y) * p.x + (p1.x - p0.x) * p.y) * sign;
        s > 0.0 && t > 0.0 && (s + t) < 2.0 * A * sign
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
        Triangle {
            normal: mul(self.normal),
            vertices: [
                mul(self.vertices[0]),
                mul(self.vertices[1]),
                mul(self.vertices[2]),
            ],
        }
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
