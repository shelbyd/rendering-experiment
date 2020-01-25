use image::{ImageBuffer, Rgb};
use indicatif::ParallelProgressIterator;
use nalgebra_glm::{self as glm, Mat4, Vec2};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::convert::TryInto;
use std::error::Error;
use std::io::{stdin, Cursor, Read};
use std::sync::Mutex;

mod geometry;
use crate::geometry::*;

mod scene;
use crate::scene::*;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

type Result<T> = core::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let stdin = full_stdin()?;
    let mesh = parse_mesh(&stdin)?;
    let scene = Scene::new(vec![PlacedMesh::new(
        mesh,
        glm::translation(&glm::vec3(0., 0., 0.))
            * glm::rotation(glm::half_pi(), &glm::vec3(1., 0., 0.))
            * glm::scaling(&glm::vec3(0.05, 0.05, 0.05)),
    )]);

    let mut camera = Camera::new(
        glm::look_at(
            &glm::vec3(0., 2., 0.),
            &glm::vec3(0., 0., 0.),
            &glm::vec3(0., 1., 0.),
        ),
        glm::perspective(
            WIDTH as f32 / HEIGHT as f32, // aspect
            glm::half_pi::<f32>() * 0.8,  // fovy
            0.1,                          // near clipping plane
            100.,                         // far clipping plane
        ),
    );

    for z in 0..1 {
        let z = z as f32 * 100.;
        camera.transform = glm::look_at(
            &glm::vec3(0., 2., z + 10.),
            &glm::vec3(0., 0., 0.),
            &glm::vec3(0., 1., 0.),
        );

        let image = RayTracing.render(&camera, &scene);
        image.save_with_format("/tmp/image.png", image::ImageFormat::PNG)?;
    }
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

trait Renderer {
    fn render(&self, camera: &Camera, scene: &Scene) -> ImageBuffer<Rgb<u8>, Vec<u8>>;
}

struct RayTracing;

impl RayTracing {
    fn render_point(&self, x: u32, y: u32, finder: &impl TriangleFinder) -> Rgb<u8> {
        let point = {
            let window_coords = Vec2::new(x as f32 / WIDTH as f32, y as f32 / HEIGHT as f32);
            let negative_one_to_one = (window_coords - Vec2::new(0.5, 0.5)) * 2.;
            negative_one_to_one
        };

        let value = match finder.triangle_at(point) {
            true => 255,
            false => 0,
        };
        Rgb([value, value, value])
    }
}

impl Renderer for RayTracing {
    fn render(&self, camera: &Camera, scene: &Scene) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let world_to_projection = camera.projection * camera.transform;
        let projection_vec = ProjectionVec::new(world_to_projection, scene);
        let finder = LineOrientedFinder::new(projection_vec, 20, Axis::Y);

        let buffer = Mutex::new(ImageBuffer::from_pixel(WIDTH, HEIGHT, Rgb([0, 0, 0])));

        let height = buffer.lock().unwrap().height();
        (0..height).into_par_iter().progress().for_each(|y| {
            let mut row_buffer = Vec::with_capacity(WIDTH.try_into().unwrap());
            row_buffer.resize(WIDTH as usize, Rgb([0, 0, 0]));

            for (x, pixel) in row_buffer.iter_mut().enumerate() {
                *pixel = self.render_point(x.try_into().unwrap(), y, &finder);
            }

            let mut buffer = buffer.lock().unwrap();
            for (x, pixel) in row_buffer.into_iter().enumerate() {
                buffer.put_pixel(x.try_into().unwrap(), y, pixel);
            }
        });

        buffer.into_inner().unwrap()
    }
}

trait TriangleFinder {
    fn triangle_at(&self, point: Vec2) -> bool;
}

struct ProjectionVec(Vec<Triangle>);

impl ProjectionVec {
    fn new(world_to_projection: Mat4, scene: &Scene) -> Self {
        let projection_triangles: Vec<_> = scene
            .triangles()
            .map(|t| &t * world_to_projection)
            .collect();
        ProjectionVec(projection_triangles)
    }
}

impl TriangleFinder for ProjectionVec {
    fn triangle_at(&self, point: Vec2) -> bool {
        self.0.iter().any(|t| t.contains(point))
    }
}

struct LineOrientedFinder {
    map: BTreeMap<OrderedFloat<f32>, ProjectionVec>,
    axis: Axis,
}

impl LineOrientedFinder {
    fn new(full_projection_vec: ProjectionVec, triangles_per_slice: usize, axis: Axis) -> Self {
        let mut bounding_box_center_scalars: Vec<_> = full_projection_vec
            .0
            .iter()
            .map(|t| OrderedFloat(axis.from_point(t.xy_bounding_box().center())))
            .collect();
        bounding_box_center_scalars.sort();

        let lines = bounding_box_center_scalars
            .iter()
            .enumerate()
            .filter(|(index, _)| index % triangles_per_slice == 0)
            .map(|tuple| tuple.1)
            .chain(bounding_box_center_scalars.last().into_iter())
            .collect::<Vec<_>>();

        let mut map = BTreeMap::new();
        for range in lines.windows(2) {
            let (start, end) = (range[0].0, range[1].0);
            let relevant_triangles = full_projection_vec
                .0
                .iter()
                .filter(|t| t.xy_bounding_box().axis(axis).intersects(start, end))
                .cloned()
                .collect();

            map.insert(
                OrderedFloat((start + end) / 2.),
                ProjectionVec(relevant_triangles),
            );
        }

        LineOrientedFinder { map, axis }
    }
}

impl TriangleFinder for LineOrientedFinder {
    fn triangle_at(&self, point: Vec2) -> bool {
        let scalar = self.axis.from_point(point);

        let above = self.map.range(OrderedFloat(scalar)..).map(|t| t.1).next();
        let below = self
            .map
            .range(..OrderedFloat(scalar))
            .map(|t| t.1)
            .next_back();

        if let Some(above) = above {
            if above.triangle_at(point) {
                return true;
            }
        }
        if let Some(below) = below {
            if below.triangle_at(point) {
                return true;
            }
        }

        false
    }
}
