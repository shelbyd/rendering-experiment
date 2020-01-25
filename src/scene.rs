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

use crate::geometry::*;

pub struct Scene {
    meshes: Vec<PlacedMesh>,
}

impl Scene {
    pub fn new(meshes: Vec<PlacedMesh>) -> Self {
        Scene { meshes }
    }

    pub fn triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
        self.meshes
            .iter()
            .flat_map(|mesh| mesh.mesh.triangles().map(move |t| t * mesh.transform))
    }
}

pub struct PlacedMesh {
    mesh: TriangleMesh,
    transform: Mat4,
}

impl PlacedMesh {
    pub fn new(mesh: TriangleMesh, transform: Mat4) -> Self {
        PlacedMesh { mesh, transform }
    }
}

pub struct Camera {
    pub transform: Mat4,
    pub projection: Mat4,
}

impl Camera {
    pub fn new(transform: Mat4, projection: Mat4) -> Self {
        Camera {
            transform,
            projection,
        }
    }
}

pub struct TriangleMesh {
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
