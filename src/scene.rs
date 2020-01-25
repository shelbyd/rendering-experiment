use nalgebra_glm::{Mat4, Vec3};
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
