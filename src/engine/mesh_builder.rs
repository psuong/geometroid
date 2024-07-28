use std::u16;

use cgmath::{InnerSpace, Vector3};

use super::render::Vertex;

pub struct MeshBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl MeshBuilder {
    /// Creates a new mesh builder allocating a Vector of vertices and indices.
    /// Note: for every 4 vertices (quad), you have 6 indices
    /// # Arguments
    ///
    /// * `vertex_count` - The total number of vertices to allocate
    pub fn with_capacity(vertex_count: u16) -> Self {
        let vertices = Vec::with_capacity(vertex_count.into());
        let index_count = (vertex_count / 4) * 6;
        let indices = Vec::with_capacity(index_count.into());

        MeshBuilder { vertices, indices }
    }

    pub fn push_vertex(&mut self, vertex: Vertex) -> &MeshBuilder {
        self.vertices.push(vertex);
        self
    }

    pub fn push_quad(&mut self, offset: Vector3<f32>, width: Vector3<f32>, length: Vector3<f32>) -> &MeshBuilder {
        let normal = width.cross(length).normalize();
        self
    }
}
