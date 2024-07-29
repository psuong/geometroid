use std::u16;
use cgmath::{InnerSpace, Vector2, Vector3, Zero};

use super::render::Vertex;

pub struct MeshBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
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

    pub fn push_quad(
        &mut self,
        offset: Vector3<f32>,
        width: Vector3<f32>,
        length: Vector3<f32>,
        color: Vector3<f32>,
    ) -> &MeshBuilder {
        let normal = width.cross(length).normalize();

        self.vertices.push(Vertex {
            position: offset,
            normal,
            color,
            uv: Vector2::new(0.0 as f32, 0.0 as f32),
        });

        self.vertices.push(Vertex {
            position: offset + length,
            normal,
            color,
            uv: Vector2::new(0.0 as f32, 1.0 as f32)
        });

        self.vertices.push(Vertex {
            position: offset + width + length,
            normal,
            color,
            uv: Vector2::new(1.0 as f32, 1.0 as f32)
        });

        self.vertices.push(Vertex {
            position: offset + width,
            normal,
            color,
            uv: Vector2::new(1.0 as f32, 0.0 as f32)
        });

        let base_index = (self.vertices.len() - 4) as u32;
        self.indices.push(base_index);
        self.indices.push(base_index + 1);
        self.indices.push(base_index + 2);
        self.indices.push(base_index);
        self.indices.push(base_index + 2);
        self.indices.push(base_index + 3);

        self
    }

    pub fn push_box(&mut self, position: Vector3<f32>, length: f32, width: f32, height: f32, color: Vector3<f32>, center_as_pivot: bool) -> &MeshBuilder {
        let up = Vector3::unit_y() * height;
        let right = Vector3::unit_x() * width;
        let forward = Vector3::unit_z() * length;

        let (near_corner, far_corner) = if center_as_pivot {
            ((up + right + forward) / 2.0, -(up + right + forward) / 2.0)
        } else {
            (Vector3::zero(), up + right + forward)
        };

        let (near_corner, far_corner) = (near_corner + position, far_corner + position);

        self.push_quad(near_corner, forward, right, color);
        self.push_quad(near_corner, right, up, color);
        self.push_quad(near_corner, up, forward, color);

        self.push_quad(far_corner, -right, -forward, color);
        self.push_quad(far_corner, -up, -right, color);
        self.push_quad(far_corner, -forward, -up, color);
        self
    }
}
