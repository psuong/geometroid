// use cgmath::{InnerSpace, Vec2, Vector3, Vector4, Zero};
use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

use super::render::Vertex;

pub struct MeshBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshBuilder {
    /// Creates a new mesh builder allocating a Vec of vertices and indices.
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

    fn push_sphere_rings_with_triangles(
        &mut self,
        segments: i32,
        center: Vec3,
        radius: f32,
        v: f32,
        color: Vec4,
        build_triangles: bool,
    ) {
        let angle_slice = PI * 2.0 / segments as f32;
        for i in 0..segments + 1 {
            let angle = angle_slice * i as f32;
            let unit_position = Vec3::new(angle.cos(), 0.0, angle.sin());
            let vertex_position = center + unit_position * radius;

            self.push_vertex(Vertex {
                position: vertex_position,
                normal: vertex_position.normalize(),
                color,
                uv: Vec2::new(i as f32 / segments as f32, v),
            });

            if build_triangles && i > 0 {
                let base_index = (self.vertices.len() - 1) as u32;
                let verts_per_row = segments + 1;
                let first_index = base_index;
                let second_index = base_index - 1;
                let third_index = base_index - (verts_per_row as u32);
                let fourth_index = third_index - 1;

                self.indices.push(first_index);
                self.indices.push(third_index);
                self.indices.push(second_index);

                self.indices.push(third_index);
                self.indices.push(fourth_index);
                self.indices.push(second_index);
            }
        }
    }

    pub fn push_vertex(&mut self, vertex: Vertex) -> &MeshBuilder {
        self.vertices.push(vertex);
        self
    }

    pub fn push_quad(
        &mut self,
        offset: Vec3,
        width:  Vec3,
        length: Vec3,
        color:  Vec4,
    ) -> &MeshBuilder {
        let normal = width.cross(length).normalize();

        self.vertices.push(Vertex {
            position: offset,
            normal,
            color,
            uv: Vec2::new(0.0_f32, 0.0_f32),
        });

        self.vertices.push(Vertex {
            position: offset + length,
            normal,
            color,
            uv: Vec2::new(0.0_f32, 1.0_f32),
        });

        self.vertices.push(Vertex {
            position: offset + width + length,
            normal,
            color,
            uv: Vec2::new(1.0_f32, 1.0_f32),
        });

        self.vertices.push(Vertex {
            position: offset + width,
            normal,
            color,
            uv: Vec2::new(1.0_f32, 0.0_f32),
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

    pub fn push_box(
        &mut self,
        position: Vec3,
        length: f32,
        width: f32,
        height: f32,
        color: Vec4,
        center_as_pivot: bool,
    ) -> &MeshBuilder {
        let up = Vec3::new(0.0, 1.0, 0.0) * height;
        let right = Vec3::new(1.0, 0.0, 0.0) * width;
        let forward = Vec3::new(0.0, 0.0, 1.0) * length;

        let (near_corner, far_corner) = if center_as_pivot {
            ((up + right + forward) / 2.0, -(up + right + forward) / 2.0)
        } else {
            (Vec3::ZERO, up + right + forward)
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

    pub fn push_sphere(&mut self, radius: f32, radial_segments: i32) -> &MeshBuilder {
        let height_segments = radial_segments / 2;
        let angle_slice = std::f32::consts::PI / height_segments as f32;
        for i in 0..height_segments + 1 {
            let new_angle = angle_slice * i as f32;
            let center = Vec3::new(0.0, (new_angle).cos() * -radius, 0.0);
            let sphere_radius = new_angle.sin() * radius;
            let v = i as f32 / height_segments as f32;
            self.push_sphere_rings_with_triangles(
                radial_segments,
                center,
                sphere_radius,
                v,
                Vec4::new(1.0, 1.0, 1.0, 1.0),
                i > 0,
            );
        }
        self
    }

    // pub fn push_curved_cone(&mut self, start_radius: f32, end_radius: f32, height: f32, bend_angle_degs: f32, radial_segments: i32, height_segments: i32) -> &MeshBuilder {
    //     let bend_angle_rads = bend_angle_degs.to_radians();
    //     let bend_radius = height / bend_angle_rads;
    //     let angle_slice = bend_angle_rads / height_segments as f32;
    //     let start_offset = Vec3::new(bend_radius, 0.0, 0.0);
    //     let slope = Vec2::new(end_radius - start_radius, height);

    //     for i in 0..height_segments {
    //         let center = Vec3::new((angle_slice * i as f32).cos(), 0.0, 0.0);
    //     }

    //     todo!("Finish");
    //     self
    // }
}
