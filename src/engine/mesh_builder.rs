use super::{
    math::{FORWARD, RIGHT, UP},
    render::Vertex,
    shapes::{Cube, Quad, Sphere},
};
use nalgebra_glm::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

pub struct MeshBuilder {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

macro_rules! make_triangle {
    ($builder:expr, $v1:expr, $v2:expr, $v3:expr) => {
        $builder.indices.push($v1);
        $builder.indices.push($v2);
        $builder.indices.push($v3);
    };
}

#[allow(dead_code)]
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

    fn push_circle(
        &mut self,
        color: Vec4,
        center: Vec3,
        radial_segments: i32,
        reverse_direction: bool,
    ) -> &MeshBuilder {
        let normal = if reverse_direction {
            Vec3::new(0.0, -1.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };

        self.push_vertex(Vertex {
            position: center,
            normal,
            color,
            uv: Vec2::new(0.5, 0.5),
        });

        let center_vertex = (self.vertices.len() - 1) as u32;
        let angle_slice = PI * 2.0 / radial_segments as f32;

        for i in 0..radial_segments + 1 {
            let angle = angle_slice * i as f32;
            let unit_position = Vec3::new(angle.cos(), 0.0, angle.sin());
            self.push_vertex(Vertex {
                position: unit_position,
                normal,
                color,
                uv: Vec2::new(unit_position.x + 1.0, unit_position.z + 1.0),
            });

            if i > 0 {
                let base_index = (self.vertices.len() - 1) as u32;

                if reverse_direction {
                    make_triangle!(self, center_vertex, base_index - 1, base_index);
                } else {
                    make_triangle!(self, center_vertex, base_index, base_index - 1);
                }
            }
        }

        self
    }

    fn push_ring(
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
            self.push_vertex(Vertex {
                position: center + unit_position * radius,
                color,
                normal: unit_position,
                uv: Vec2::new(i as f32 / segments as f32, v),
            });

            if i > 0 && build_triangles {
                let base_index = (self.vertices.len() - 1) as u32;
                let verts_per_row = (segments + 1) as u32;

                let first_index = base_index;
                let second_index = base_index - 1;
                let third_index = base_index - verts_per_row;
                let fourth_index = third_index - 1;

                make_triangle!(self, first_index, third_index, second_index);
                make_triangle!(self, first_index, fourth_index, second_index);
            }
        }
    }

    fn push_sphere_rings_no_triangles(
        &mut self,
        segments: i32,
        center: Vec3,
        radius: f32,
        v: f32,
        color: Vec4,
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
        }
    }

    fn push_sphere_rings_with_triangles(
        &mut self,
        segments: i32,
        center: Vec3,
        radius: f32,
        v: f32,
        color: Vec4,
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

            if i > 0 {
                let base_index = (self.vertices.len() - 1) as u32;
                let verts_per_row = segments + 1;
                let first_index = base_index;
                let second_index = base_index - 1;
                let third_index = base_index - (verts_per_row as u32);
                let fourth_index = third_index - 1;

                make_triangle!(self, first_index, third_index, second_index);
                make_triangle!(self, third_index, fourth_index, second_index);
            }
        }
    }

    #[inline]
    pub fn push_vertex(&mut self, vertex: Vertex) -> &MeshBuilder {
        self.vertices.push(vertex);
        self
    }

    pub fn push_quad(&mut self, quad: Quad, color: Vec4) -> &MeshBuilder {
        // let normal = quad.width.cross(&quad.length).normalize();
        let normal = quad.cross();

        self.push_vertex(Vertex {
            position: quad.offset,
            normal,
            color,
            uv: Vec2::new(0.0_f32, 0.0_f32),
        });

        self.push_vertex(Vertex {
            position: quad.offset_plus_length(),
            normal,
            color,
            uv: Vec2::new(0.0_f32, 1.0_f32),
        });

        self.push_vertex(Vertex {
            position: quad.offset_plus_length_width(),
            normal,
            color,
            uv: Vec2::new(1.0_f32, 1.0_f32),
        });

        self.push_vertex(Vertex {
            position: quad.offset_plus_width(),
            normal,
            color,
            uv: Vec2::new(1.0_f32, 0.0_f32),
        });

        let base_index = (self.vertices.len() - 4) as u32;
        make_triangle!(self, base_index, base_index + 1, base_index + 2);
        make_triangle!(self, base_index, base_index + 2, base_index + 3);

        self
    }

    pub fn push_box(&mut self, cube: Cube, color: Vec4) -> &MeshBuilder {
        let up = UP * cube.height();
        let right = RIGHT * cube.width();
        let forward = FORWARD * cube.length();

        let (near_corner, far_corner) = cube.corners();
        let (near_corner, far_corner) = (near_corner + cube.position, far_corner + cube.position);

        self.push_quad(Quad::new(near_corner, forward, right), color);
        self.push_quad(Quad::new(near_corner, right, up), color);
        self.push_quad(Quad::new(near_corner, up, forward), color);

        self.push_quad(Quad::new(far_corner, -right, -forward), color);
        self.push_quad(Quad::new(far_corner, -up, -right), color);
        self.push_quad(Quad::new(far_corner, -forward, -up), color);

        self
    }

    pub fn push_sphere(&mut self, sphere: Sphere) -> &MeshBuilder {
        let height_segments = sphere.radial_segments / 2;
        let angle_slice = std::f32::consts::PI / height_segments as f32;
        let white = Vec4::new(1.0, 1.0, 1.0, 1.0);

        self.push_sphere_rings_no_triangles(
            sphere.radial_segments,
            Vec3::new(0.0, angle_slice.cos() * -sphere.radius, 0.0),
            angle_slice.sin() * sphere.radius,
            0.0,
            white,
        );

        for i in 1..height_segments + 1 {
            let new_angle = angle_slice * i as f32;
            let center = Vec3::new(0.0, (new_angle).cos() * -sphere.radius, 0.0);
            let sphere_radius = new_angle.sin() * sphere.radius;
            let v = i as f32 / height_segments as f32;
            self.push_sphere_rings_with_triangles(
                sphere.radial_segments,
                center,
                sphere_radius,
                v,
                white,
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
