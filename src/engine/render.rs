use std::{ffi::OsStr, mem, path::Path};

use ash::vk::{
    Format, VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
};
use memoffset::offset_of;
use nalgebra_glm::{Vec2, Vec3, Vec4};

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Mesh { vertices, indices }
    }

    pub fn index_count(&self) -> usize {
        self.indices.len() as usize
    }
}

// pub const VERTICES: [Vertex; 8] = [
//     // First quad
//     Vertex {
//         position: vec3(-0.5, -0.5, 0.0),
//         color: vec3(1.0, 1.0, 1.0),
//         uv: vec2(0.0, 0.0),
//     },
//     Vertex {
//         position: vec3(0.5, -0.5, 0.0),
//         color: vec3(1.0, 1.0, 1.0),
//         uv: vec2(1.0, 0.0),
//     },
//     Vertex {
//         position: vec3(0.5, 0.5, 0.0),
//         color: vec3(1.0, 1.0, 1.0),
//         uv: vec2(1.0, 1.0),
//     },
//     Vertex {
//         position: vec3(-0.5, 0.5, 0.0),
//         color: vec3(1.0, 1.0, 1.0),
//         uv: vec2(0.0, 1.0),
//     },
//     // Second quad
//     Vertex {
//         position: vec3(-0.5, -0.5, -0.5),
//         color: vec3(1.0, 0.0, 0.0),
//         uv: vec2(0.0, 0.0),
//     },
//     Vertex {
//         position: vec3(0.5, -0.5, -0.5),
//         color: vec3(1.0, 0.0, 0.0),
//         uv: vec2(1.0, 0.0),
//     },
//     Vertex {
//         position: vec3(0.5, 0.5, -0.5),
//         color: vec3(1.0, 0.0, 0.0),
//         uv: vec2(1.0, 1.0),
//     },
//     Vertex {
//         position: vec3(-0.5, 0.5, -0.5),
//         color: vec3(1.0, 0.0, 0.0),
//         uv: vec2(0.0, 1.0),
//     },
// ];

// pub const INDICES: [u16; 12] = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec4,
    pub uv: Vec2,
}

impl Vertex {
    /// A vertex binding describes at which rate to load data from memory
    /// throughout the vertices. It specifies the number of bytes between
    /// data entries and whether to move to the next data entry after each
    /// vertex or after each instance.
    pub fn get_binding_description() -> VertexInputBindingDescription {
        VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as u32)
            .input_rate(VertexInputRate::VERTEX)
    }

    // TODO: Update doc strings
    /// An attribute description struct describes how to extract a vertex
    /// attribute from a chunk of vertex data originating from a binding
    /// description. We have two attributes, position and color, so we need
    /// two attribute description structs.
    pub fn get_attribute_descriptions() -> [VertexInputAttributeDescription; 4] {
        let position_desc = VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, position) as u32);

        let normal_desc = VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, normal) as u32);

        let color_desc = VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(Format::R32G32B32A32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32);

        let tex_coord_desc = VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, uv) as u32);

        [position_desc, normal_desc, color_desc, tex_coord_desc]
    }
}
