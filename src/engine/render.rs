use std::mem;

use ash::vk::{
    Format, VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
};
use glam::{const_vec2, const_vec3, Vec2, Vec3};
use memoffset::offset_of;

pub const VERTICES: [Vertex; 4] = [
    Vertex {
        position: const_vec2!([-0.5, -0.5]),
        color: const_vec3!([1.0, 1.0, 1.0]),
        coordinates: const_vec2!([0.0, 0.0])
    },
    Vertex {
        position: const_vec2!([-0.5, 0.5]),
        color: const_vec3!([1.0, 1.0, 1.0]),
        coordinates: const_vec2!([0.0, 1.0])
    },
    Vertex {
        position: const_vec2!([0.5, 0.5]),
        color: const_vec3!([1.0, 1.0, 1.0]),
        coordinates: const_vec2!([1.0, 1.0])
    },
    Vertex {
        position: const_vec2!([0.5, -0.5]),
        color: const_vec3!([1.0, 1.0, 1.0]),
        coordinates: const_vec2!([1.0, 0.0])
    },
];

pub const INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: Vec2,
    pub color: Vec3,
    pub coordinates: Vec2,
}

impl Vertex {
    /// A vertex binding describes at which rate to load data from memory
    /// throughout the vertices. It specifies the number of bytes between
    /// data entries and whether to move to the next data entry after each
    /// vertex or after each instance.
    pub fn get_binding_description() -> VertexInputBindingDescription {
        VertexInputBindingDescription::builder()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as u32)
            .input_rate(VertexInputRate::VERTEX)
            .build()
    }

    // TODO: Update doc strings
    /// An attribute description struct describes how to extract a vertex
    /// attribute from a chunk of vertex data originating from a binding
    /// description. We have two attributes, position and color, so we need
    /// two attribute description structs.
    pub fn get_attribute_descriptions() -> [VertexInputAttributeDescription; 3] {
        let position_desc = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, position) as u32)
            .build();

        let color_desc = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32)
            .build();

        let tex_coord_desc = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, coordinates) as u32)
            .build();

        [position_desc, color_desc, tex_coord_desc]
    }
}
