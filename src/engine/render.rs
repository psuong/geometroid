use ash::vk::{self, VertexInputAttributeDescription};
use glam::{const_vec2, const_vec3, Vec2, Vec3};
use memoffset::offset_of;

pub const VERTICES: [Vertex; 4] = [
    Vertex {
        position: const_vec2!([0.0, -0.5]),
        color: const_vec3!([1.0, 0.0, 0.0]),
    },
    Vertex {
        position: const_vec2!([0.5, 0.5]),
        color: const_vec3!([0.0, 1.0, 0.0]),
    },
    Vertex {
        position: const_vec2!([-0.5, 0.5]),
        color: const_vec3!([0.0, 0.0, 1.0]),
    },
    Vertex {
        position: const_vec2!([0.5, -0.5]),
        color: const_vec3!([1.0, 1.0, 1.0]),
    }
];

pub const INDICES: [u32; 6] = [ 0, 1, 2, 2, 3, 0 ];

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: Vec2,
    pub color: Vec3,
}

impl Vertex {
    /// A vertex binding describes at which rate to load data from memory
    /// throughout the vertices. It specifies the number of bytes between
    /// data entries and whether to move to the next data entry after each
    /// vertex or after each instance.
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    /// An attribute description struct describes how to extract a vertex
    /// attribute from a chunk of vertex data originating from a binding
    /// description. We have two attributes, position and color, so we need
    /// two attribute description structs.
    pub fn get_attribute_descriptions() -> [VertexInputAttributeDescription; 2] {
        let position_desc = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, position) as u32)
            .build();

        let color_desc = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32)
            .build();

        [position_desc, color_desc]
    }
}
