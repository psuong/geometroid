use ash::vk::{self, VertexInputAttributeDescription};
use glam::{Vec2, Vec3};
use memoffset::offset_of;

pub struct Vertex {
    pub position : Vec2,
    pub color : Vec3
}

// TODO: Add a docstrings
impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

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
