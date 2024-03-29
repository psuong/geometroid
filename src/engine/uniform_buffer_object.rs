use ash::vk::{DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags};
use cgmath::Matrix4;

/// A descriptor layout specifies the types of resources that will be accessed. We need the
/// model view projection matrix.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

impl UniformBufferObject {
    pub fn get_descriptor_set_layout_binding() -> DescriptorSetLayoutBinding {
        let ubo_layout_binding = DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX)
            // TODO: Check if we need a sampler descriptor later on
            .build();

        ubo_layout_binding
    }
}
