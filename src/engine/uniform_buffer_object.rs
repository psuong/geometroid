use ash::vk::{DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags};
use glam::Mat4;

/// A descriptor layout specifies the types of resources that will be accessed. We need the 
/// model view projection matrix.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

impl UniformBufferObject {
    pub fn get_descriptor_set_layout_bindings() -> [DescriptorSetLayoutBinding; 1] {
        let ubo_layout_binding = DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX)
            // TODO: Check if we need a sampler descriptor later on
            .build();

        [ubo_layout_binding]
    }
}

