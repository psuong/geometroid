use ash::{
    vk::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType, ShaderStageFlags,
    },
    Device,
};
use nalgebra_glm::Mat4;

use crate::to_array;

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
    #[deprecated]
    pub fn get_descriptor_set_layout_binding<'a>() -> DescriptorSetLayoutBinding<'a> {
        // TODO: Check if we need a sampler descriptor later on
        DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX)
    }

    /// The descriptor_set_layout lets vulkan know the layout of the uniform buffers so that
    /// the shader has enough information.
    ///
    /// A common example is binding 2 buffers and an image to the mesh.
    pub fn get_descriptor_set_layout(device: &Device) -> DescriptorSetLayout {
        let ubo_binding = DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX);

        let sampler_binding = DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(ShaderStageFlags::FRAGMENT);

        let bindings = to_array!(ubo_binding, sampler_binding);
        let layout_info = DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }
}
