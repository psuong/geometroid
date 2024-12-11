use std::ffi::CString;

use crate::{
    engine::{
        context::VkContext,
        memory::create_buffer,
        render::Vertex,
        shader_utils::{create_shader_module, read_shader_from_file},
        swapchain_wrapper::SwapchainProperties,
        texture::Texture,
        uniform_buffer_object::UniformBufferObject,
    },
    to_array,
};
use ash::{
    vk::{
        AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference,
        AttachmentStoreOp, BlendFactor, BlendOp, Buffer, BufferUsageFlags, ColorComponentFlags,
        CompareOp, CullModeFlags, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool,
        DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo,
        DescriptorSetLayout, DescriptorType, DeviceMemory, DeviceSize, Format, Framebuffer,
        FramebufferCreateInfo, FrontFace, GraphicsPipelineCreateInfo, ImageLayout, ImageView,
        LogicOp, MemoryPropertyFlags, Offset2D, Pipeline, PipelineBindPoint, PipelineCache,
        PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
        PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PrimitiveTopology, Rect2D, RenderPass, RenderPassCreateInfo, SampleCountFlags,
        ShaderStageFlags, SubpassDependency, SubpassDescription, Viewport, WriteDescriptorSet,
        SUBPASS_EXTERNAL,
    },
    Device,
};

// TODO: Implement the copy trait or wrap it into an Option
#[derive(Clone, Debug)]
pub struct RenderPipeline {
    pub framebuffers: Vec<Framebuffer>,
    pub render_pass: Option<RenderPass>,
    pub descriptor_set_layout: Option<DescriptorSetLayout>,
    pub descriptor_pool: Option<DescriptorPool>,
    pub descriptor_sets: Vec<DescriptorSet>,
    pub uniform_buffers: Vec<(Buffer, DeviceMemory)>,
    pub pipeline: Option<Pipeline>,
    pub pipeline_layout: Option<PipelineLayout>,
}

impl RenderPipeline {
    pub fn new(capacity: usize) -> Self {
        Self {
            framebuffers: Vec::with_capacity(capacity),
            render_pass: None,
            descriptor_set_layout: None,
            descriptor_pool: None,
            descriptor_sets: Vec::new(),
            uniform_buffers: Vec::new(),
            pipeline: None,
            pipeline_layout: None,
        }
    }

    pub fn initialize_render_pass(
        &mut self,
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: SampleCountFlags,
        depth_format: Format,
    ) -> &mut RenderPipeline {
        let color_attachment_desc = AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment_desc = AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_attachment_desc = AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::DONT_CARE)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR);

        let attachment_descs = to_array!(
            color_attachment_desc,
            depth_attachment_desc,
            resolve_attachment_desc
        );

        // The first attachment is pretty much a color buffer
        let color_attachment_ref = AttachmentReference::default()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_refs = [color_attachment_ref];

        let depth_attachment_ref = AttachmentReference::default()
            .attachment(1)
            .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_attachment_ref = AttachmentReference::default()
            .attachment(2)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_attachment_refs = [resolve_attachment_ref];

        // Every subpass references 1 or more attachment descriptions.
        let subpass_desc = SubpassDescription::default()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .resolve_attachments(&resolve_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref);
        let subpass_descs = to_array!(subpass_desc);

        // Subpasses in a render pass automatically take care of image layout transitions.
        // Transitions are controlled by subpass dependencies, which describe the memory layout &
        // execution dependencies between each subpass.
        //
        // There are typically 2 builtin dependencies that take care of the transiation at the
        // start + end of the renderpass.
        //
        // The subpass here uses the COLOR_ATTACHMENT_OUTPUT. Another way is to make the semaphore
        // to PIPELINE_STAGE_TOP_OF_PIPE_BIT instead (TODO: Look into this and remove the subpass).
        let subpass_dep = SubpassDependency::default()
            .src_subpass(SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(AccessFlags::empty())
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
            );
        let subpass_deps = [subpass_dep];

        let render_pass_info = RenderPassCreateInfo::default()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps);

        self.render_pass =
            Some(unsafe { device.create_render_pass(&render_pass_info, None).unwrap() });
        self
    }

    pub fn initialize_framebuffers(
        &mut self,
        device: &Device,
        image_views: &[ImageView],
        color_texture: Texture,
        depth_texture: Texture,
        swapchain_properties: SwapchainProperties,
    ) -> &mut RenderPipeline {
        if let Some(pass) = self.render_pass {
            self.framebuffers.clear();
            image_views
                .iter()
                .map(|view| [color_texture.view, depth_texture.view, *view])
                .map(|attachment| {
                    let framebuffer_info = FramebufferCreateInfo::default()
                        .render_pass(pass)
                        .attachments(&attachment)
                        .width(swapchain_properties.extent.width)
                        .height(swapchain_properties.extent.height)
                        // since we only have 1 layer defined in the swapchain, the framebuffer
                        // must also only define 1 layer.
                        .layers(1);

                    unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
                })
                .for_each(|framebuffer| {
                    self.framebuffers.push(framebuffer);
                });
        }
        self
    }

    pub fn create_descriptor_set_layout(&mut self, device: &Device) -> &mut RenderPipeline {
        Some(UniformBufferObject::get_descriptor_set_layout(device));
        self
    }

    /// Descriptor set layouts can only be created in a pool like a command buffer.
    /// The pool size needs to accomodate the image sampler and the uniform buffer.
    pub fn create_descriptor_pool(&mut self, device: &Device, size: u32) -> &mut RenderPipeline {
        let descriptor_pool_size = DescriptorPoolSize {
            ty: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let ubo_pool_size = descriptor_pool_size;
        let sampler_pool_size = DescriptorPoolSize {
            ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: size,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(size);
        self.descriptor_pool =
            Some(unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() });
        self
    }

    /// A descriptor set is pretty much like a handle or a pointer to a resource (Image, Buffer, or
    /// some other information.
    /// DescriptorSets are just a pack of that information and vulkan requires that data be packed
    /// together because it is much more efficient than individually binding resources.
    ///
    /// https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxBoundDescriptorSets&platform=windows
    ///
    /// Some devices only let us bind 4 descriptors and an efficient way to use descriptors is to
    /// have per type descriptors.
    ///
    /// - 0. Engine Global Resources
    /// - 1. Per Pass Resources
    /// - 2. Material Resources
    /// - 3. Per Object Resources
    pub fn create_descriptor_sets(
        &mut self,
        device: &Device,
        texture: Texture,
    ) -> &mut RenderPipeline {
        if self.descriptor_set_layout.is_some() && self.descriptor_pool.is_some() {
            let pool = self.descriptor_pool.unwrap();
            let layout = self.descriptor_set_layout.unwrap();

            let layouts = (0..self.uniform_buffers.len())
                .map(|_| layout)
                .collect::<Vec<_>>();

            let alloc_info = DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };
            descriptor_sets
                .iter()
                .zip(self.uniform_buffers.iter())
                .for_each(|(set, (buffer, _))| {
                    let buffer_info = DescriptorBufferInfo::default()
                        .buffer(*buffer)
                        .offset(0)
                        .range(size_of::<UniformBufferObject>() as DeviceSize);
                    let buffer_infos = [buffer_info];

                    // Create the descriptor set for the image here
                    let image_info = DescriptorImageInfo::default()
                        .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(texture.view)
                        .sampler(texture.sampler.unwrap());
                    let image_infos = to_array!(image_info);

                    let ubo_descriptor_write = WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_infos);

                    let sampler_descriptor_write = WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_infos);

                    let descriptor_writes = [ubo_descriptor_write, sampler_descriptor_write];
                    let null = [];

                    unsafe { device.update_descriptor_sets(&descriptor_writes, &null) }
                });
        }
        self
    }

    pub fn create_uniform_buffers(
        &mut self,
        vk_context: &VkContext,
        count: usize,
    ) -> &mut RenderPipeline {
        self.uniform_buffers.clear();

        let size = size_of::<UniformBufferObject>() as DeviceSize;
        for _ in 0..count {
            let (buffer, memory, _) = create_buffer(
                vk_context,
                size,
                BufferUsageFlags::UNIFORM_BUFFER,
                MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            );
            self.uniform_buffers.push((buffer, memory));
        }
        self
    }

    pub fn create_pipeline(
        &mut self,
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: SampleCountFlags,
    ) -> &mut RenderPipeline {
        let current_dir = std::env::current_dir().unwrap();
        log::info!("Current directory: {:?}", current_dir);

        // TODO: Create a method that loads a shader
        let vert_path = current_dir.join("assets/shaders/unlit-vert.spv");
        let vert_source = read_shader_from_file(vert_path);
        let frag_path = current_dir.join("assets/shaders/unlit-frag.spv");
        let frag_source = read_shader_from_file(frag_path);

        let vertex_shader_module = create_shader_module(device, &vert_source);
        let fragment_shader_module = create_shader_module(device, &frag_source);

        let vert_entry_point = CString::new("vert").unwrap();
        let vertex_shader_state_info = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&vert_entry_point);

        let frag_entry_point = CString::new("frag").unwrap();
        let fragment_shader_state_info = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&frag_entry_point);

        let shader_states_info = to_array!(vertex_shader_state_info, fragment_shader_state_info);

        let vertex_binding_descs = to_array!(Vertex::get_binding_description());
        let vertex_attribute_descs = Vertex::get_attribute_descriptions();

        // Describes the layout of the vertex data.
        // TODO: Uncomment when I create a mesh struct.
        let vertex_input_info = PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs);

        // Describes the kind of input geometry that will be drawn from the vertices and if primtive
        // restart is enabled.
        // Reusing vertices is a pretty standard optimization, so primtive restart
        let input_assembly_info = PipelineInputAssemblyStateCreateInfo::default()
            .topology(PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let height = swapchain_properties.extent.height as f32;
        let viewport = Viewport {
            x: 0.0,
            y: height,
            width: swapchain_properties.extent.width as _,
            height: height * -1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let viewports = to_array!(viewport);
        let scissor = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: swapchain_properties.extent,
        };
        let scissors = to_array!(scissor);

        let viewport_info = PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        // Can perform depth testing/face culling/scissor test at this stage.
        // Takes the geometry that is shaped by the vertices & turns it into a fragments that can be
        // colored.
        // Can configure to output fragments that fill entire polygons or just the edges (wireframe
        // shading).
        let rasterizer_info = PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(CullModeFlags::BACK)
            .front_face(FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .depth_clamp_enable(false); // Clamp anything that bleeds outside of 0.0 - 1.0 range

        // An easy way to do some kind of anti-aliasing.
        let multisampling_info = PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples)
            .min_sample_shading(1.0)
            // .sample_mask() // null
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let depth_stencil_info = PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default());

        let color_blending_attachment = PipelineColorBlendAttachmentState::default()
            .color_write_mask(ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(BlendFactor::ONE)
            .dst_color_blend_factor(BlendFactor::ZERO)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD);
        let color_blend_attachments = [color_blending_attachment];

        let color_blending_info = PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let layout = {
            let layouts = to_array!(self.descriptor_set_layout.unwrap());
            let layout_info = PipelineLayoutCreateInfo::default().set_layouts(&layouts);

            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline_info = GraphicsPipelineCreateInfo::default()
            .stages(&shader_states_info)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            .layout(layout)
            .render_pass(self.render_pass.unwrap())
            .subpass(0);
        let pipeline_infos = to_array!(pipeline_info);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(PipelineCache::null(), &pipeline_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        };

        self.pipeline = Some(pipeline);
        self.pipeline_layout = Some(layout);
        self
    }

    pub fn release(&self, device: &Device) {
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|f| device.destroy_framebuffer(*f, None));
        }
    }
}
