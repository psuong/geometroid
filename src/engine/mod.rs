use crate::engine::shader_utils::read_shader_from_file;

use ash::vk::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor,
    BlendOp, ClearColorValue, ClearValue, ColorComponentFlags, CommandBuffer,
    CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags,
    CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, Framebuffer, FramebufferCreateInfo,
    FrontFace, GraphicsPipelineCreateInfo, ImageLayout, LogicOp, Pipeline, PipelineBindPoint,
    PipelineCache, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
    PipelineLayout, PipelineLayoutCreateInfo, PipelineShaderStageCreateInfo, RenderPass,
    RenderPassBeginInfo, RenderPassCreateInfo, SampleCountFlags, Semaphore, SemaphoreCreateInfo,
    ShaderStageFlags, SubpassContents, SubpassDescription,
};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{
        self, ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR, CullModeFlags,
        DeviceCreateInfo, DeviceQueueCreateInfo, Image, ImageAspectFlags, ImageSubresourceRange,
        ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, Offset2D, PhysicalDevice,
        PhysicalDeviceFeatures, PipelineInputAssemblyStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineVertexInputStateCreateInfo,
        PipelineViewportStateCreateInfo, PolygonMode, PrimitiveTopology, Queue, QueueFlags, Rect2D,
        SharingMode, SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR, Viewport,
    },
    Device, Entry, Instance,
};
use std::{
    error::Error,
    ffi::{CStr, CString},
};

use winit::window::Window;

mod context;
mod debug;
mod shader_utils;
mod utils;

use context::VkContext;
use debug::{
    get_layer_names_and_pointers, setup_debug_messenger, ENABLE_VALIDATION_LAYERS, REQUIRED_LAYERS,
};
use utils::QueueFamiliesIndices;

use self::{shader_utils::create_shader_module, utils::SwapchainProperties};
use crate::{common::HEIGHT, engine::utils::SwapchainSupportDetails, WIDTH};

pub struct Engine {
    _physical_device: PhysicalDevice,
    _graphics_queue: Queue,
    _present_queue: Queue,
    _images: Vec<Image>,
    _swapchain_properties: SwapchainProperties,
    _command_buffers: Vec<CommandBuffer>,
    vk_context: VkContext,
    swapchain: Swapchain,
    swapchain_khr: SwapchainKHR,
    swapchain_image_views: Vec<ImageView>,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    render_pass: RenderPass,
    swapchain_framebuffers: Vec<Framebuffer>,
    command_pool: CommandPool,
    image_available_semaphore: Semaphore,
    render_finished_semaphore: Semaphore,
}

impl Engine {
    pub fn new(_window: &Window) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::new().expect("Failed to create entry") };
        let instance = Self::create_instance(&entry).unwrap();
        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let surface = Surface::new(&entry, &instance);
        let surface_khr =
            unsafe { ash_window::create_surface(&entry, &instance, _window, None).unwrap() };

        let (physical_device, queue_families_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr);
        let (logical_device, graphics_queue, present_queue) =
            Self::create_logical_device_with_graphics_queue(
                &instance,
                physical_device,
                queue_families_indices,
            );

        let vk_context = VkContext::new(
            entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            logical_device,
        );

        let dimensions = [WIDTH, HEIGHT];
        let (swapchain, swapchain_khr, properties, images) =
            Self::create_swapchain_and_images(&vk_context, queue_families_indices, dimensions);

        let swapchain_image_views =
            Self::create_swapchain_image_views(vk_context.device(), &images, properties);

        let render_pass = Self::create_render_pass(vk_context.device(), properties);
        let (pipeline, layout) =
            Self::create_pipeline(vk_context.device(), properties, render_pass);

        let swapchain_framebuffers = Self::create_framebuffers(
            vk_context.device(),
            &swapchain_image_views,
            render_pass,
            properties,
        );

        let command_pool = Self::create_command_pool(
            vk_context.device(),
            vk_context.instance(),
            vk_context.surface(),
            surface_khr,
            physical_device,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            &vk_context.device(),
            command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            pipeline,
        );

        let (image_available_semaphore, render_finished_semaphore) =
            Self::create_semaphores(vk_context.device());

        Ok(Engine {
            _physical_device: physical_device,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
            _images: images,
            _swapchain_properties: properties,
            _command_buffers: command_buffers,
            vk_context,
            swapchain,
            swapchain_khr,
            swapchain_image_views,
            pipeline_layout: layout,
            pipeline,
            render_pass,
            swapchain_framebuffers,
            command_pool,
            image_available_semaphore,
            render_finished_semaphore
        })
    }

    pub fn update(&mut self) {}

    fn create_instance(entry: &Entry) -> Result<Instance, Box<dyn Error>> {
        let app_name = CString::new("Geometroid").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();

        let mut extension_names = utils::required_extension_names();
        // Enable validation layers
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugUtils::name().as_ptr());
        }

        let layer_names: Vec<CString> = REQUIRED_LAYERS
            .iter()
            .map(|name| CString::new(*name).expect("Failed to build CString"))
            .collect();

        let layer_name_ptrs: Vec<*const i8> =
            layer_names.iter().map(|name| name.as_ptr()).collect();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if ENABLE_VALIDATION_LAYERS {
            Self::check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_name_ptrs);
        }

        unsafe { Ok(entry.create_instance(&instance_create_info, None)?) }
    }

    fn check_validation_layer_support(entry: &Entry) {
        for required in REQUIRED_LAYERS {
            let found = entry
                .enumerate_instance_layer_properties()
                .unwrap()
                .iter()
                .any(|layer| {
                    let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                    let name = name.to_str().expect("Failed to get layer name pointer");
                    required == name
                });

            if !found {
                panic!("Validation layer not supported: {}", required);
            }
        }
    }

    /// Pick an actual graphics card that exists on the machine.
    fn pick_physical_device(
        instance: &Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
    ) -> (PhysicalDevice, QueueFamiliesIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(&instance, &surface, surface_khr, *device))
            .expect("No suitable physical device!");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let queue_families_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        (device, queue_families_indices)
    }

    /// Checks if the physical device can do rendering. Ensures that there is a graphics and present
    /// queue index, which may be at different indices.
    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
        device: PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        graphics.is_some() && present.is_some()
    }

    /// Queues only support a subset of commands. It finds a graphics queue and present queue that
    /// can present images to the surface that is created.
    fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
        device: PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics: Option<u32> = None;
        let mut present: Option<u32> = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };

        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(device, index, surface_khr)
                    .unwrap()
            };

            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }

    /// Create a logical device based on the validation layers that are enabled.
    /// The logical device will interact with the physical device (our discrete video card).
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        device: vk::PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> (Device, Queue, Queue) {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let queue_priorities: [f32; 1] = [1.0f32];

        let queue_create_infos: Vec<DeviceQueueCreateInfo> = {
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            indices
                .iter()
                .map(|index| {
                    DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect()
        };

        // Grab the device extensions so that we can build the swapchain
        let device_extensions = Self::get_required_device_extensions();
        let device_extension_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = PhysicalDeviceFeatures::builder().build();
        let (_layer_names, layer_ptrs) = get_layer_names_and_pointers();

        let mut device_create_info_builder = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_ptrs)
            .enabled_features(&device_features);

        if ENABLE_VALIDATION_LAYERS {
            device_create_info_builder = device_create_info_builder.enabled_layer_names(&layer_ptrs)
        }

        let device_create_info = device_create_info_builder.build();
        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device!")
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };
        (device, graphics_queue, present_queue)
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [Swapchain::name()]
    }

    fn create_swapchain_and_images(
        vk_context: &VkContext,
        queue_families_indices: QueueFamiliesIndices,
        dimensions: [u32; 2],
    ) -> (Swapchain, SwapchainKHR, SwapchainProperties, Vec<Image>) {
        let details = SwapchainSupportDetails::query(
            vk_context.physical_device(),
            vk_context.surface(),
            vk_context.surface_khr(),
        );

        let properties = details.get_ideal_swapchain_properties(dimensions);

        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;

        // When selecting the image count, a size of 1 may cause us to wait before displaying the
        // second image. When we can use multiple images, we should try to.
        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }

            preferred
        };

        log::debug!(
            "Creating swapchain. \n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresent Mode: {:?}\n\tExtent: {:?}\n\tImage Count: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count
        );

        let graphics = queue_families_indices.graphics_index;
        let present = queue_families_indices.present_index;
        let families_indices = [graphics, present];

        let create_info = {
            let mut builder = SwapchainCreateInfoKHR::builder()
                .surface(vk_context.surface_khr())
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(ImageUsageFlags::COLOR_ATTACHMENT);

            builder = if graphics != present {
                builder
                    .image_sharing_mode(SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .build()
        };

        let swapchain = Swapchain::new(vk_context.instance(), vk_context.device());
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        (swapchain, swapchain_khr, properties, images)
    }

    /// Creates a VKImageView so that we can use VKImage in the render pipeline. Image Views
    /// describe how to access the image and which part of the images we can access. E.g. depth maps
    /// don't need to be mipmapped since it's just a single view of the entire screen.
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_properties: SwapchainProperties,
    ) -> Vec<ImageView> {
        swapchain_images
            .into_iter()
            .map(|image| {
                let create_info = ImageViewCreateInfo::builder()
                    .image(*image)
                    .view_type(ImageViewType::TYPE_2D) // We can use 3D or 1D textures
                    .format(swapchain_properties.format.format)
                    .components(ComponentMapping {
                        r: ComponentSwizzle::IDENTITY,
                        b: ComponentSwizzle::IDENTITY,
                        g: ComponentSwizzle::IDENTITY,
                        a: ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(ImageSubresourceRange {
                        // Describes the image's purpose
                        aspect_mask: ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build();

                unsafe { device.create_image_view(&create_info, None).unwrap() }
            })
            .collect::<Vec<_>>()
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        render_pass: RenderPass,
    ) -> (Pipeline, PipelineLayout) {
        // TODO: Load through a config file to make this work?
        let vert_source = read_shader_from_file(
            "D:/Documents/Projects/Rust/geometroid/src/shaders/shader.vert.spv",
        );
        let frag_source = read_shader_from_file(
            "D:/Documents/Projects/Rust/geometroid/src/shaders/shader.frag.spv",
        );

        let vertex_shader_module = create_shader_module(device, &vert_source);
        let fragment_shader_module = create_shader_module(device, &frag_source);

        let entry_point_name = CString::new("main").unwrap();
        let vertex_shader_state_info = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point_name)
            .build();

        let fragment_shader_state_info = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point_name)
            .build();

        let shader_states_info = [vertex_shader_state_info, fragment_shader_state_info];

        // Describes the layout of the vertex data.
        // TODO: Uncomment when I create a mesh struct.
        let vertex_input_info = PipelineVertexInputStateCreateInfo::builder()
            // .vertex_binding_descriptions(vertex_binding_descriptions)
            // .vertex_attribute_descriptions(vertex_attribute_descriptions)
            .build();

        // Describes the kind of input geometry that will be drawn from the vertices and if primtive
        // restart is enabled.
        // Reusing vertices is a pretty standard optimization, so primtive restart
        let input_assembly_info = PipelineInputAssemblyStateCreateInfo::builder()
            .topology(PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_properties.extent.width as _,
            height: swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let viewports = [viewport];
        let scissor = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: swapchain_properties.extent,
        };
        let scissors = [scissor];

        let viewport_info = PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors)
            .build();

        // Can perform depth testing/face culling/scissor test at this stage.
        // Takes the geometry that is shaped by the vertices & turns it into a fragments that can be
        // colored.
        // Can configure to output fragments that fill entire polygons or just the edges (wireframe
        // shading).
        let rasterizer_info = PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(CullModeFlags::BACK)
            .front_face(FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .depth_clamp_enable(false) // Clamp anything that bleeds outside of 0.0 - 1.0 range
            .build();

        // An easy way to do some kind of anti-aliasing.
        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            // .sample_mask() // null
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        let color_blending_attachment = PipelineColorBlendAttachmentState::builder()
            .color_write_mask(ColorComponentFlags::all())
            .blend_enable(false)
            .src_color_blend_factor(BlendFactor::ONE)
            .dst_color_blend_factor(BlendFactor::ZERO)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD)
            .build();
        let color_blend_attachments = [color_blending_attachment];

        let color_blending_info = PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        // TODO: Add depth & stencil testing here.
        let layout = {
            let layout_info = PipelineLayoutCreateInfo::builder()
                // .set_layouts(set_layouts)
                // .push_constant_ranges(push_constant_ranges)
                .build();

            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline_info = GraphicsPipelineCreateInfo::builder()
            .stages(&shader_states_info)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            // .depth_stencil_state(depth_stencil_state)
            .color_blend_state(&color_blending_info)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0)
            .build();
        let pipeline_infos = [pipeline_info];

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(PipelineCache::null(), &pipeline_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        };

        (pipeline, layout)
    }

    /// Create the renderpass + multiple subpasses. Subpasses are rendering ops that rely on the
    /// previous framebuffer. Post processing fx are common subpasses.
    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
    ) -> RenderPass {
        let attachment_desc = AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .build();
        let attachment_descs = [attachment_desc];

        // The first attachment is pretty much a color buffer
        let attachment_ref = AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let attachment_refs = [attachment_ref];

        // Every subpass references 1 or more attachment descriptions.
        let subpass_desc = SubpassDescription::builder()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_refs)
            .build();
        let subpass_descs = [subpass_desc];

        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .build();

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }

    /// Iterate through each image in the image view and create a framebuffer for each of them.
    /// Framebuffers have to be bound to a renderpass. So whatever properties that are defined in
    /// the renderpass should be the same properties defined for the frame buffer.
    fn create_framebuffers(
        device: &Device,
        image_views: &[ImageView],
        render_pass: RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<Framebuffer> {
        image_views
            .into_iter()
            .map(|view| [*view])
            .map(|attachment| {
                let framebuffer_info = FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachment)
                    .width(swapchain_properties.extent.width)
                    .height(swapchain_properties.extent.height)
                    // since we only have 1 layer defined in the swapchain, the framebuffer
                    // must also only define 1 layer.
                    .layers(1)
                    .build();

                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect::<Vec<Framebuffer>>()
    }

    /// We need a command pool which stores all command buffers. Manage some unmanaged memory
    /// cause never trust the idiot behind the screen to program something :)
    fn create_command_pool(
        device: &Device,
        instance: &Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
        physical_device: PhysicalDevice,
    ) -> CommandPool {
        let (graphics_family, _) =
            Self::find_queue_families(instance, surface, surface_khr, physical_device);

        let command_pool_info = CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_family.unwrap())
            .flags(CommandPoolCreateFlags::empty())
            .build();

        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        }
    }

    fn create_and_register_command_buffers(
        device: &Device,
        pool: CommandPool,
        framebuffers: &[Framebuffer],
        render_pass: RenderPass,
        swapchain_properties: SwapchainProperties,
        graphics_pipeline: Pipeline,
    ) -> Vec<CommandBuffer> {
        let allocate_info = CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32)
            .build();

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        buffers
            .iter()
            .zip(framebuffers.iter())
            .for_each(|(buffer, framebuffer)| {
                let buffer = *buffer;

                // begin the command buffer
                {
                    let command_buffer_begin_info = CommandBufferBeginInfo::builder()
                        .flags(CommandBufferUsageFlags::SIMULTANEOUS_USE)
                        // typically there would be an inheritance info here.
                        .build();
                    unsafe {
                        device
                            .begin_command_buffer(buffer, &command_buffer_begin_info)
                            .unwrap()
                    };
                }

                // Begin the render_pass
                {
                    let clear_values = [ClearValue {
                        color: ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }];

                    let render_pass_begin_info = RenderPassBeginInfo::builder()
                        .render_pass(render_pass)
                        .framebuffer(*framebuffer)
                        .render_area(Rect2D {
                            offset: Offset2D { x: 0, y: 0 },
                            extent: swapchain_properties.extent,
                        })
                        .clear_values(&clear_values)
                        .build();

                    unsafe {
                        device.cmd_begin_render_pass(
                            buffer,
                            &render_pass_begin_info,
                            SubpassContents::INLINE,
                        )
                    };
                }

                unsafe {
                    device.cmd_bind_pipeline(buffer, PipelineBindPoint::GRAPHICS, graphics_pipeline)
                }

                // Playback the command buffer
                unsafe { device.cmd_draw(buffer, 3, 1, 0, 0) };

                // End the renderpass
                unsafe { device.cmd_end_render_pass(buffer) };

                // End the command buffer
                unsafe { device.end_command_buffer(buffer).unwrap() };
            });
        buffers
    }

    /// Create 2 semaphores for when the image has been acquired and when we finish rendering so
    /// that presenting it can be done.
    fn create_semaphores(device: &Device) -> (Semaphore, Semaphore) {
        let image_available = {
            let semaphore_info = SemaphoreCreateInfo::builder().build();
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
        };

        let render_finished = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
        };
        (image_available, render_finished)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        log::info!("Releasing engine.");

        let device = self.vk_context.device();
        unsafe {
            log::debug!("Cleaning up the semaphores...");
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);

            log::debug!("Cleaning up CommandPool...");
            device.destroy_command_pool(self.command_pool, None);

            log::debug!("Cleaning up framebuffers...");
            // Framebuffers need to be destroyed before the pipeline.
            self.swapchain_framebuffers.iter().for_each(|framebuffer| {
                device.destroy_framebuffer(*framebuffer, None);
            });
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            self.swapchain_image_views
                .iter()
                .for_each(|v| device.destroy_image_view(*v, None));
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }
}
