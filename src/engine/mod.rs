use crate::common::MAX_FRAMES_IN_FLIGHT;
use crate::engine::render::Vertex;
use crate::engine::shader_utils::read_shader_from_file;

use ash::util::Align;
use ash::vk::{
    AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
    BlendFactor, BlendOp, Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, ClearColorValue,
    ClearValue, ColorComponentFlags, CommandBuffer, CommandBufferAllocateInfo,
    CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags, CommandPool,
    CommandPoolCreateFlags, CommandPoolCreateInfo, DeviceMemory, DeviceSize, Fence,
    FenceCreateFlags, FenceCreateInfo, Framebuffer, FramebufferCreateInfo, FrontFace,
    GraphicsPipelineCreateInfo, ImageLayout, IndexType, InstanceCreateInfo, LogicOp,
    MemoryAllocateInfo, MemoryMapFlags, MemoryPropertyFlags, MemoryRequirements,
    PhysicalDeviceMemoryProperties, Pipeline, PipelineBindPoint, PipelineCache,
    PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo, PipelineLayout,
    PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo, PipelineShaderStageCreateInfo,
    PipelineStageFlags, RenderPass, RenderPassBeginInfo, RenderPassCreateInfo, SampleCountFlags,
    SemaphoreCreateInfo, ShaderStageFlags, SubmitInfo, SubpassContents, SubpassDependency,
    SubpassDescription, SUBPASS_EXTERNAL,
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
use std::mem::{align_of, size_of};
use std::panic;
use std::{
    error::Error,
    ffi::{CStr, CString},
};
use winit::window::Window;

pub mod context;
pub mod debug;
pub mod render;
pub mod shader_utils;
pub mod utils;

use context::VkContext;
use debug::{
    get_layer_names_and_pointers, setup_debug_messenger, ENABLE_VALIDATION_LAYERS, REQUIRED_LAYERS,
};
use utils::QueueFamiliesIndices;

use self::render::{INDICES, VERTICES};
use self::utils::{InFlightFrames, SyncObjects};
use self::{shader_utils::create_shader_module, utils::SwapchainProperties};
use crate::{common::HEIGHT, engine::utils::SwapchainSupportDetails, WIDTH};

pub struct Engine {
    resize_dimensions: Option<[u32; 2]>,
    physical_device: PhysicalDevice,
    queue_families_indices: QueueFamiliesIndices,
    graphics_queue: Queue,
    present_queue: Queue,
    images: Vec<Image>,
    swapchain_properties: SwapchainProperties,
    vertex_buffer: Buffer,
    vertex_buffer_memory: DeviceMemory,
    index_buffer: Buffer,
    index_buffer_memory: DeviceMemory,
    command_buffers: Vec<CommandBuffer>,
    vk_context: VkContext,
    swapchain: Swapchain,
    swapchain_khr: SwapchainKHR,
    swapchain_image_views: Vec<ImageView>,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    render_pass: RenderPass,
    swapchain_framebuffers: Vec<Framebuffer>,
    command_pool: CommandPool,
    transient_command_pool: CommandPool,
    in_flight_frames: InFlightFrames,
}

impl Engine {
    pub fn new(_window: &Window) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::new().expect("Failed to create entry") };
        let instance = Self::create_instance(&entry);
        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let surface = Surface::new(&entry, &instance);
        let surface_khr =
            unsafe { ash_window::create_surface(&entry, &instance, _window, None).unwrap() };

        let (physical_device, queue_families_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr);

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

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
            Self::create_swapchain_image_views(vk_context.device_ref(), &images, properties);

        let render_pass = Self::create_render_pass(vk_context.device_ref(), properties);
        let (pipeline, layout) =
            Self::create_pipeline(vk_context.device_ref(), properties, render_pass);

        let swapchain_framebuffers = Self::create_framebuffers(
            vk_context.device_ref(),
            &swapchain_image_views,
            render_pass,
            properties,
        );

        let command_pool = Self::create_command_pool(
            vk_context.device_ref(),
            queue_families_indices,
            CommandPoolCreateFlags::empty(),
        );

        let transient_command_pool = Self::create_command_pool(
            vk_context.device_ref(),
            queue_families_indices,
            CommandPoolCreateFlags::empty(),
        );

        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            vk_context.device_ref(),
            memory_properties,
            transient_command_pool,
            graphics_queue,
        );

        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            vk_context.device_ref(),
            memory_properties,
            transient_command_pool,
            graphics_queue,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            &vk_context.device_ref(),
            command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            vertex_buffer,
            index_buffer,
            pipeline,
        );

        let in_flight_frames = Self::create_sync_objects(vk_context.device_ref());

        Ok(Engine {
            resize_dimensions: None,
            physical_device,
            queue_families_indices,
            graphics_queue,
            present_queue,
            images,
            swapchain_properties: properties,
            vk_context,
            swapchain,
            swapchain_khr,
            swapchain_image_views,
            pipeline_layout: layout,
            pipeline,
            render_pass,
            swapchain_framebuffers,
            command_pool,
            transient_command_pool,
            command_buffers,
            in_flight_frames,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
        })
    }

    fn create_instance(entry: &Entry) -> Instance {
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

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut instance_create_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if ENABLE_VALIDATION_LAYERS {
            Self::check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }

    // fn update(&mut self) -> bool {
    //     let draw_frame = self.draw_frame();
    //     self.wait_gpu_idle();
    //     draw_frame
    // }

    pub fn draw_frame(&mut self) -> bool {
        log::trace!("Drawing frame.");
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.vk_context
                .device_ref()
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap()
        };

        let result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                std::u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return true;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe {
            self.vk_context
                .device_ref()
                .reset_fences(&wait_fences)
                .unwrap()
        };

        // self.update_uniform_buffers(image_index);

        let device = self.vk_context.device_ref();
        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

        // Submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(self.graphics_queue, &submit_infos, in_flight_fence)
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain_khr];
        let images_indices = [image_index];

        {
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices)
                // .results() null since we only have one swapchain
                .build();
            let result = unsafe {
                self.swapchain
                    .queue_present(self.present_queue, &present_info)
            };
            match result {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return true;
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }
        }

        false
    }

    /// Cleans up the swapchain by destroying the framebuffers, freeing the command buffers,
    /// destroying the pipeline, pipeline layout, renderpass, swapchain image views, and
    /// finally swapchain.
    fn cleanup_swapchain(&mut self) {
        let device = self.vk_context.device_ref();
        unsafe {
            self.swapchain_framebuffers
                .iter()
                .for_each(|f| device.destroy_framebuffer(*f, None));
            device.free_command_buffers(self.command_pool, &self.command_buffers);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            self.swapchain_image_views
                .iter()
                .for_each(|v| device.destroy_image_view(*v, None));
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }

    /// When we recreate the swapchain we have to recreate the following
    /// - Image views - b/c they are based directly on the swapchain images
    /// - Render Pass because it depends on the format of the swpachain images
    /// - Swapchain format - rare for it to change during resizing, but should still be necessary
    /// - Viewport & scissor rectangle size - specified during the graphics pipeline creation
    /// - Framebuffers - directly depend on the swapchain images
    /// All fields have to be reassigned to the engine after creating them. Now we only need to
    /// recreate the swapchain _when_ the swapchain is incompatible with the surface (typically on
    /// resize) or if the window surface properties no longer match the swapchain's properties.
    pub fn recreate_swapchain(&mut self) {
        log::debug!("Recreating swapchain");

        // We must wait for the device to be idling before we recreate the swapchain
        self.wait_gpu_idle();

        self.cleanup_swapchain();

        let device = self.vk_context.device_ref();
        let dimensions = self.resize_dimensions.unwrap_or([
            self.swapchain_properties.extent.width,
            self.swapchain_properties.extent.height,
        ]);

        let (swapchain, swapchain_khr, properties, images) = Self::create_swapchain_and_images(
            &self.vk_context,
            self.queue_families_indices,
            dimensions,
        );
        let swapchain_image_views = Self::create_swapchain_image_views(device, &images, properties);

        let render_pass = Self::create_render_pass(device, properties);
        let (pipeline, layout) = Self::create_pipeline(device, properties, render_pass);
        let swapchain_framebuffers =
            Self::create_framebuffers(device, &swapchain_image_views, render_pass, properties);

        let command_buffers = Self::create_and_register_command_buffers(
            device,
            self.command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            self.vertex_buffer,
            self.index_buffer,
            pipeline,
        );

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = properties;
        self.images = images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
        self.pipeline = pipeline;
        self.pipeline_layout = layout;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.command_buffers = command_buffers;
    }

    /// Force the engine to wait because ALL vulkan operations are async.
    pub fn wait_gpu_idle(&self) {
        unsafe { self.vk_context.device_ref().device_wait_idle().unwrap() };
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
        device: PhysicalDevice,
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
            vk_context.physical_device_ref(),
            vk_context.surface_ref(),
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

        let swapchain = Swapchain::new(vk_context.instance_ref(), vk_context.device_ref());
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        (swapchain, swapchain_khr, properties, images)
    }

    /// Creates a VKImageView so that we can use VKImage in the render pipeline. Image Views
    /// describe how to access the image and which part of the images we can access. E.g. depth maps
    /// don't need to be mipmapped since it's just a single view of the entire screen.
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[Image],
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

        let vertex_binding_descs = [Vertex::get_binding_description()];
        let vertex_attribute_descs = Vertex::get_attribute_descriptions();

        // Describes the layout of the vertex data.
        // TODO: Uncomment when I create a mesh struct.
        let vertex_input_info = PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs)
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
        let multisampling_info = PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(SampleCountFlags::TYPE_1)
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

        // Subpasses in a render pass automatically take care of image layout transitions.
        // Transitions are controlled by subpass dependencies, which describe the memory layout &
        // execution dependencies between each subpass.
        //
        // There are typically 2 builtin dependencies that take care of the transiation at the
        // start + end of the renderpass.
        //
        // The subpass here uses the COLOR_ATTACHMENT_OUTPUT. Another way is to make the semaphore
        // to PIPELINE_STAGE_TOP_OF_PIPE_BIT instead (TODO: Look into this and remove the subpass).
        let subpass_dep = SubpassDependency::builder()
            .src_subpass(SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(AccessFlags::empty())
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();
        let subpass_deps = [subpass_dep];

        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps)
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

    fn create_vertex_buffer(
        device: &Device,
        mem_properties: PhysicalDeviceMemoryProperties,
        command_pool: CommandPool,
        transfer_queue: Queue,
    ) -> (Buffer, DeviceMemory) {
        Self::create_device_local_buffer_with_data::<u32, _>(
            device,
            mem_properties,
            command_pool,
            transfer_queue,
            BufferUsageFlags::VERTEX_BUFFER,
            &VERTICES,
        )
    }

    fn create_index_buffer(
        device: &Device,
        mem_properties: PhysicalDeviceMemoryProperties,
        command_pool: CommandPool,
        transfer_queue: Queue,
    ) -> (Buffer, DeviceMemory) {
        Self::create_device_local_buffer_with_data::<u16, _>(
            device,
            mem_properties,
            command_pool,
            transfer_queue,
            BufferUsageFlags::INDEX_BUFFER,
            &INDICES,
        )
    }

    fn create_device_local_buffer_with_data<A, T: Copy>(
        device: &Device,
        mem_properties: PhysicalDeviceMemoryProperties,
        command_pool: CommandPool,
        transfer_queue: Queue,
        usage: BufferUsageFlags,
        data: &[T],
    ) -> (vk::Buffer, DeviceMemory) {
        let size = (data.len() * size_of::<T>()) as DeviceSize;

        let (staging_buffer, staging_memory, staging_mem_size) = Self::create_buffer(
            device,
            mem_properties,
            size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_memory, 0, size, MemoryMapFlags::empty())
                .unwrap();
            let mut align = Align::new(data_ptr, align_of::<A>() as _, staging_mem_size);
            align.copy_from_slice(&VERTICES);
            device.unmap_memory(staging_memory);
        };

        let (buffer, memory, _) = Self::create_buffer(
            device,
            mem_properties,
            size,
            BufferUsageFlags::TRANSFER_DST | usage,
            MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // Copy from staging -> buffer - this will hold the Vertex data
        Self::copy_buffer(
            device,
            command_pool,
            transfer_queue,
            staging_buffer,
            buffer,
            size,
        );

        // Clean up the staging buffer b/c we've already copied the data!
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        };
        (buffer, memory)
    }

    /// Copies the size first bytes of src into dst
    ///
    /// Allocates a command buffer allocated from the 'command_pool'. The command buffer is
    /// submitted to the transfer_queue.
    fn copy_buffer(
        device: &Device,
        command_pool: CommandPool,
        transfer_queue: Queue,
        src: Buffer,
        dst: Buffer,
        size: DeviceSize,
    ) {
        let command_buffer = {
            let alloc_info = CommandBufferAllocateInfo::builder()
                .level(CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1)
                .build();

            unsafe { device.allocate_command_buffers(&alloc_info).unwrap()[0] }
        };

        let command_buffers = [command_buffer];
        {
            // Begin the recording
            let begin_info = CommandBufferBeginInfo::builder()
                .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build();

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap();
            };
        }
        {
            // Copy
            let region = BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            let regions = [region];
            unsafe {
                device.cmd_copy_buffer(command_buffer, src, dst, &regions);
            }
        }
        // End the recording
        unsafe { device.end_command_buffer(command_buffer).unwrap() };
        // Submit and wait
        {
            let submit_info = SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build();

            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(transfer_queue, &submit_infos, Fence::null())
                    .unwrap();
                device.queue_wait_idle(transfer_queue).unwrap();
            }
        }

        // Free
        unsafe { device.free_command_buffers(command_pool, &command_buffers) };
    }

    /// Creates a buffer and allocates the memory required.
    /// # Returns
    /// The buffer, its memory and the actual size in bytes of the allocated memory since it may
    /// be different from the requested size.
    fn create_buffer(
        device: &Device,
        device_mem_properties: PhysicalDeviceMemoryProperties,
        size: DeviceSize,
        usage: BufferUsageFlags,
        mem_properties: MemoryPropertyFlags,
    ) -> (Buffer, DeviceMemory, DeviceSize) {
        let buffer = {
            let buffer_info = BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(SharingMode::EXCLUSIVE)
                .build();

            unsafe { device.create_buffer(&buffer_info, None).unwrap() }
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = {
            let mem_type =
                Self::find_memory_type(mem_requirements, device_mem_properties, mem_properties);
            let alloc_info = MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type)
                .build();

            unsafe { device.allocate_memory(&alloc_info, None).unwrap() }
        };

        unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() }
        (buffer, memory, mem_requirements.size)
    }

    /// Finds a memory type in the mem_properties that is suitable
    /// for requirements and supports required_properties.
    ///
    /// # Returns
    /// The index of the memory type from mem_properties.
    fn find_memory_type(
        requirements: MemoryRequirements,
        mem_properties: PhysicalDeviceMemoryProperties,
        required_properties: MemoryPropertyFlags,
    ) -> u32 {
        for i in 0..mem_properties.memory_type_count {
            if requirements.memory_type_bits & (1 << i) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(required_properties)
            {
                return i;
            }
        }
        panic!("Failed to find a suitable memory type!");
    }

    /// We need a command pool which stores all command buffers. Manage some unmanaged memory
    /// cause never trust the idiot behind the screen to program something :)
    fn create_command_pool(
        device: &Device,
        queue_families_indices: QueueFamiliesIndices,
        create_flags: CommandPoolCreateFlags,
    ) -> CommandPool {
        let command_pool_info = CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families_indices.graphics_index)
            .flags(create_flags)
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
        vertex_buffer: Buffer,
        index_buffer: Buffer,
        graphics_pipeline: Pipeline,
    ) -> Vec<CommandBuffer> {
        let allocate_info = CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32)
            .build();

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        buffers.iter().enumerate().for_each(|(i, buffer)| {
            let buffer = *buffer;
            let framebuffer = framebuffers[i];

            // Begin the command buffer
            {
                let command_buffer_begin_info = CommandBufferBeginInfo::builder()
                    .flags(CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    .build();
                unsafe {
                    device
                        .begin_command_buffer(buffer, &command_buffer_begin_info)
                        .unwrap();
                }
            }

            // begin the render pass
            {
                let clear_values = [ClearValue {
                    color: ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];

                let render_pass_begin_info = RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
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

            // bind the pipeline
            unsafe {
                device.cmd_bind_pipeline(buffer, PipelineBindPoint::GRAPHICS, graphics_pipeline)
            };

            // Bind vertex buffer
            let vertex_buffers = [vertex_buffer];
            let offsets = [0];
            unsafe { device.cmd_bind_vertex_buffers(buffer, 0, &vertex_buffers, &offsets) };

            // Bind the index buffer
            unsafe { device.cmd_bind_index_buffer(buffer, index_buffer, 0, IndexType::UINT32) };

            // TODO: Bind the descriptor set
            unsafe { device.cmd_draw_indexed(buffer, INDICES.len() as _, 1, 0, 0, 0) };

            // End the renderpass
            unsafe { device.cmd_end_render_pass(buffer) };

            // End the cmd buffer
            unsafe { device.end_command_buffer(buffer).unwrap() };
        });

        buffers
    }

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec: Vec<SyncObjects> = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = SemaphoreCreateInfo::builder().build();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let render_finished_semaphore = {
                let semaphore_info = SemaphoreCreateInfo::builder().build();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let in_flight_fence = {
                let fence_info = FenceCreateInfo::builder()
                    .flags(FenceCreateFlags::SIGNALED)
                    .build();
                unsafe { device.create_fence(&fence_info, None).unwrap() }
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects);
        }
        InFlightFrames::new(sync_objects_vec)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        log::info!("Releasing engine.");
        self.cleanup_swapchain();

        let device = self.vk_context.device_ref();
        self.in_flight_frames.destroy(self.vk_context.device_ref());
        unsafe {
            log::debug!("Freeing index buffer memory...");
            device.free_memory(self.index_buffer_memory, None);
            device.destroy_buffer(self.index_buffer, None);

            log::debug!("Freeing vertex buffer memory...");
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);

            log::debug!("Cleaning up CommandPool...");
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_command_pool(self.transient_command_pool, None);
        }
    }
}
