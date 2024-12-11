pub(crate) use crate::common::MAX_FRAMES_IN_FLIGHT;
use crate::engine::render::render_desc::RenderDescriptor;
use crate::engine::{render::Vertex, shader_utils::read_shader_from_file};
use crate::math::{select, FORWARD, UP};
use crate::{to_array, unwrap_read_ref, unwrap_value};
use array_util::empty;
use ash::util::Align;
use ash::{
    ext::debug_utils,
    khr::{surface as khr_surface, swapchain as khr_swapchain},
    vk::{
        self, AccessFlags, ApplicationInfo, BlendFactor, BlendOp, BorderColor, Buffer,
        BufferImageCopy, BufferMemoryBarrier, BufferUsageFlags, ClearColorValue,
        ClearDepthStencilValue, ClearValue, ColorComponentFlags, CommandBuffer,
        CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo,
        CompareOp, CullModeFlags, DependencyFlags, DescriptorBufferInfo, DescriptorImageInfo,
        DescriptorPool, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet,
        DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateInfo, DescriptorType, DeviceCreateInfo, DeviceMemory,
        DeviceQueueCreateInfo, DeviceSize, Extent2D, Extent3D, Fence, FenceCreateFlags,
        FenceCreateInfo, Filter, Format, FormatFeatureFlags, Framebuffer, FramebufferCreateInfo,
        FrontFace, GraphicsPipelineCreateInfo, Image, ImageAspectFlags, ImageBlit,
        ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers,
        ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags, ImageView,
        ImageViewCreateInfo, ImageViewType, IndexType, InstanceCreateFlags, InstanceCreateInfo,
        LogicOp, MemoryAllocateInfo, MemoryBarrier, MemoryMapFlags, MemoryPropertyFlags, Offset2D,
        Offset3D, PhysicalDevice, PhysicalDeviceFeatures, Pipeline, PipelineBindPoint,
        PipelineCache, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
        PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PresentInfoKHR, PrimitiveTopology, Queue, Rect2D, RenderPass, RenderPassBeginInfo,
        SampleCountFlags, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
        SemaphoreCreateInfo, ShaderStageFlags, SharingMode, SubmitInfo, SubpassContents, Viewport,
        WriteDescriptorSet, QUEUE_FAMILY_IGNORED,
    },
    Device as AshDevice, Entry, Instance,
};
use memory::{create_buffer, execute_one_time_commands, find_memory_type};
use nalgebra::{Point3, Unit};
use nalgebra_glm::{Mat4, Vec2, Vec3, Vec4};
use physical_devices::pick_physical_device;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use render::render_pipeline::RenderPipeline;
use render::Mesh;
use std::{
    ffi::{CStr, CString},
    mem::{align_of, size_of},
    panic,
    path::Path,
    time::Instant,
};
use swapchain_wrapper::{
    create_swapchain_and_images, create_swapchain_image_views, SwapchainProperties,
    SwapchainWrapper,
};
use winit::window::Window;

pub mod array_util;
pub mod camera;
pub mod context;
pub mod debug;
pub mod inputs;
pub mod memory;
pub mod mesh_builder;
pub mod physical_devices;
pub mod render;
pub mod shader_utils;
pub mod shapes;
pub mod swapchain_wrapper;
pub mod texture;
pub mod uniform_buffer_object;
pub mod utils;

use self::shader_utils::create_shader_module;
use self::texture::Texture;
use self::uniform_buffer_object::UniformBufferObject;
use self::utils::{InFlightFrames, SyncObjects};
use crate::{common::HEIGHT, WIDTH};
use context::VkContext;
use debug::{
    check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger,
    ENABLE_VALIDATION_LAYERS,
};
use utils::QueueFamiliesIndices;

pub struct Engine {
    pub dirty_swapchain: bool,
    // pub mouse_inputs: MouseInputs,
    command_buffers: Vec<CommandBuffer>,
    pub command_pool: CommandPool,
    // descriptor_set_layout: DescriptorSetLayout,
    // descriptor_pool: DescriptorPool,
    // descriptor_sets: Vec<DescriptorSet>,
    pub graphics_queue: Queue,
    in_flight_frames: InFlightFrames,
    // #[deprecated]
    // pipeline: Pipeline, // TODO: Move this a render pipeline struct
    // #[deprecated]
    // pipeline_layout: PipelineLayout, // TODO: Move this to a render pipeline wrapper
    present_queue: Queue,
    queue_families_indices: QueueFamiliesIndices,
    resize_dimensions: Option<[u32; 2]>,
    _start_instant: Instant,
    swapchain_wrapper: SwapchainWrapper,
    render_pipeline: RenderPipeline,
    transient_command_pool: CommandPool,
    msaa_samples: SampleCountFlags,
    depth_format: Format,
    depth_texture: Texture,
    color_texture: Texture,
    texture: Texture,
    // uniform_buffers: Vec<Buffer>,
    // uniform_buffer_memories: Vec<DeviceMemory>,
    render_params: Vec<RenderDescriptor>,
    pub vk_context: VkContext,
}

impl Engine {
    pub fn new(window: &Window) -> Self {
        let entry = unsafe { Entry::load().unwrap() };
        let instance = Self::create_instance(&entry, window);
        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let surface = khr_surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };

        let (physical_device, queue_families_indices) =
            pick_physical_device(&instance, &surface, surface_khr);

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

        let (swapchain_loader, swapchain_khr, properties, images) =
            create_swapchain_and_images(&vk_context, queue_families_indices, dimensions);

        let swapchain_image_views =
            create_swapchain_image_views(vk_context.device_ref(), &images, properties);

        let msaa_samples = vk_context.get_max_usable_sample_count();
        let depth_format = Self::find_depth_format(vk_context.instance_ref(), physical_device);

        // let render_pass = create_render_pass(
        //     vk_context.device_ref(),
        //     properties,
        //     msaa_samples,
        //     depth_format,
        // );
        // let descriptor_set_layout = Self::create_descriptor_set_layout(vk_context.device_ref());
        // let (pipeline, layout) = Self::create_pipeline(
        //     vk_context.device_ref(),
        //     properties,
        //     msaa_samples,
        //     render_pass,
        //     descriptor_set_layout,
        // );

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

        let color_texture = Self::create_color_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            properties,
            msaa_samples,
        );

        let depth_texture = Self::create_depth_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            depth_format,
            properties.extent,
            msaa_samples,
        );

        // let swapchain_framebuffers = Self::create_framebuffers(
        //     vk_context.device_ref(),
        //     &swapchain_image_views,
        //     color_texture,
        //     depth_texture,
        //     render_pass,
        //     properties,
        // );

        let texture = Self::create_texture_image(&vk_context, command_pool, graphics_queue);

        let render_pipeline = RenderPipeline::new(3)
            .initialize_render_pass(
                vk_context.device_ref(),
                properties,
                msaa_samples,
                depth_format,
            )
            .initialize_framebuffers(
                vk_context.device_ref(),
                &swapchain_image_views,
                color_texture,
                depth_texture,
                properties,
            )
            .create_descriptor_pool(vk_context.device_ref(), images.len() as _)
            .create_descriptor_set_layout(vk_context.device_ref())
            .create_uniform_buffers(&vk_context, images.len())
            .create_descriptor_sets(vk_context.device_ref(), texture)
            .create_pipeline(vk_context.device_ref(), properties, msaa_samples)
            .build();

        // let descriptor_set_layout = Self::create_descriptor_set_layout(vk_context.device_ref());
        // let (pipeline, layout) = Self::create_pipeline(
        //     vk_context.device_ref(),
        //     properties,
        //     msaa_samples,
        //     render_pass,
        //     descriptor_set_layout,
        // );

        // TODO: create entities to load
        // let mut mesh_builder = MeshBuilder::with_capacity(24);
        // mesh_builder.push_box(
        //     shapes::Cube::new(Vec3::zeros(), 1.0, 1.0, 1.0, true),
        //     Vec4::new(1.0, 1.0, 1.0, 1.0),
        // );
        // mesh_builder.push_sphere(Sphere::new(0.5, 100));

        // TODO: Abstract this part, because i have to keep creating descriptors, but this is in the
        // constructor so wtf
        // mesh_builder.push_sphere(0.5, 50);
        // let (vertices, indices) = (mesh_builder.vertices, mesh_builder.indices);

        // TODO: Move this from the constructor to the update loop. We need to create new uniform
        // buffers and descriptor sets in frame.
        let (vertices, indices) = Self::load_model();
        let mesh = Mesh::new(vertices, indices);
        let render_desc =
            RenderDescriptor::new(&vk_context, graphics_queue, transient_command_pool, &mesh);

        // let (uniform_buffers, uniform_buffer_memories) =
        //     Self::create_uniform_buffers(&vk_context, images.len());

        // let descriptor_pool =
        //     Self::create_descriptor_pool(vk_context.device_ref(), images.len() as _);
        // let descriptor_sets = Self::create_descriptor_sets(
        //     vk_context.device_ref(),
        //     descriptor_pool,
        //     descriptor_set_layout,
        //     &uniform_buffers,
        //     texture,
        // );

        let swapchain_wrapper = SwapchainWrapper::new(
            swapchain_loader,
            swapchain_khr,
            images,
            swapchain_image_views,
            properties,
        );

        let render_descriptors = vec![render_desc];

        let command_buffers = Self::create_and_register_command_buffers(
            vk_context.device_ref(),
            command_pool,
            &swapchain_wrapper,
            &render_descriptors,
            &render_pipeline, // render_pass,
                              // layout,
                              // &descriptor_sets,
                              // pipeline,
        );

        let in_flight_frames = Self::create_sync_objects(vk_context.device_ref());

        Self {
            dirty_swapchain: false,
            // mouse_inputs: MouseInputs::new(),
            _start_instant: Instant::now(),
            resize_dimensions: None,
            vk_context,
            queue_families_indices,
            graphics_queue,
            present_queue,
            // descriptor_set_layout,
            // pipeline_layout: layout,
            // pipeline,
            swapchain_wrapper,
            command_pool,
            transient_command_pool,
            msaa_samples,
            depth_format,
            depth_texture,
            texture,
            render_pipeline: render_pipeline,
            // uniform_buffers,
            // uniform_buffer_memories,
            // descriptor_pool,
            // descriptor_sets,
            command_buffers,
            in_flight_frames,
            color_texture,
            render_params: render_descriptors,
        }
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = CString::new("Geometroid").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap();

        let mut extension_names = extension_names.to_vec();
        // Enable validation layers
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(debug_utils::NAME.as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut instance_create_info = InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(InstanceCreateFlags::default());

        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }

    pub fn draw_frame(&mut self) -> bool {
        // log::trace!("Drawing frame.");
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.vk_context
                .device_ref()
                .wait_for_fences(&wait_fences, true, u64::MAX)
                .unwrap()
        };

        // TODO: Write a wrapper for acquiring the next image
        let result = unsafe {
            self.swapchain_wrapper.loader.acquire_next_image(
                self.swapchain_wrapper.khr,
                u64::MAX,
                image_available_semaphore,
                Fence::null(),
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

        self.update_uniform_buffers(image_index);

        let device = self.vk_context.device_ref();
        let wait_semaphores = to_array!(image_available_semaphore);
        let signal_semaphores = to_array!(render_finished_semaphore);

        // Submit command buffer
        {
            let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(self.graphics_queue, &submit_infos, in_flight_fence)
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain_wrapper.khr];
        let images_indices = [image_index];

        {
            let present_info = PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices);
            // .results() null since we only have one swapchain

            let result = unsafe {
                self.swapchain_wrapper
                    .loader
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

    /// Cleans up the swapchain by destroying the framebuffers, freeing the command
    /// buffers, destroying the pipeline, pipeline layout, renderpass, swapchain image
    /// views, and finally swapchain.
    fn cleanup_swapchain(&mut self) {
        let device = self.vk_context.device_ref();
        unsafe {
            self.depth_texture.destroy(device);
            self.color_texture.destroy(device);
            // self.swapchain_wrapper
            //     .framebuffers
            //     .iter()
            //     .for_each(|f| device.destroy_framebuffer(*f, None));
            device.free_command_buffers(self.command_pool, &self.command_buffers);
            self.render_pipeline.release(device);
            // device.destroy_pipeline(self.pipeline, None);
            // device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.swapchain_wrapper.release_swapchain_resources(device);
        }
    }

    /// Descriptor set layouts can only be created in a pool like a command buffer.
    /// The pool size needs to accomodate the image sampler and the uniform buffer.
    #[deprecated]
    fn create_descriptor_pool(device: &AshDevice, size: u32) -> DescriptorPool {
        let ubo_pool_size = DescriptorPoolSize {
            ty: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let sampler_pool_size = DescriptorPoolSize {
            ty: DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: size,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(size);
        unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() }
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
    #[deprecated]
    fn create_descriptor_sets(
        device: &AshDevice,
        pool: DescriptorPool,
        layout: DescriptorSetLayout,
        uniform_buffers: &[Buffer],
        texture: Texture,
    ) -> Vec<DescriptorSet> {
        let layouts = (0..uniform_buffers.len())
            .map(|_| layout)
            .collect::<Vec<_>>();
        let alloc_info = DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        descriptor_sets
            .iter()
            .zip(uniform_buffers.iter())
            .for_each(|(set, buffer)| {
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
                let image_infos = [image_info];

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

        descriptor_sets
    }

    #[deprecated(note = "Move to render_pipeline")]
    fn update_uniform_buffers(&mut self, current_image: u32) {
        // let elapsed = self._start_instant.elapsed();
        // let elapsed = elapsed.as_secs() as f32 + (elapsed.subsec_millis() as f32) / 1000.0;
        let elapsed = 0.0;

        let axis = Unit::new_normalize(UP);
        let model = Mat4::from_axis_angle(&axis, elapsed * 0.1667);

        let eye = Point3::new(2.0, 2.0, 2.0);
        let origin = Point3::new(0.0, 0.0, 0.0);

        let ubo = UniformBufferObject {
            model,
            view: Mat4::look_at_rh(&eye, &origin, &FORWARD),
            proj: nalgebra_glm::perspective_rh(
                self.swapchain_wrapper.properties.aspect_ratio(),
                60.0_f32.to_radians(),
                0.1,
                10.0,
            ),
        };

        let ubos = [ubo];
        let (_, buffer_mem) =
            unwrap_read_ref!(self.render_pipeline.uniform_buffers)[current_image as usize];
        // let buffer_mem = self.uniform_buffer_memories[current_image as usize];
        let size = size_of::<UniformBufferObject>() as DeviceSize;
        unsafe {
            let device = self.vk_context.device_ref();
            let data_ptr = device
                .map_memory(buffer_mem, 0, size, MemoryMapFlags::empty())
                .unwrap();

            let mut align = Align::new(data_ptr, align_of::<f32>() as _, size);
            align.copy_from_slice(&ubos);
            device.unmap_memory(buffer_mem);
        }
    }

    /// When we recreate the swapchain we have to recreate the following
    /// - Image views - b/c they are based directly on the swapchain images
    /// - Render Pass because it depends on the format of the swpachain images
    /// - Swapchain format - rare for it to change during resizing, but should still be necessary
    /// - Viewport & scissor rectangle size - specified during the graphics pipeline creation
    /// - Framebuffers - directly depend on the swapchain images
    ///     All fields have to be reassigned to the engine after creating them. Now we only need to
    ///     recreate the swapchain _when_ the swapchain is incompatible with the surface (typically on
    ///     resize) or if the window surface properties no longer match the swapchain's properties.
    pub fn recreate_swapchain(&mut self) {
        log::debug!("Recreating swapchain");
        // We must wait for the device to be idling before we recreate the swapchain
        self.wait_gpu_idle();
        self.cleanup_swapchain();

        // let device = self.vk_context.device_ref();
        // let extent = self.swapchain_wrapper.properties.extent;
        // let dimensions = self
        //     .resize_dimensions
        //     .unwrap_or(to_array!(extent.width, extent.height));

        // let (swapchain_loader, swapchain_khr, properties, images) =
        //     create_swapchain_and_images(&self.vk_context, self.queue_families_indices, dimensions);
        // let swapchain_image_views = create_swapchain_image_views(device, &images, properties);

        // let render_pass =
        //     create_render_pass(device, properties, self.msaa_samples, self.depth_format);

        // let (pipeline, layout) = Self::create_pipeline(
        //     device,
        //     properties,
        //     self.msaa_samples,
        //     render_pass,
        //     self.descriptor_set_layout,
        // );

        // let color_texture = Self::create_color_texture(
        //     &self.vk_context,
        //     self.command_pool,
        //     self.graphics_queue,
        //     properties,
        //     self.msaa_samples,
        // );

        // let depth_texture = Self::create_depth_texture(
        //     &self.vk_context,
        //     self.command_pool,
        //     self.graphics_queue,
        //     self.depth_format,
        //     properties.extent,
        //     self.msaa_samples,
        // );

        // let swapchain_framebuffers = Self::create_framebuffers(
        //     device,
        //     &swapchain_image_views,
        //     color_texture,
        //     depth_texture,
        //     render_pass,
        //     properties,
        // );

        // self.swapchain_wrapper.update_internal_resources(
        //     swapchain_loader,
        //     swapchain_khr,
        //     properties,
        //     images,
        //     swapchain_image_views,
        // );

        // let command_buffers = Self::create_and_register_command_buffers(
        //     device,
        //     self.command_pool,
        //     &self.swapchain_wrapper,
        //     &self.render_params,
        //     self.render_pipeline,
        // );

        // // let command_buffers = Self::create_and_register_command_buffers(
        // //     device,
        // //     self.command_pool,
        // //     &self.swapchain_wrapper,
        // //     &self.render_params,
        // //     render_pass,
        // //     layout,
        // //     &self.descriptor_sets,
        // //     pipeline,
        // // );

        // // self.pipeline = pipeline;
        // // self.pipeline_layout = layout;
        // self.color_texture = color_texture;
        // self.depth_texture = depth_texture;
        // self.command_buffers = command_buffers;
    }

    /// Force the engine to wait because ALL vulkan operations are async.
    pub fn wait_gpu_idle(&self) {
        unsafe { self.vk_context.device_ref().device_wait_idle().unwrap() };
    }

    /// Create a logical device based on the validation layers that are enabled.
    /// The logical device will interact with the physical device (our discrete video card).
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        device: PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> (AshDevice, Queue, Queue) {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let queue_priorities: [f32; 1] = [1.0f32];

        let queue_create_infos: Vec<DeviceQueueCreateInfo> = {
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            indices
                .iter()
                .map(|index| {
                    DeviceQueueCreateInfo::default()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                })
                .collect::<Vec<_>>()
        };

        // Grab the device extensions so that we can build the swapchain
        let device_extensions = Self::get_required_device_extensions();
        let device_extension_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = PhysicalDeviceFeatures::default().sampler_anisotropy(true);

        let device_create_info = DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_ptrs)
            .enabled_features(&device_features);

        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device!")
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };
        (device, graphics_queue, present_queue)
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn get_required_device_extensions() -> [&'static CStr; 2] {
        [khr_swapchain::NAME, ash::khr::portability_subset::NAME]
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [khr_swapchain::NAME]
    }

    /// An abstraction of the internals of create_swapchain_image_views. All images are accessed
    /// view VkImageView.
    fn create_image_view(
        device: &AshDevice,
        image: Image,
        mip_levels: u32,
        format: Format,
        aspect_mask: ImageAspectFlags,
    ) -> ImageView {
        let create_info = ImageViewCreateInfo::default()
            .image(image)
            .view_type(ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe { device.create_image_view(&create_info, None).unwrap() }
    }

    /// The descriptor_set_layout lets vulkan know the layout of the uniform buffers so that
    /// the shader has enough information.
    ///
    /// A common example is binding 2 buffers and an image to the mesh.
    #[deprecated]
    fn create_descriptor_set_layout(device: &AshDevice) -> DescriptorSetLayout {
        let ubo_binding = UniformBufferObject::get_descriptor_set_layout_binding();
        let sampler_binding = DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(ShaderStageFlags::FRAGMENT);
        let bindings = [ubo_binding, sampler_binding];
        let layout_info = DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    #[deprecated]
    fn create_pipeline(
        device: &AshDevice,
        swapchain_properties: SwapchainProperties,
        msaa_samples: SampleCountFlags,
        render_pass: RenderPass,
        descriptor_set_layout: DescriptorSetLayout,
    ) -> (Pipeline, PipelineLayout) {
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
            let layouts = to_array!(descriptor_set_layout);
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
            .render_pass(render_pass)
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

        (pipeline, layout)
    }

    /// Iterate through each image in the image view and create a framebuffer for each of them.
    /// Framebuffers have to be bound to a renderpass. So whatever properties that are defined in
    /// the renderpass should be the same properties defined for the frame buffer.
    fn create_framebuffers(
        device: &AshDevice,
        image_views: &[ImageView],
        color_texture: Texture,
        depth_texture: Texture,
        render_pass: RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<Framebuffer> {
        image_views
            .iter()
            .map(|view| [color_texture.view, depth_texture.view, *view])
            .map(|attachment| {
                let framebuffer_info = FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachment)
                    .width(swapchain_properties.extent.width)
                    .height(swapchain_properties.extent.height)
                    // since we only have 1 layer defined in the swapchain, the framebuffer
                    // must also only define 1 layer.
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect::<Vec<Framebuffer>>()
    }

    fn create_texture_image(
        vk_context: &VkContext,
        command_pool: CommandPool,
        copy_queue: Queue,
    ) -> Texture {
        let image = image::open("assets/textures/viking_room.png").unwrap();
        let image_as_rgb = image.to_rgba8();
        let extent = Extent2D {
            width: (image_as_rgb).width(),
            height: (image_as_rgb).height(),
        };
        let max_mip_levels = ((extent.width.min(extent.height) as f32).log2().floor() + 1.0) as u32;

        let pixels = image_as_rgb.into_raw();
        let image_size = (pixels.len() * size_of::<u8>()) as DeviceSize;
        let device = vk_context.device_ref();

        let (buffer, memory, mem_size) = create_buffer(
            vk_context,
            image_size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let ptr = device
                .map_memory(memory, 0, image_size, MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(ptr, align_of::<u8>() as _, mem_size);
            align.copy_from_slice(&pixels);
            device.unmap_memory(memory);
        }

        let (image, image_memory) = Self::create_image(
            vk_context,
            MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            max_mip_levels,
            SampleCountFlags::TYPE_1,
            Format::R8G8B8A8_UNORM,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::TRANSFER_DST
                | ImageUsageFlags::TRANSFER_SRC
                | ImageUsageFlags::SAMPLED,
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            Self::transition_image_layout(
                device,
                command_pool,
                copy_queue,
                image,
                max_mip_levels,
                Format::R8G8B8A8_UNORM,
                ImageLayout::UNDEFINED,
                ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            Self::copy_buffer_to_image(device, command_pool, copy_queue, buffer, image, extent);

            Self::generate_mipmaps(
                vk_context,
                command_pool,
                copy_queue,
                image,
                extent,
                Format::R8G8B8A8_UNORM,
                max_mip_levels,
            );
        }

        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }

        let image_view = Self::create_image_view(
            device,
            image,
            max_mip_levels,
            Format::R8G8B8A8_UNORM,
            ImageAspectFlags::COLOR,
        );

        let sampler = {
            let sampler_info = SamplerCreateInfo::default()
                .mag_filter(Filter::LINEAR)
                .min_filter(Filter::LINEAR)
                .address_mode_u(SamplerAddressMode::REPEAT)
                .address_mode_v(SamplerAddressMode::REPEAT)
                .address_mode_w(SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(CompareOp::ALWAYS)
                .mipmap_mode(SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(1.0)
                .max_lod(max_mip_levels as f32);

            unsafe { device.create_sampler(&sampler_info, None).unwrap() }
        };

        Texture::new(image, image_memory, image_view, Some(sampler))
    }

    /// Creates an image with common properties
    ///
    /// # Arguments
    /// usage - Describes how the image will be used. Textures that need to be applied to the mesh
    /// will typically require that the image is destination & can be sampled by the shader.
    fn create_image(
        vk_context: &VkContext,
        mem_properties: MemoryPropertyFlags,
        extent: Extent2D,
        mip_levels: u32,
        sample_count: SampleCountFlags,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
    ) -> (Image, DeviceMemory) {
        let image_info = ImageCreateInfo::default()
            // By declaring the image type as 2D, we access coordinates via x & y
            .image_type(ImageType::TYPE_2D)
            .extent(Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            // Undefined layotus will discard the texels, preinitialized ones will preserve the
            // texels
            // You may use preinitialized if you want to use the image as a staging image in
            // combination with with a linear tiling layout.
            .initial_layout(ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .samples(sample_count)
            .flags(ImageCreateFlags::empty()); // TODO: Look into this when I want to use a terrain.

        let image = unsafe {
            vk_context
                .device_ref()
                .create_image(&image_info, None)
                .unwrap()
        };

        // Like a buffer we need to know what are the requirements for the image.
        let mem_requirements =
            unsafe { vk_context.device_ref().get_image_memory_requirements(image) };
        let mem_type_index = find_memory_type(
            mem_requirements,
            vk_context.get_mem_properties(),
            mem_properties,
        );

        let alloc_info = MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            let mem = vk_context
                .device_ref()
                .allocate_memory(&alloc_info, None)
                .unwrap();
            vk_context
                .device_ref()
                .bind_image_memory(image, mem, 0)
                .unwrap();
            mem
        };

        (image, memory)
    }

    fn transition_image_layout(
        device: &AshDevice,
        command_pool: CommandPool,
        transition_queue: Queue,
        image: Image,
        mip_levels: u32,
        format: Format,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
    ) {
        execute_one_time_commands(device, command_pool, transition_queue, |buffer| {
            let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
                match (old_layout, new_layout) {
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ) => (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    ),
                    (
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    ),
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    ),
                    _ => panic!(
                        "Unsupported layout transition({:?} => {:?}).",
                        old_layout, new_layout
                    ),
                };

            let aspect_mask = if new_layout == ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                let mut mask = ImageAspectFlags::DEPTH;
                if Self::has_stencil_component(format) {
                    mask |= ImageAspectFlags::STENCIL;
                }
                mask
            } else {
                ImageAspectFlags::COLOR
            };

            let barrier = ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask);

            let barriers = [barrier];
            unsafe {
                device.cmd_pipeline_barrier(
                    buffer,
                    src_stage,
                    dst_stage,
                    DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }
        });
    }

    fn copy_buffer_to_image(
        device: &AshDevice,
        command_pool: CommandPool,
        transition_queue: Queue,
        buffer: Buffer,
        image: Image,
        extent: Extent2D,
    ) {
        execute_one_time_commands(device, command_pool, transition_queue, |command_buffer| {
            let region = BufferImageCopy::default()
                .buffer_offset(0) // Where does the pixel data actually start?
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(ImageSubresourceLayers {
                    aspect_mask: ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });

            let regions = [region];
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
            }
        });
    }

    fn generate_mipmaps(
        vk_context: &VkContext,
        command_pool: CommandPool,
        transfer_queue: Queue,
        image: Image,
        extent: Extent2D,
        format: Format,
        mip_levels: u32,
    ) {
        let format_properties = unsafe {
            vk_context
                .instance_ref()
                .get_physical_device_format_properties(vk_context.physical_device_ref(), format)
        };

        if !format_properties
            .optimal_tiling_features
            .contains(FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("Linear blitting is not supported for format {:?}.", format);
        }

        execute_one_time_commands(
            vk_context.device_ref(),
            command_pool,
            transfer_queue,
            |buffer| {
                let mut barrier = ImageMemoryBarrier::default()
                    .image(image)
                    .src_queue_family_index(QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
                    .subresource_range(ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                let mut mip_width = extent.width as i32;
                let mut mip_height = extent.height as i32;

                for level in 1..mip_levels {
                    let next_mip_width = select(mip_width > 1, mip_width / 2, mip_width);
                    let next_mip_height = select(mip_height > 1, mip_height / 2, mip_height);

                    barrier.subresource_range.base_mip_level = level - 1;
                    barrier.old_layout = ImageLayout::TRANSFER_DST_OPTIMAL;
                    barrier.new_layout = ImageLayout::TRANSFER_SRC_OPTIMAL;
                    barrier.src_access_mask = AccessFlags::TRANSFER_WRITE;
                    barrier.dst_access_mask = AccessFlags::TRANSFER_READ;

                    let barriers = to_array!(barrier);

                    unsafe {
                        vk_context.device_ref().cmd_pipeline_barrier(
                            buffer,
                            PipelineStageFlags::TRANSFER,
                            PipelineStageFlags::TRANSFER,
                            DependencyFlags::empty(),
                            &empty::<MemoryBarrier>(),
                            &empty::<BufferMemoryBarrier>(),
                            &barriers,
                        );
                    };

                    let blits = to_array!(ImageBlit::default()
                        .src_offsets([
                            Offset3D { x: 0, y: 0, z: 0 },
                            Offset3D {
                                x: mip_width,
                                y: mip_height,
                                z: 1,
                            },
                        ])
                        .src_subresource(ImageSubresourceLayers {
                            aspect_mask: ImageAspectFlags::COLOR,
                            mip_level: level - 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .dst_offsets([
                            Offset3D { x: 0, y: 0, z: 0 },
                            Offset3D {
                                x: next_mip_width,
                                y: next_mip_height,
                                z: 1,
                            },
                        ])
                        .dst_subresource(ImageSubresourceLayers {
                            aspect_mask: ImageAspectFlags::COLOR,
                            mip_level: level,
                            base_array_layer: 0,
                            layer_count: 1,
                        }));

                    unsafe {
                        vk_context.device_ref().cmd_blit_image(
                            buffer,
                            image,
                            ImageLayout::TRANSFER_SRC_OPTIMAL,
                            image,
                            ImageLayout::TRANSFER_DST_OPTIMAL,
                            &blits,
                            Filter::LINEAR,
                        )
                    };

                    barrier.old_layout = ImageLayout::TRANSFER_SRC_OPTIMAL;
                    barrier.new_layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    barrier.src_access_mask = AccessFlags::TRANSFER_READ;
                    barrier.dst_access_mask = AccessFlags::SHADER_READ;
                    let barriers = to_array!(barrier);

                    unsafe {
                        vk_context.device_ref().cmd_pipeline_barrier(
                            buffer,
                            PipelineStageFlags::TRANSFER,
                            PipelineStageFlags::FRAGMENT_SHADER,
                            DependencyFlags::empty(),
                            &empty::<MemoryBarrier>(),
                            &empty::<BufferMemoryBarrier>(),
                            &barriers,
                        )
                    };

                    mip_width = next_mip_width;
                    mip_height = next_mip_height;
                }

                barrier.subresource_range.base_mip_level = mip_levels - 1;
                barrier.old_layout = ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = AccessFlags::SHADER_READ;
                let barriers = [barrier];

                unsafe {
                    vk_context.device_ref().cmd_pipeline_barrier(
                        buffer,
                        PipelineStageFlags::TRANSFER,
                        PipelineStageFlags::FRAGMENT_SHADER,
                        DependencyFlags::empty(),
                        &empty::<MemoryBarrier>(),
                        &empty::<BufferMemoryBarrier>(),
                        &barriers,
                    )
                };
            },
        );
    }

    #[deprecated]
    fn create_uniform_buffers(
        vk_context: &VkContext,
        count: usize,
    ) -> (Vec<Buffer>, Vec<DeviceMemory>) {
        let size = size_of::<UniformBufferObject>() as DeviceSize;
        let mut buffers = Vec::new();
        let mut memories = Vec::new();

        for _ in 0..count {
            let (buffer, memory, _) = create_buffer(
                vk_context,
                size,
                BufferUsageFlags::UNIFORM_BUFFER,
                MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(buffer);
            memories.push(memory);
        }

        (buffers, memories)
    }

    fn find_depth_format(instance: &Instance, device: PhysicalDevice) -> Format {
        let candidates = vec![
            Format::D32_SFLOAT,
            Format::D32_SFLOAT_S8_UINT,
            Format::D24_UNORM_S8_UINT,
        ];

        Self::find_supported_format(
            instance,
            device,
            &candidates,
            ImageTiling::OPTIMAL,
            FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
        .expect("Failed to find a supported depth format")
    }

    fn find_supported_format(
        instance: &Instance,
        device: PhysicalDevice,
        candidates: &[Format],
        tiling: ImageTiling,
        features: FormatFeatureFlags,
    ) -> Option<Format> {
        candidates.iter().copied().find(|candidate| {
            let props =
                unsafe { instance.get_physical_device_format_properties(device, *candidate) };

            if tiling == ImageTiling::LINEAR && props.linear_tiling_features.contains(features) {
                true
            } else {
                tiling == ImageTiling::OPTIMAL && props.optimal_tiling_features.contains(features)
            }
        })
    }

    fn has_stencil_component(format: Format) -> bool {
        format == Format::D32_SFLOAT_S8_UINT || format == Format::D24_UNORM_S8_UINT
    }

    fn create_color_texture(
        vk_context: &VkContext,
        command_pool: CommandPool,
        transition_queue: Queue,
        swapchain_properties: SwapchainProperties,
        msaa_samples: SampleCountFlags,
    ) -> Texture {
        let format = swapchain_properties.format.format;
        let (image, memory) = Self::create_image(
            vk_context,
            MemoryPropertyFlags::DEVICE_LOCAL,
            swapchain_properties.extent,
            1,
            msaa_samples,
            format,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::TRANSIENT_ATTACHMENT | ImageUsageFlags::COLOR_ATTACHMENT,
        );

        Self::transition_image_layout(
            vk_context.device_ref(),
            command_pool,
            transition_queue,
            image,
            1,
            format,
            ImageLayout::UNDEFINED,
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let view = Self::create_image_view(
            vk_context.device_ref(),
            image,
            1,
            format,
            ImageAspectFlags::COLOR,
        );

        Texture::new(image, memory, view, None)
    }

    /// We need a command pool which stores all command buffers. Manage some unmanaged memory
    /// cause never trust the idiot behind the screen to program something :)
    fn create_command_pool(
        device: &AshDevice,
        queue_families_indices: QueueFamiliesIndices,
        create_flags: CommandPoolCreateFlags,
    ) -> CommandPool {
        let command_pool_info = CommandPoolCreateInfo::default()
            .queue_family_index(queue_families_indices.graphics_index)
            .flags(create_flags);

        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        }
    }

    fn create_and_register_command_buffers(
        device: &AshDevice,
        pool: CommandPool,
        swapchain_wrapper: &SwapchainWrapper,
        render_descs: &Vec<RenderDescriptor>,
        render_pipeline: &RenderPipeline, // render_pass: RenderPass,
                                         // pipeline_layout: PipelineLayout,
                                         // descriptor_sets: &[DescriptorSet],
                                         // graphics_pipeline: Pipeline,
    ) -> Vec<CommandBuffer> {
        log::info!("Registering cmd buffer.");
        let framebuffers = unwrap_read_ref!(render_pipeline.framebuffers);
        let allocate_info = CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32);

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };
        let swapchain_properties = swapchain_wrapper.properties;

        buffers.iter().enumerate().for_each(|(index, buffer)| {
            let buffer = *buffer;
            let framebuffer = framebuffers[index];

            // Begin the command buffer
            let command_buffer_begin_info =
                CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::SIMULTANEOUS_USE);
            unsafe {
                device
                    .begin_command_buffer(buffer, &command_buffer_begin_info)
                    .unwrap();
            }

            // begin the render pass
            let clear_values = [
                ClearValue {
                    color: ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                ClearValue {
                    depth_stencil: ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            log::info!("Begin info!");
            let render_pass_begin_info = RenderPassBeginInfo::default()
                .render_pass(render_pipeline.render_pass.unwrap())
                .framebuffer(framebuffer)
                .render_area(Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: swapchain_properties.extent,
                })
                .clear_values(&clear_values);

            unsafe {
                log::info!("Begin render pass!");
                device.cmd_begin_render_pass(
                    buffer,
                    &render_pass_begin_info,
                    SubpassContents::INLINE,
                )
            };

            // bind the pipeline
            unsafe {
                log::info!("Binding the pipeline");
                device.cmd_bind_pipeline(
                    buffer,
                    PipelineBindPoint::GRAPHICS,
                    render_pipeline.pipeline.unwrap(),
                )
            };

            let offsets = [0];

            unsafe {
                log::info!("Render descs: {}", render_descs.len());
                for render_desc in render_descs {
                    let vertex_buffers = to_array!(render_desc.vertex_buffer);
                    device.cmd_bind_vertex_buffers(buffer, 0, &vertex_buffers, &offsets);
                    device.cmd_bind_index_buffer(
                        buffer,
                        render_desc.index_buffer,
                        0,
                        IndexType::UINT32,
                    )
                }
            }

            // Bind vertex buffer
            // let vertex_buffers = to_array!(render_desc.vertex_buffer);
            // let offsets = [0];
            // unsafe { device.cmd_bind_vertex_buffers(buffer, 0, &vertex_buffers, &offsets) };

            // Bind the index buffer
            // unsafe {
            //     device.cmd_bind_index_buffer(buffer, render_desc.index_buffer, 0, IndexType::UINT32)
            // };

            // TODO: Bind the descriptor set
            unsafe {
                let null = [];
                device.cmd_bind_descriptor_sets(
                    buffer,
                    PipelineBindPoint::GRAPHICS,
                    render_pipeline.pipeline_layout.unwrap(),
                    0,
                    &unwrap_read_ref!(render_pipeline.descriptor_sets)[index..=index],
                    &null,
                );
            };
            log::info!("Finished processing index: {}", index);

            // Draw
            unsafe {
                for render_desc in render_descs {
                    device.cmd_draw_indexed(buffer, render_desc.index_count as _, 1, 0, 0, 0);
                }
            }

            // End the renderpass
            unsafe { device.cmd_end_render_pass(buffer) };

            // End the cmd buffer
            unsafe { device.end_command_buffer(buffer).unwrap() };
            log::info!("Finsihed registering cmd buffer.");
        });

        buffers
    }

    fn create_sync_objects(device: &AshDevice) -> InFlightFrames {
        let mut sync_objects_vec: Vec<SyncObjects> =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let render_finished_semaphore = {
                let semaphore_info = SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let in_flight_fence = {
                let fence_info = FenceCreateInfo::default().flags(FenceCreateFlags::SIGNALED);
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

    /// Create the depth buffer texture (image, memory and view).
    ///
    /// This function also transitions the image to be ready to be used
    /// as a depth/stencil attachement.
    fn create_depth_texture(
        vk_context: &VkContext,
        command_pool: CommandPool,
        transition_queue: Queue,
        format: Format,
        extent: Extent2D,
        msaa_samples: SampleCountFlags,
    ) -> Texture {
        let (image, mem) = Self::create_image(
            vk_context,
            MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            1,
            msaa_samples,
            format,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        let device = vk_context.device_ref();
        Self::transition_image_layout(
            device,
            command_pool,
            transition_queue,
            image,
            1,
            format,
            ImageLayout::UNDEFINED,
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        let view = Self::create_image_view(device, image, 1, format, ImageAspectFlags::DEPTH);

        Texture::new(image, mem, view, None)
    }

    // TODO: Move to an asset pipeline mod
    fn load_model() -> (Vec<Vertex>, Vec<u32>) {
        let path = Path::new("assets/models/viking_room.obj");
        log::info!("Loading model...{p}", p = path.to_str().unwrap());
        let (models, _) = tobj::load_obj(path, &tobj::LoadOptions::default()).unwrap();
        let mesh = &models[0].mesh;
        let positions = mesh.positions.as_slice();
        let coords = mesh.texcoords.as_slice();
        let vertex_count = mesh.positions.len() / 3;

        let mut vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            let u = coords[i * 2];
            let v = coords[i * 2 + 1];

            let vertex = Vertex {
                position: Vec3::new(x, y, z),
                uv: Vec2::new(u, 1.0 - v),
                normal: Vec3::new(0.0, 0.0, 0.0),
                color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            };
            vertices.push(vertex);
        }

        (vertices, mesh.indices.clone())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        log::debug!("Releasing engine.");
        self.cleanup_swapchain();

        let device = self.vk_context.device_ref();
        self.in_flight_frames.destroy(device);
        unsafe {
            // TODO: Move this to engine releasing
            // device.destroy_descriptor_pool(self.descriptor_pool, None);
            // device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            // self.uniform_buffer_memories
            //     .iter()
            //     .for_each(|m| device.free_memory(*m, None));
            // self.uniform_buffers
            //     .iter()
            //     .for_each(|b| device.destroy_buffer(*b, None));

            self.render_pipeline.drop(device);
            self.render_params.iter_mut().for_each(|render_param| {
                render_param.release(device);
            });

            self.texture.destroy(device);
            device.destroy_command_pool(self.transient_command_pool, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}
