pub(crate) use crate::common::MAX_FRAMES_IN_FLIGHT;
use crate::engine::render::render_desc::RenderDescriptor;
use crate::engine::render::Vertex;
use crate::math::{select, FORWARD, UP};
use crate::{to_array, unwrap_read_ref};
use array_util::empty;
use ash::util::Align;
use ash::{
    ext::debug_utils,
    khr::{surface as khr_surface, swapchain as khr_swapchain},
    vk::{
        self, AccessFlags, ApplicationInfo, BorderColor, Buffer, BufferImageCopy,
        BufferMemoryBarrier, BufferUsageFlags, ClearColorValue, ClearDepthStencilValue, ClearValue,
        CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo,
        CompareOp, DependencyFlags, DeviceCreateInfo, DeviceMemory, DeviceQueueCreateInfo,
        DeviceSize, Extent2D, Extent3D, Fence, FenceCreateFlags, FenceCreateInfo, Filter, Format,
        FormatFeatureFlags, Image, ImageAspectFlags, ImageBlit, ImageCreateFlags, ImageCreateInfo,
        ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers, ImageSubresourceRange,
        ImageTiling, ImageType, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType,
        IndexType, InstanceCreateFlags, InstanceCreateInfo, MemoryAllocateInfo, MemoryBarrier,
        MemoryMapFlags, MemoryPropertyFlags, Offset2D, Offset3D, PhysicalDevice,
        PhysicalDeviceFeatures, PipelineBindPoint, PipelineStageFlags, PresentInfoKHR, Queue,
        Rect2D, RenderPassBeginInfo, SampleCountFlags, SamplerAddressMode, SamplerCreateInfo,
        SamplerMipmapMode, SemaphoreCreateInfo, SharingMode, SubmitInfo, SubpassContents,
        QUEUE_FAMILY_IGNORED,
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
    pub graphics_queue: Queue,
    in_flight_frames: InFlightFrames,
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

        let texture = Self::create_texture_image(&vk_context, command_pool, graphics_queue);
        let img_count = images.len();

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
            .create_descriptor_pool(vk_context.device_ref(), img_count as _)
            .create_descriptor_set_layout(vk_context.device_ref())
            .create_uniform_buffers(&vk_context, img_count)
            .create_descriptor_sets(vk_context.device_ref(), texture)
            // TODO: Load the entities and update the ubos
            .create_pipeline(vk_context.device_ref(), properties, msaa_samples)
            .build();

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
            &render_pipeline,
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
            swapchain_wrapper,
            command_pool,
            transient_command_pool,
            msaa_samples,
            depth_format,
            depth_texture,
            texture,
            render_pipeline,
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

        let ubos = to_array!(ubo);
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

        let device = self.vk_context.device_ref();
        let extent = self.swapchain_wrapper.properties.extent;
        let dimensions = self
            .resize_dimensions
            .unwrap_or(to_array!(extent.width, extent.height));

        let (swapchain_loader, swapchain_khr, properties, images) =
            create_swapchain_and_images(&self.vk_context, self.queue_families_indices, dimensions);
        let swapchain_image_views = create_swapchain_image_views(device, &images, properties);

        let color_texture = Self::create_color_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            properties,
            self.msaa_samples,
        );

        let depth_texture = Self::create_depth_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            self.depth_format,
            properties.extent,
            self.msaa_samples,
        );

        let render_pipeline = self
            .render_pipeline
            .initialize_render_pass(device, properties, self.msaa_samples, self.depth_format)
            .create_pipeline(device, properties, self.msaa_samples)
            .initialize_framebuffers(
                device,
                &swapchain_image_views,
                color_texture,
                depth_texture,
                properties,
            )
            .build();

        self.swapchain_wrapper.update_internal_resources(
            swapchain_loader,
            swapchain_khr,
            properties,
            images,
            swapchain_image_views,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            device,
            self.command_pool,
            &self.swapchain_wrapper,
            &self.render_params,
            &render_pipeline,
        );

        self.render_pipeline = render_pipeline;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.command_buffers = command_buffers;
    }

    /// Force the engine to wait because ALL vulkan operations are async.
    #[inline]
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
        render_pipeline: &RenderPipeline,
    ) -> Vec<CommandBuffer> {
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

            let render_pass_begin_info = RenderPassBeginInfo::default()
                .render_pass(render_pipeline.render_pass.unwrap())
                .framebuffer(framebuffer)
                .render_area(Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: swapchain_properties.extent,
                })
                .clear_values(&clear_values);

            unsafe {
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
