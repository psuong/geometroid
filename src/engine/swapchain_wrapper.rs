use super::{context::VkContext, utils::QueueFamiliesIndices};
use ash::{
    khr::{surface, swapchain::Device as SwapchainDevice},
    vk::{
        ColorSpaceKHR, CompositeAlphaFlagsKHR, Extent2D, Format, Image, ImageAspectFlags,
        ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType,
        PhysicalDevice, PresentModeKHR, SharingMode, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
        SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR,
    },
    Device as AshDevice,
};

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub format: SurfaceFormatKHR,
    pub present_mode: PresentModeKHR,
    pub extent: Extent2D,
}

impl SwapchainProperties {
    pub fn aspect_ratio(&self) -> f32 {
        self.extent.width as f32 / self.extent.height as f32
    }
}

pub struct SwapchainWrapper {
    pub loader: SwapchainDevice,
    pub khr: SwapchainKHR,
    pub images: Vec<Image>,
    pub image_views: Vec<ImageView>,
    pub properties: SwapchainProperties,
}

impl SwapchainWrapper {
    pub fn new(
        loader: SwapchainDevice,
        khr: SwapchainKHR,
        images: Vec<Image>,
        image_views: Vec<ImageView>,
        swapchain_properties: SwapchainProperties,
    ) -> Self {
        SwapchainWrapper {
            loader,
            khr,
            images,
            image_views,
            properties: swapchain_properties,
        }
    }

    pub fn update_internal_resources(
        &mut self,
        loader: SwapchainDevice,
        khr: SwapchainKHR,
        swapchain_properties: SwapchainProperties,
        images: Vec<Image>,
        image_views: Vec<ImageView>,
    ) {
        self.loader = loader;
        self.khr = khr;
        self.properties = swapchain_properties;
        self.images = images;
        self.image_views = image_views;
    }

    pub fn release_swapchain_resources(&self, ash_device: &AshDevice) {
        unsafe {
            // ash_device.destroy_render_pass(self.render_pass, None);
            self.image_views.iter().for_each(|v| {
                ash_device.destroy_image_view(*v, None);
            });
            self.loader.destroy_swapchain(self.khr, None);
        }
    }
}

pub struct SwapchainSupportDetails {
    pub capabilities: SurfaceCapabilitiesKHR,
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        device: PhysicalDevice,
        surface: &surface::Instance,
        surface_khr: SurfaceKHR,
    ) -> Self {
        let capabilities = unsafe {
            surface
                .get_physical_device_surface_capabilities(device, surface_khr)
                .unwrap()
        };

        let formats = unsafe {
            surface
                .get_physical_device_surface_formats(device, surface_khr)
                .unwrap()
        };

        let present_modes = unsafe {
            surface
                .get_physical_device_surface_present_modes(device, surface_khr)
                .unwrap()
        };

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }

    pub fn get_ideal_swapchain_properties(
        &self,
        preferred_dimensions: [u32; 2],
    ) -> SwapchainProperties {
        let format = choose_swapchain_surface_format(&self.formats);
        let present_mode = choose_swapchain_surface_present_mode(&self.present_modes);
        let extent = Self::choose_swapchain_extent(self.capabilities, &preferred_dimensions);
        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
    }

    /// Creates the swapchain extent, which is typically the resolution of the windowing surface.
    /// I _think_ this is where - if I wanted to implement FSR, I can do half resolution and
    /// upscale it.
    /// TODO: Definitely try implementing FSR :)
    fn choose_swapchain_extent(
        capabilities: SurfaceCapabilitiesKHR,
        preferred_dimensions: &[u32; 2],
    ) -> Extent2D {
        // Pick the animation studio.
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        let min = capabilities.min_image_extent;
        let max = capabilities.max_image_extent;

        Extent2D {
            width: preferred_dimensions[0].min(max.width).max(min.width),
            height: preferred_dimensions[1].min(max.height).max(min.height),
        }
    }
}

/// An abstraction of the internals of create_swapchain_image_views. All images are accessed
/// view VkImageView.
pub fn create_image_view(
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

/// Does exactly what it says, chooses the swap chain format based on whatever is available.
/// If R8G8R8A8 is available then it is selected.
fn choose_swapchain_surface_format(available_formats: &[SurfaceFormatKHR]) -> SurfaceFormatKHR {
    if available_formats.len() == 1 && available_formats[0].format == Format::UNDEFINED {
        return SurfaceFormatKHR {
            format: Format::B8G8R8A8_UNORM,
            color_space: ColorSpaceKHR::SRGB_NONLINEAR,
        };
    }

    *available_formats
        .iter()
        .find(|format| {
            format.format == Format::B8G8R8_UNORM
                && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(&available_formats[0])
}

/// Chooses the swapchain present mode. MAILBOX -> FIFO -> IMMEDIATE are the order of priority
/// when chosen.
///
/// IMMEDIATE means that the moment the image is submitted to the screen, the image is shown
/// right away! May cause tearing.
///
/// FIFO follows the queue priority. This is pretty much like most modern games with VSYNC.
/// Submit - if queue is full, wait until queue is emptied.
///
/// MAILBOX is like a queue & immediate mode. If the presentation queue is filled to the brim,
/// then we just overwrite whatever is in queue.
fn choose_swapchain_surface_present_mode(
    available_present_modes: &[PresentModeKHR],
) -> PresentModeKHR {
    if available_present_modes.contains(&PresentModeKHR::MAILBOX) {
        PresentModeKHR::MAILBOX
    } else if available_present_modes.contains(&PresentModeKHR::FIFO) {
        PresentModeKHR::FIFO
    } else {
        PresentModeKHR::IMMEDIATE
    }
}

pub fn create_swapchain_and_images(
    vk_context: &VkContext,
    queue_families_indices: QueueFamiliesIndices,
    dimensions: [u32; 2],
) -> (
    SwapchainDevice,
    SwapchainKHR,
    SwapchainProperties,
    Vec<Image>,
) {
    let details = SwapchainSupportDetails::new(
        vk_context.physical_device_ref(),
        vk_context.surface_ref(),
        vk_context.surface_khr,
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
        let mut builder = SwapchainCreateInfoKHR::default()
            .surface(vk_context.surface_khr)
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
    };

    let swapchain = SwapchainDevice::new(vk_context.instance_ref(), vk_context.device_ref());
    let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
    (swapchain, swapchain_khr, properties, images)
}

/// Creates a VKImageView so that we can use VKImage in the render pipeline. Image Views
/// describe how to access the image and which part of the images we can access. E.g. depth maps
/// don't need to be mipmapped since it's just a single view of the entire screen.
pub fn create_swapchain_image_views(
    device: &AshDevice,
    swapchain_images: &[Image],
    swapchain_properties: SwapchainProperties,
) -> Vec<ImageView> {
    swapchain_images
        .iter()
        .map(|image| {
            create_image_view(
                device,
                *image,
                1,
                swapchain_properties.format.format,
                ImageAspectFlags::COLOR,
            )
        })
        .collect::<Vec<ImageView>>()
}
