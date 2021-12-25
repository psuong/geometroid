use ash::{
    extensions::khr::{Surface, Win32Surface},
    vk::{
        ColorSpaceKHR, Extent2D, Format, PhysicalDevice, PresentModeKHR, SurfaceCapabilitiesKHR,
        SurfaceFormatKHR, SurfaceKHR,
    },
};

/// Get required extensions on Windows.
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[derive(Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
}

pub struct SwapchainSupportDetails {
    pub capabilities: SurfaceCapabilitiesKHR,
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
}

pub struct SwapchainProperties {
    pub format: SurfaceFormatKHR,
    pub present_mode: PresentModeKHR,
    pub extent: Extent2D
}

impl SwapchainSupportDetails {
    pub fn query(device: PhysicalDevice, surface: &Surface, surface_khr: SurfaceKHR) -> Self {
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

    pub fn get_ideal_sawpchain_properties(
        &self,
        preferred_dimensions: [u32; 2],
    ) -> SwapchainProperties {
        let format = Self::choose_swapchain_surface_format(&self.formats);
        let present_mode = Self::choose_swapchain_surface_present_mode(&self.present_modes);
        let extent = Self::choose_swapchain_extent(&self.capabilities, &preferred_dimensions);
        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
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

    /// Creates the swapchain extent, which is typically the resolution of the windowing surface.
    /// I _think_ this is where - if I wanted to implement FSR, I can do half resolution and
    /// upscale it.
    /// TODO: Definitely try implementing FSR :)
    fn choose_swapchain_extent(
        capabilities: &SurfaceCapabilitiesKHR, 
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

