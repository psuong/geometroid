use ash::{
    extensions::khr::{Surface, Win32Surface},
    vk::{PhysicalDevice, PresentModeKHR, SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR},
};

/// Get required extensions on Windows.
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

pub struct SwapchainSupportDetails {
    capabilities: SurfaceCapabilitiesKHR,
    formats: Vec<SurfaceFormatKHR>,
    present_modes: Vec<PresentModeKHR>,
}

impl SwapchainSupportDetails {
    fn query(device: PhysicalDevice, surface: &Surface, surface_khr: SurfaceKHR) -> Self {
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
            present_modes
        }
    }
}
