use ash::{
    extensions::{ext::DebugUtils, khr::Surface},
    vk::{
        self, DebugUtilsMessengerEXT, Format, FormatFeatureFlags, ImageTiling, PhysicalDevice,
        PhysicalDeviceMemoryProperties, SampleCountFlags, SurfaceKHR,
    },
    Device, Entry, Instance,
};

pub struct VkContext {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
    surface: Surface,
    surface_khr: SurfaceKHR,
    physical_device: PhysicalDevice,
    device: Device,
}

impl VkContext {
    pub fn new(
        entry: Entry,
        instance: Instance,
        debug_report_callback: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
        surface: Surface,
        surface_khr: SurfaceKHR,
        physical_device: PhysicalDevice,
        device: Device,
    ) -> Self {
        VkContext {
            _entry: entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            device,
        }
    }

    pub fn get_mem_properties(&self) -> PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        }
    }

    pub fn find_supported_format(
        &self,
        candidates: &[Format],
        tiling: ImageTiling,
        features: FormatFeatureFlags,
    ) -> Option<Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical_device, *candidate)
            };

            (tiling == ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                || (tiling == ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }

    pub fn get_max_usable_sample_count(&self) -> vk::SampleCountFlags {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_64) {
            SampleCountFlags::TYPE_64
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_32) {
            SampleCountFlags::TYPE_32
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_16) {
            SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) {
            SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) {
            SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) {
            SampleCountFlags::TYPE_2
        } else {
            SampleCountFlags::TYPE_1
        }
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    pub fn surface_khr(&self) -> SurfaceKHR {
        self.surface_khr
    }

    pub fn physical_device(&self) -> PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        log::info!("Releasing VkContext");
        unsafe {
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((utils, messenger)) = self.debug_report_callback.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
