use ash::{
    ext::debug_utils,
    khr::surface,
    vk::{
        DebugUtilsMessengerEXT, PhysicalDevice, PhysicalDeviceMemoryProperties, SampleCountFlags,
        SurfaceKHR,
    },
    Device, Entry, Instance,
};

pub struct VkContext {
    _entry: Entry,
    pub instance: Instance,
    debug_report_callback: Option<(debug_utils::Instance, DebugUtilsMessengerEXT)>,
    pub surface: surface::Instance,
    pub surface_khr: SurfaceKHR,
    pub physical_device: PhysicalDevice,
    pub device: Device,
}

impl VkContext {
    pub fn new(
        entry: Entry,
        instance: Instance,
        debug_report_callback: Option<(debug_utils::Instance, DebugUtilsMessengerEXT)>,
        surface: surface::Instance,
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

    pub fn get_max_usable_sample_count(&self) -> SampleCountFlags {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(SampleCountFlags::TYPE_64) {
            SampleCountFlags::TYPE_64
        } else if sample_counts.contains(SampleCountFlags::TYPE_32) {
            SampleCountFlags::TYPE_32
        } else if sample_counts.contains(SampleCountFlags::TYPE_16) {
            SampleCountFlags::TYPE_16
        } else if sample_counts.contains(SampleCountFlags::TYPE_8) {
            SampleCountFlags::TYPE_8
        } else if sample_counts.contains(SampleCountFlags::TYPE_4) {
            SampleCountFlags::TYPE_4
        } else if sample_counts.contains(SampleCountFlags::TYPE_2) {
            SampleCountFlags::TYPE_2
        } else {
            SampleCountFlags::TYPE_1
        }
    }

    pub fn instance_ref(&self) -> &Instance {
        &self.instance
    }

    pub fn surface_ref(&self) -> &surface::Instance {
        &self.surface
    }
    
    pub fn physical_device_ref(&self) -> PhysicalDevice {
        self.physical_device
    }

    pub fn device_ref(&self) -> &Device {
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
