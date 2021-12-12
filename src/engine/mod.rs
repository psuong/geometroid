use ash::{
    extensions::ext::DebugUtils,
    vk::{
        self, DebugUtilsMessengerEXT, DeviceCreateInfo, DeviceQueueCreateInfo, PhysicalDevice,
        PhysicalDeviceFeatures, Queue, QueueFlags,
    },
    Device, Entry, Instance,
};
use std::{
    error::Error,
    ffi::{CStr, CString},
};
use winit::window::Window;

mod utils;
use utils::vulkan_debug_callback;

// TODO: Move this to a feature instead
const REQUIRED_LAYERS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];
const ENABLE_VALIDATION_LAYERS: bool = true;

pub struct Engine {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
    _physical_device: PhysicalDevice,
    logical_device: Device,
    _graphics_queue: Queue
}

impl Engine {
    pub fn new(_window: &Window) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::new().expect("Failed to create entry") };
        let instance = Self::create_instance(&entry).unwrap();
        let debug_report_callback = Self::setup_debug_messenger(&entry, &instance);
        let physical_device = Self::pick_physical_device(&instance);
        let (logical_device, graphics_queue) = Self::create_logical_device_with_graphics_queue(&instance, physical_device);

        Ok(Engine {
            _entry: entry,
            instance,
            debug_report_callback,
            _physical_device: physical_device,
            logical_device,
            _graphics_queue : graphics_queue
        })
    }

    pub fn update(&mut self) {
        log::info!("Running engine");
    }

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
    fn pick_physical_device(instance: &Instance) -> PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(&instance, *device))
            .expect("No suitable physical device!");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        device
    }

    /// Checks if the physical device can do rendering.
    fn is_device_suitable(instance: &Instance, device: PhysicalDevice) -> bool {
        Self::find_queue_families(instance, device).is_some()
    }

    /// Queues only support a subset of commands. Find the queue family which bests matches
    /// our need to render graphics.
    fn find_queue_families(instance: &Instance, device: PhysicalDevice) -> Option<usize> {
        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        props
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_count > 0 && family.queue_flags.contains(QueueFlags::GRAPHICS)
            })
            .map(|(index, _)| index)
    }

    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        device: PhysicalDevice,
    ) -> (Device, Queue) {
        let queue_family_index = Self::find_queue_families(instance, device).unwrap() as u32;
        let queue_priorities: [f32; 1] = [1.0f32];
        let queue_create_info = [DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)
            .build()];

        let device_features = PhysicalDeviceFeatures::builder().build();
        let (_layer_names, layer_ptrs) = Self::get_layer_names_and_pointers();

        let mut device_create_info_builder = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_info)
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

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        (device, graphics_queue)
    }

    /// Gets all the layer names from the REQUIRED_LAYERS
    /// We only care about VK_LAYER_KHRONOS_validation for now.
    fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const i8>) {
        let layer_names: Vec<CString> = REQUIRED_LAYERS
            .iter()
            .map(|name| CString::new(*name).expect("Failed to build CString!"))
            .collect();

        let layer_name_pointers: Vec<*const i8> =
            layer_names.iter().map(|name| name.as_ptr()).collect();

        (layer_names, layer_name_pointers)
    }

    /// Sets up a validation layer to print out all messages and severity because I'm a
    /// beginner and I'm going to need this :)
    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Option<(DebugUtils, DebugUtilsMessengerEXT)> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::all())
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(Some(vulkan_debug_callback))
            .build();

        let debug_report = DebugUtils::new(entry, instance);
        let debug_report_callback = unsafe {
            debug_report
                .create_debug_utils_messenger(&create_info, None)
                .unwrap()
        };
        Some((debug_report, debug_report_callback))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        log::info!("Releasing engine.");
        unsafe {
            self.logical_device.destroy_device(None);
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
