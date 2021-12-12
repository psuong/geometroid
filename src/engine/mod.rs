use ash::{Entry, Instance, extensions::ext::DebugUtils, vk::{self, DebugUtilsMessengerEXT}};
use std::{ffi::{CString, CStr}, error::Error};
use winit::window::Window;

mod utils;
use utils::vulkan_debug_callback;

// TODO: Move this to a feature instead
const REQUIRED_LAYERS : [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];
const ENABLE_VALIDATION_LAYERS: bool = true;

pub struct Engine {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(DebugUtils, DebugUtilsMessengerEXT)>
}

impl Engine {
    pub fn new(_window: &Window) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::new().expect("Failed to create entry") };
        let instance = Self::create_instance(&entry).unwrap();
        let debug_report_callback = Self::setup_debug_messenger(&entry, &instance);

        Ok(Engine {
            _entry: entry,
            instance: instance,
            debug_report_callback: debug_report_callback
        })
    }

    pub fn update(&mut self) {
        println!("Running engine");
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

        let layer_name_ptrs : Vec<*const i8> = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if ENABLE_VALIDATION_LAYERS {
            Self::check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_name_ptrs);
        }

        unsafe {
            Ok(entry.create_instance(&instance_create_info, None)?)
        }
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

    fn setup_debug_messenger(entry: &Entry, instance: &Instance) -> Option<(DebugUtils, DebugUtilsMessengerEXT)> {
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
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
