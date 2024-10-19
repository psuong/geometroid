use std::ffi::{c_void, CStr, CString};

use ash::{
    ext::debug_utils,
    vk::{
        self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageSeverityFlagsEXT as SeverityFlag,
        DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessageTypeFlagsEXT as TypeFlag,
        DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateInfoEXT,
        DebugUtilsMessengerEXT,
    },
    Entry, Instance,
};

pub const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

/// Gets all the layer names from the REQUIRED_LAYERS.
/// We only care about "VK_LAYER_KHRONOS_validation" for now.
pub fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const i8>) {
    let layer_names: Vec<CString> = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).expect("Failed to build CString!"))
        .collect();

    let layer_name_pointers: Vec<*const i8> =
        layer_names.iter().map(|name| name.as_ptr()).collect();

    (layer_names, layer_name_pointers)
}

pub fn check_validation_layer_support(entry: &Entry) {
    let supported_layers = unsafe { entry.enumerate_instance_layer_properties().unwrap() };
    for required in REQUIRED_LAYERS {
        let found = supported_layers.iter().any(|layer| {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            let name = name.to_str().expect("Failed to get layer name pointer");
            required == name
        });

        if !found {
            panic!("Validation layer not supported: {}", required);
        }
    }
}

/// Sets up a validation layer to print out all messages and severity because I'm a
/// beginner and I'm going to need this :)
pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Option<(debug_utils::Instance, DebugUtilsMessengerEXT)> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }

    let create_info = DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            DebugUtilsMessageSeverityFlagsEXT::ERROR
                | DebugUtilsMessageSeverityFlagsEXT::INFO
                | DebugUtilsMessageSeverityFlagsEXT::WARNING,
        )
        .message_type(
            DebugUtilsMessageTypeFlagsEXT::GENERAL
                | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils = debug_utils::Instance::new(entry, instance);
    let debug_utils_messenger = unsafe {
        debug_utils
            .create_debug_utils_messenger(&create_info, None)
            .unwrap()
    };
    Some((debug_utils, debug_utils_messenger))
}

/// Delegate allows the engine to log messages from vulkan to stdout.
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: SeverityFlag,
    message_types: TypeFlag,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*p_callback_data).p_message);
    match message_severity {
        SeverityFlag::VERBOSE => log::debug!("{:?} - {:?}", message_types, message),
        SeverityFlag::INFO => log::info!("{:?} - {:?}", message_types, message),
        SeverityFlag::WARNING => log::warn!("{:?} - {:?}", message_types, message),
        _ => log::error!("{:?} - {:?}", message_types, message),
    }
    vk::FALSE
}
