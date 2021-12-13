use std::ffi::{c_void, CStr, CString};

use ash::{
    extensions::ext::DebugUtils,
    vk::{
        self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageSeverityFlagsEXT as SeverityFlag,
        DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessageTypeFlagsEXT as TypeFlag,
        DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerCreateFlagsEXT,
        DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT,
    },
    Entry, Instance,
};

pub const REQUIRED_LAYERS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

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

/// Sets up a validation layer to print out all messages and severity because I'm a
/// beginner and I'm going to need this :)
pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Option<(DebugUtils, DebugUtilsMessengerEXT)> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }

    let create_info = DebugUtilsMessengerCreateInfoEXT::builder()
        .flags(DebugUtilsMessengerCreateFlagsEXT::all())
        .message_severity(DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(DebugUtilsMessageTypeFlagsEXT::all())
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
