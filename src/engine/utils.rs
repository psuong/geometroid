use ash::{
    extensions::khr::{Surface, Win32Surface},
    vk::{
        self, DebugUtilsMessageSeverityFlagsEXT as SeverityFlag,
        DebugUtilsMessageTypeFlagsEXT as TypeFlag, DebugUtilsMessengerCallbackDataEXT,
    },
};
use std::ffi::{c_void, CStr};

/// Get required extensions on Windows.
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

/// Maps Message Severity and
pub unsafe extern "system" fn vulkan_debug_callback(
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
