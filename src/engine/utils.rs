use ash::extensions::khr::{Surface, Win32Surface};

/// Get required extensions for windows.
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}