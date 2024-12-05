use std::ffi::CStr;

use ash::{
    khr::surface,
    vk::{PhysicalDevice, PhysicalDeviceType, QueueFlags, SurfaceKHR, TRUE},
    Instance,
};

use crate::engine::utils::QueueFamiliesIndices;

use super::swapchain_wrapper::SwapchainSupportDetails;

/// Attempts to pick the physical graphics device that supports what we need. The highest priority
/// is a discrete GPU and supports our device. Otherwise, we need to just pick a device that
/// supports our needs.
pub fn pick_physical_device(
    instance: &Instance,
    surface: &surface::Instance,
    surface_khr: SurfaceKHR,
) -> (PhysicalDevice, QueueFamiliesIndices) {
    let devices = unsafe { instance.enumerate_physical_devices().unwrap() };

    let opt_device = devices.iter().find(|device| {
        let has_base_needs = is_device_suitable(instance, surface, surface_khr, **device);
        let device_properties = unsafe { instance.get_physical_device_properties(**device) };
        device_properties.device_type == PhysicalDeviceType::DISCRETE_GPU && has_base_needs
    });

    let device = *(opt_device.unwrap_or(
        devices
            .iter()
            .find(|device| is_device_suitable(instance, surface, surface_khr, **device))
            .expect("No suitable physical device was found!"),
    ));

    let props = unsafe { instance.get_physical_device_properties(device) };
    log::info!("Selected physical device: {:?}", unsafe {
        CStr::from_ptr(props.device_name.as_ptr())
    });

    let (graphics, present) = find_queue_families(instance, surface, surface_khr, device);
    let queue_families_indices = QueueFamiliesIndices {
        graphics_index: graphics.unwrap(),
        present_index: present.unwrap(),
    };

    (device, queue_families_indices)
}

/// Checks if the physical device can do rendering. Ensures that there is a graphics and present
/// queue index, which may be at different indices.
fn is_device_suitable(
    instance: &Instance,
    surface: &surface::Instance,
    surface_khr: SurfaceKHR,
    device: PhysicalDevice,
) -> bool {
    let (graphics, present) = find_queue_families(instance, surface, surface_khr, device);
    let extension_support = check_device_extension_support(instance, device);
    let is_swapchain_adequate = {
        let details = SwapchainSupportDetails::new(device, surface, surface_khr);
        !details.formats.is_empty() && !details.present_modes.is_empty()
    };
    let features = unsafe { instance.get_physical_device_features(device) };
    graphics.is_some()
        && present.is_some()
        && extension_support
        && is_swapchain_adequate
        && features.sampler_anisotropy == TRUE
}

/// Queues only support a subset of commands. It finds a graphics queue and present queue that
/// can present images to the surface that is created.
fn find_queue_families(
    instance: &Instance,
    surface: &surface::Instance,
    surface_khr: SurfaceKHR,
    device: PhysicalDevice,
) -> (Option<u32>, Option<u32>) {
    let mut graphics: Option<u32> = None;
    let mut present: Option<u32> = None;

    let props = unsafe { instance.get_physical_device_queue_family_properties(device) };

    for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
        let index = index as u32;

        if family.queue_flags.contains(QueueFlags::GRAPHICS) && graphics.is_none() {
            graphics = Some(index);
        }

        let present_support = unsafe {
            surface
                .get_physical_device_surface_support(device, index, surface_khr)
                .unwrap()
        };

        if present_support && present.is_none() {
            present = Some(index);
        }

        if graphics.is_some() && present.is_some() {
            break;
        }
    }

    (graphics, present)
}

fn check_device_extension_support(instance: &Instance, device: PhysicalDevice) -> bool {
    let required_extentions = get_required_device_extensions();

    let extension_props = unsafe {
        instance
            .enumerate_device_extension_properties(device)
            .unwrap()
    };

    for required in required_extentions.iter() {
        let found = extension_props.iter().any(|ext| {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            required == &name
        });

        if !found {
            return false;
        }
    }

    true
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[inline(always)]
fn get_required_device_extensions() -> [&'static CStr; 1] {
    [ash::khr::swapchain::NAME]
}
