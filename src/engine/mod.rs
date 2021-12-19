use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{
        self, ColorSpaceKHR, DebugUtilsMessengerEXT, DeviceCreateInfo, DeviceQueueCreateInfo,
        Extent2D, Format, Image, PhysicalDevice, PhysicalDeviceFeatures, PresentModeKHR, Queue,
        QueueFlags, SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainKHR,
    },
    Device, Entry, Instance,
};
use std::{
    error::Error,
    ffi::{CStr, CString},
};

use winit::window::Window;

mod debug;
mod utils;

use debug::{
    get_layer_names_and_pointers, setup_debug_messenger, ENABLE_VALIDATION_LAYERS, REQUIRED_LAYERS,
};

use crate::{WIDTH, common::HEIGHT, engine::utils::SwapchainSupportDetails};

pub struct Engine {
    _entry: Entry,
    _physical_device: PhysicalDevice,
    _graphics_queue: Queue,
    instance: Instance,
    debug_report_callback: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
    surface: Surface,
    surface_khr: SurfaceKHR,
    logical_device: Device,
    swapchain: Swapchain,
    swapchain_khr: SwapchainKHR,
}

impl Engine {
    pub fn new(_window: &Window) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::new().expect("Failed to create entry") };
        let instance = Self::create_instance(&entry).unwrap();
        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let surface = Surface::new(&entry, &instance);
        let surface_khr =
            unsafe { ash_window::create_surface(&entry, &instance, _window, None).unwrap() };

        let physical_device = Self::pick_physical_device(&instance, &surface, surface_khr);
        let (logical_device, graphics_queue) = Self::create_logical_device_with_graphics_queue(
            &instance,
            &surface,
            surface_khr,
            physical_device,
        );

        let (swapchain, swapchain_khr, format, extent, images) = Self::create_swapchain_and_images(
            &instance, 
            physical_device, 
            &logical_device, 
            &surface, 
            surface_khr);

        Ok(Engine {
            _entry: entry,
            _physical_device: physical_device,
            _graphics_queue: graphics_queue,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            logical_device,
            swapchain,
            swapchain_khr
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
    fn pick_physical_device(
        instance: &Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
    ) -> PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(&instance, &surface, surface_khr, *device))
            .expect("No suitable physical device!");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        device
    }

    /// Checks if the physical device can do rendering. Ensures that there is a graphics and present
    /// queue index, which may be at different indices.
    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
        device: PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        graphics.is_some() && present.is_some()
    }

    /// Queues only support a subset of commands. It finds a graphics queue and present queue that
    /// can present images to the surface that is created.
    fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
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

    /// Create a logical device based on the validation layers that are enabled.
    /// The logical device will interact with the physical device (our discrete video card).
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
        device: PhysicalDevice,
    ) -> (Device, Queue) {
        let (graphics_index, present_index) =
            Self::find_queue_families(instance, surface, surface_khr, device);

        let graphics_family_index = graphics_index.unwrap();
        let present_family_index = present_index.unwrap();
        let queue_priorities: [f32; 1] = [1.0f32];

        let queue_create_infos: Vec<DeviceQueueCreateInfo> = {
            let mut indices: Vec<u32> = vec![graphics_family_index, present_family_index];
            indices.dedup();

            indices
                .iter()
                .map(|index| {
                    DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect()
        };

        let device_features = PhysicalDeviceFeatures::builder().build();
        let (_layer_names, layer_ptrs) = get_layer_names_and_pointers();

        let mut device_create_info_builder = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
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

        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        (device, graphics_queue)
    }

    /// Does exactly what it says, chooses the swap chain format based on whatever is available.
    /// If R8G8R8A8 is available then it is selected.
    fn choose_swapchain_surface_format(available_formats: &[SurfaceFormatKHR]) -> SurfaceFormatKHR {
        if available_formats.len() == 1 && available_formats[0].format == Format::UNDEFINED {
            return SurfaceFormatKHR {
                format: Format::B8G8R8A8_UNORM,
                color_space: ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }

        *available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8_UNORM
                    && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    /// Chooses the swapchain present mode. MAILBOX -> FIFO -> IMMEDIATE are the order of priority
    /// when chosen.
    ///
    /// IMMEDIATE means that the moment the image is submitted to the screen, the image is shown
    /// right away! May cause tearing.
    ///
    /// FIFO follows the queue priority. This is pretty much like most modern games with VSYNC.
    /// Submit - if queue is full, wait until queue is emptied.
    ///
    /// MAILBOX is like a queue & immediate mode. If the presentation queue is filled to the brim,
    /// then we just overwrite whatever is in queue.
    fn choose_swapchain_surface_present_mode(
        available_present_modes: &[PresentModeKHR],
    ) -> PresentModeKHR {
        if available_present_modes.contains(&PresentModeKHR::MAILBOX) {
            PresentModeKHR::MAILBOX
        } else if available_present_modes.contains(&PresentModeKHR::FIFO) {
            PresentModeKHR::FIFO
        } else {
            PresentModeKHR::IMMEDIATE
        }
    }

    /// Creates the swapchain extent, which is typically the resolution of the windowing surface.
    /// I _think_ this is where - if I wanted to implement FSR, I can do half resolution and
    /// upscale it.
    /// TODO: Definitely try implementing FSR :)
    fn choose_swapchain_extent(capabilities: SurfaceCapabilitiesKHR) -> Extent2D {
        // Pick the animation studio.
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        let min = capabilities.min_image_extent;
        let max = capabilities.max_image_extent;

        Extent2D {
            width: WIDTH.min(max.width).max(min.width),
            height: HEIGHT.min(max.height).max(min.height),
        }
    }

    fn create_swapchain_and_images(
        instance: &Instance,
        physical_device: PhysicalDevice,
        device: &Device,
        surface: &Surface,
        surface_khr: SurfaceKHR,
    ) -> (Swapchain, SwapchainKHR, Format, Extent2D, Vec<Image>) {

        let details = SwapchainSupportDetails::query(
            physical_device, 
            surface, 
            surface_khr);

        let format = Self::choose_swapchain_surface_format(&details.formats);
        let present_mode = Self::choose_swapchain_surface_present_mode(&details.present_modes);
        let extent = Self::choose_swapchain_extent(&details.capabilities);

        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }

            preferred
        };

        todo!("Creating the swapchain is not implemented unfortunately!")
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        log::info!("Releasing engine.");
        unsafe {
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
            self.logical_device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
