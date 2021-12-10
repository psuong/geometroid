use ash::{vk, Entry, Instance};
use std::{ffi::CString, error::Error};
use winit::window::Window;

mod utils;

pub struct Engine {
    _entry: Entry,
    instance: Instance
}

impl Engine {
    pub fn new(_window: &Window) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::new().expect("Failed to create entry") };
        let instance = Self::create_instance(&entry).unwrap();

        Ok(Engine {
            _entry: entry,
            instance: instance
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

        let extension_names = utils::required_extension_names();

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        unsafe {
            Ok(entry.create_instance(&instance_create_info, None)?)
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        log::info!("Releasing engine.");
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}