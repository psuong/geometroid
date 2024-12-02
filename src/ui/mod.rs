use egui::Context;
use egui_ash_renderer::vulkan::{create_vulkan_descriptor_pool, create_vulkan_descriptor_set_layout};

use crate::{engine::Engine, App};

mod image;

pub trait Interface {
    fn new(engine: &App) -> Self;
    fn build_ui(&mut self, ctx: Context);
    fn clean(&mut self, engine: &Engine);
}

impl Interface for App {
    fn new(engine: &App) -> Self {
        // TODO: Finish setting up the image, honestly I may not even need it at the moment. I just need to emit
        // widgets for now.
        let engine = engine.vulkan.as_ref().unwrap();
        let vk_context = &engine.vk_context;

        let memory_properties = unsafe {
            engine
                .vk_context
                .instance
                .get_physical_device_memory_properties(vk_context.physical_device)
        };

        let descriptor_set_layout = create_vulkan_descriptor_set_layout(&vk_context.device).unwrap();
        let descriptor_pool = create_vulkan_descriptor_pool(&vk_context.device, 2);
    }

    fn build_ui(&mut self, ctx: Context) {
        todo!()
    }

    fn clean(&mut self, engine: &Engine) {
        todo!()
    }
}
