use egui::{Context, Window};

use crate::{engine::Engine, App};

mod image;

pub trait ImGuiSystem {
    fn initialize_engine_ui(engine: &mut Engine);
    fn build_ui(&mut self, ctx: Context);
}

impl ImGuiSystem for App {
    fn initialize_engine_ui(engine: &mut Engine) {
        todo!()
    }

    fn build_ui(&mut self, ctx: Context) {
        Window::new("Test").show(&ctx, |ui| {
            ui.label("Hi I'm a label.");
        });
    }
}
