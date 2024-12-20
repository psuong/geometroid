use crate::engine::{context::VkContext, render::render_pipeline::RenderPipeline, Engine};
use egui::{Context, ViewportId};
use egui_ash_renderer::{Options, Renderer as UIRenderer};
use egui_winit::State;
use winit::window::Window;

mod image;

pub struct UISystem {
    pub egui_ctx: Context,
    pub egui_winit: State,
    pub renderer: UIRenderer,
}

impl UISystem {
    pub fn new(
        winit_window: &Window,
        vulkan_context: &VkContext,
        render_pipeline: &RenderPipeline,
    ) -> Self {
        let egui_ctx = Context::default();
        // TODO: Implement image loaders
        let egui_winit = State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            &winit_window,
            None,
            None,
            None,
        );

        let renderer = UIRenderer::with_default_allocator(
            &vulkan_context.instance,
            vulkan_context.physical_device,
            vulkan_context.device.clone(),
            render_pipeline.render_pass.unwrap(),
            Options {
                srgb_framebuffer: true,
                ..Default::default()
            },
        )
        .expect("Renderer failed to be created for egui-ash");

        UISystem {
            egui_ctx,
            egui_winit,
            renderer,
        }
    }
}
