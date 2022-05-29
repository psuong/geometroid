mod common;
mod engine;

use env_logger::{Builder, Target};
use log::LevelFilter;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use crate::common::{HEIGHT, WIDTH};
use crate::engine::Engine;

fn init_logger(target: Target) {
    Builder::from_default_env()
        .target(target)
        .filter_level(LevelFilter::Info)
        .init();
}

fn main() {
    init_logger(Target::Stdout);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Geometroid")
        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let mut engine = Engine::new(&window);
    let mut dirty_swapchain = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => {
                if dirty_swapchain {
                    let size = window.inner_size();
                    println!("{} {}", size.width, size.height);
                    if size.width > 0 && size.height > 0 {
                        engine.recreate_swapchain();
                    } else {
                        return;
                    }
                }
                dirty_swapchain = engine.draw_frame();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized { .. } => dirty_swapchain = true,
                _ => (),
            },
            Event::LoopDestroyed => engine.wait_gpu_idle(),
            _ => (),
        }
    });
}
