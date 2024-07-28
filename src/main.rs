mod common;
mod engine;

use env_logger::{Builder, Target};
use log::LevelFilter;
use winit::{
    dpi::PhysicalSize,
    event::{Event, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
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

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Phylum")
        .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let mut engine = Engine::new(&window);
    let mut dirty_swapchain = false;

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::NewEvents(_) => {
                log::warn!("Use this to reset inputs")
            }
            Event::AboutToWait => {
                // TODO: Handle mouse inputs
                // Render
                {
                    if dirty_swapchain {
                        let size = window.inner_size();
                        if size.width > 0 && size.height > 0 {
                            engine.recreate_swapchain();
                        } else {
                            return;
                        }
                    }
                    dirty_swapchain = engine.draw_frame();
                }
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(..) => dirty_swapchain = true,
                WindowEvent::CursorMoved { position, .. } => {
                    log::warn!("Cursor moved not implemented!");
                    // let position: (i32, i32) = position.into();
                    // cursor_position = Some([position.0, position.1]);
                }
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, _v_lines),
                    ..
                } => {
                    // wheel_delta = Some(v_lines);
                    log::warn!("Wheel movement not implemented!");
                }
                _ => (),
            },
            Event::LoopExiting => engine.wait_gpu_idle(),
            _ => (),
        }
    });
}
