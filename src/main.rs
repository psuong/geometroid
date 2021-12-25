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

    let mut engine = Engine::new(&window).unwrap();

    println!("{:?}", std::env::current_exe());

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => {
                engine.update();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized { .. } => log::debug!("Resize not implemented!"),
                _ => (),
            },
            _ => (),
        }
    });
}
