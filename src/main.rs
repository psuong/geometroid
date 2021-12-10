mod engine;

use env_logger::{Builder, Target};
use winit::{
    dpi::PhysicalSize,
    window::{WindowBuilder}, event_loop::EventLoop
};

use crate::engine::{Engine};

fn init_logger(target: Target) {
    let mut builder = Builder::from_default_env();
    builder.target(target);
    builder.init();
}

fn main() {
    init_logger(Target::Stdout);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Geometroid")
        .with_inner_size(PhysicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    match Engine::new(&window) {
        Ok(mut engine) => engine.update(),
        Err(error) => log::error!("Failed to create application due to: {}", error),
    }
}
