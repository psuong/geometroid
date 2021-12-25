use std::{fs::File, path::Path};

use ash::{
    util::read_spv,
    vk::{ShaderModule, ShaderModuleCreateInfo},
    Device,
};

pub fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Vec<u32> {
    let mut file = File::open(path).unwrap();
    read_spv(&mut file).unwrap()
}

pub fn create_shader_module(device: &Device, code: &[u32]) -> ShaderModule {
    let create_info = ShaderModuleCreateInfo::builder().code(code).build();
    unsafe { device.create_shader_module(&create_info, None).unwrap() }
}
