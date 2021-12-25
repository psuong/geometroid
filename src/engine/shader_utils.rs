use std::{fs::File, path::{self, Path}};

use ash::util::read_spv;

fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Vec<u32> {
    let mut file = File::open(path).unwrap();
    read_spv(&mut file).unwrap()
}
