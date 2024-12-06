use ash::vk::{Buffer, DeviceMemory};

pub struct RenderParams {
    pub vertex_buffer: Buffer,
    pub vertex_buffer_memory: DeviceMemory,
    pub index_buffer: Buffer,
    pub index_buffer_memory: DeviceMemory,
    pub index_count: usize
}

