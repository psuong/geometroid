use ash::vk::{Buffer, BufferUsageFlags, CommandPool, DeviceMemory, Queue};

use super::{context::VkContext, memory::create_device_local_buffer_with_data, render::Mesh};

pub struct RenderParams {
    pub vertex_buffer: Buffer,
    pub vertex_buffer_memory: DeviceMemory,
    pub index_buffer: Buffer,
    pub index_buffer_memory: DeviceMemory,
    pub index_count: usize,
}

impl RenderParams {
    pub fn new(
        vk_context: &VkContext,
        transfer_queue: Queue,
        command_pool: CommandPool,
        mesh: &Mesh,
    ) -> Self {
        let (vertex_buffer, vertex_buffer_memory) = create_device_local_buffer_with_data::<u32, _>(
            vk_context,
            command_pool,
            transfer_queue,
            BufferUsageFlags::VERTEX_BUFFER,
            &mesh.vertices,
        );

        let (index_buffer, index_buffer_memory) = create_device_local_buffer_with_data::<u32, _>(
            vk_context,
            command_pool,
            transfer_queue,
            BufferUsageFlags::INDEX_BUFFER,
            &mesh.indices,
        );

        RenderParams {
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count: mesh.index_count(),
        }
    }
}
