use ash::{
    vk::{Buffer, BufferUsageFlags, CommandPool, DeviceMemory, Queue},
    Device,
};

use crate::engine::{context::VkContext, memory::create_device_local_buffer_with_data};

use super::Mesh;

pub struct RenderDescriptor {
    pub vertex_buffer: Buffer,
    pub vertex_buffer_memory: DeviceMemory,
    pub index_buffer: Buffer,
    pub index_buffer_memory: DeviceMemory,
    pub index_count: usize,
}

impl RenderDescriptor {
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

        RenderDescriptor {
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count: mesh.index_count(),
        }
    }

    pub fn release(&mut self, device: &Device) {
        unsafe {
            device.free_memory(self.index_buffer_memory, None);
            device.destroy_buffer(self.index_buffer, None);
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);
        }
    }
}
