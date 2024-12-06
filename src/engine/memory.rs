use ash::{
    util::Align,
    vk::{
        Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, CommandBuffer,
        CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsageFlags, CommandPool, DeviceMemory, DeviceSize, Fence, MemoryAllocateInfo,
        MemoryMapFlags, MemoryPropertyFlags, MemoryRequirements, PhysicalDeviceMemoryProperties,
        Queue, SharingMode, SubmitInfo,
    },
    Device as AshDevice,
};

use crate::to_array;

use super::context::VkContext;

pub fn create_device_local_buffer_with_data<A, T: Copy>(
    vk_context: &VkContext,
    command_pool: CommandPool,
    transfer_queue: Queue,
    usage: BufferUsageFlags,
    data: &[T],
) -> (Buffer, DeviceMemory) {
    let size = size_of_val(data) as DeviceSize;

    let (staging_buffer, staging_memory, staging_mem_size) = create_buffer(
        vk_context,
        size,
        BufferUsageFlags::TRANSFER_SRC,
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
    );

    unsafe {
        let data_ptr = vk_context
            .device_ref()
            .map_memory(staging_memory, 0, size, MemoryMapFlags::empty())
            .unwrap();
        let mut align = Align::new(data_ptr, align_of::<A>() as _, staging_mem_size);
        align.copy_from_slice(data);
        vk_context.device_ref().unmap_memory(staging_memory);
    };

    let (buffer, memory, _) = create_buffer(
        vk_context,
        size,
        BufferUsageFlags::TRANSFER_DST | usage,
        MemoryPropertyFlags::DEVICE_LOCAL,
    );

    // Copy from staging -> buffer - this will hold the Vertex data
    copy_buffer(
        vk_context.device_ref(),
        command_pool,
        transfer_queue,
        staging_buffer,
        buffer,
        size,
    );

    // Clean up the staging buffer b/c we've already copied the data!
    unsafe {
        vk_context.device_ref().destroy_buffer(staging_buffer, None);
        vk_context.device_ref().free_memory(staging_memory, None);
    };
    (buffer, memory)
}

/// Creates a buffer and allocates the memory required.
/// # Returns
/// The buffer, its memory and the actual size in bytes of the allocated memory since it may
/// be different from the requested size.
pub fn create_buffer(
    vk_context: &VkContext,
    size: DeviceSize,
    usage: BufferUsageFlags,
    mem_properties: MemoryPropertyFlags,
) -> (Buffer, DeviceMemory, DeviceSize) {
    let buffer = {
        let buffer_info = BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE);

        unsafe {
            vk_context
                .device_ref()
                .create_buffer(&buffer_info, None)
                .unwrap()
        }
    };

    let mem_requirements = unsafe {
        vk_context
            .device_ref()
            .get_buffer_memory_requirements(buffer)
    };
    let memory = {
        let mem_type = find_memory_type(
            mem_requirements,
            vk_context.get_mem_properties(),
            mem_properties,
        );
        let alloc_info = MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);

        unsafe {
            vk_context
                .device_ref()
                .allocate_memory(&alloc_info, None)
                .unwrap()
        }
    };

    unsafe {
        vk_context
            .device_ref()
            .bind_buffer_memory(buffer, memory, 0)
            .unwrap()
    }
    (buffer, memory, mem_requirements.size)
}

/// Copies the size first bytes of src into dst
///
/// Allocates a command buffer allocated from the 'command_pool'. The command buffer is
/// submitted to the transfer_queue.
pub fn copy_buffer(
    device: &AshDevice,
    command_pool: CommandPool,
    transfer_queue: Queue,
    src: Buffer,
    dst: Buffer,
    size: DeviceSize,
) {
    execute_one_time_commands(device, command_pool, transfer_queue, |buffer| {
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        let regions = [region];

        unsafe { device.cmd_copy_buffer(buffer, src, dst, &regions) };
    });
}

/// A one time executor that takes in a lambda to execute. This can be used in multiple
/// places such as copying a buffer.
pub fn execute_one_time_commands<T: FnOnce(CommandBuffer)>(
    device: &AshDevice,
    command_pool: CommandPool,
    queue: Queue,
    executor: T,
) {
    let command_buffer = {
        let alloc_info = CommandBufferAllocateInfo::default()
            .level(CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);
        unsafe { device.allocate_command_buffers(&alloc_info).unwrap()[0] }
    };

    let command_buffers = to_array!(command_buffer);

    // Begin recording
    let begin_info =
        CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();
    }

    // Execute command (pretty much a delegate)
    executor(command_buffer);

    unsafe { device.end_command_buffer(command_buffer).unwrap() };
    let submit_info = to_array!(SubmitInfo::default().command_buffers(&command_buffers));

    unsafe {
        device
            .queue_submit(queue, &submit_info, Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap();
        device.free_command_buffers(command_pool, &command_buffers)
    };
}

/// Finds a memory type in the mem_properties that is suitable
/// for requirements and supports required_properties.
///
/// # Returns
/// The index of the memory type from mem_properties.
fn find_memory_type(
    requirements: MemoryRequirements,
    mem_properties: PhysicalDeviceMemoryProperties,
    required_properties: MemoryPropertyFlags,
) -> u32 {
    for i in 0..mem_properties.memory_type_count {
        if requirements.memory_type_bits & (1 << i) != 0
            && mem_properties.memory_types[i as usize]
                .property_flags
                .contains(required_properties)
        {
            return i;
        }
    }
    panic!("Failed to find a suitable memory type!");
}
