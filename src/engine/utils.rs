use ash::{
    vk::{Fence, Semaphore},
    Device,
};

#[derive(Clone, Copy, Debug)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
}

#[derive(Clone, Copy)]
pub struct SyncObjects {
    pub image_available_semaphore: Semaphore,
    pub render_finished_semaphore: Semaphore,
    pub fence: Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

pub struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    pub fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    pub fn destroy(&self, device: &Device) {
        self.sync_objects
            .iter()
            .for_each(|sync| sync.destroy(device));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];
        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();
        Some(next)
    }
}
