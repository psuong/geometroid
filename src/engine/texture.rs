use ash::{
    vk::{DeviceMemory, Image, ImageView, Sampler},
    Device,
};

#[derive(Clone, Copy)]
pub struct Texture {
    pub image: Image,
    pub memory: DeviceMemory,
    pub view: ImageView,
    pub sampler: Option<Sampler>,
}

impl Texture {
    pub fn new(
        image: Image,
        memory: DeviceMemory,
        view: ImageView,
        sampler: Option<Sampler>,
    ) -> Self {
        Texture {
            image,
            memory,
            view,
            sampler,
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if let Some(sampler) = self.sampler.take() {
                device.destroy_sampler(sampler, None);
            }
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
