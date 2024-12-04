use std::error::Error;

use ash::{
    khr::swapchain::Device,
    vk::{
        AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference,
        AttachmentStoreOp, ColorSpaceKHR, CompositeAlphaFlagsKHR, Extent2D, Format, Framebuffer,
        Image, ImageAspectFlags, ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageView,
        ImageViewCreateInfo, ImageViewType, PipelineBindPoint, PipelineStageFlags, PresentModeKHR,
        RenderPass, RenderPassCreateInfo, SampleCountFlags, SharingMode, SubpassDependency,
        SubpassDescription, SwapchainCreateInfoKHR, SwapchainKHR, SUBPASS_EXTERNAL,
    },
};
use log::debug;

use crate::common::{HEIGHT, WIDTH};

use super::{
    context::VkContext,
    utils::{QueueFamiliesIndices, SwapchainProperties},
};

// NOTE: I really have no idea if this will work
struct Swapchain {
    // TODO: Store the extent here
    loader: Device,
    extent: Extent2D,
    khr: SwapchainKHR,
    images: Vec<Image>,
    image_views: Vec<ImageView>,
    render_pass: RenderPass,
    framebuffers: Vec<Framebuffer>,
}

struct SwapchainSetup {
    device: Device,
    khr: SwapchainKHR,
    extent: Extent2D,
    format: Format,
    images: Vec<Image>,
    image_views: Vec<ImageView>,
}

impl Swapchain {
    fn new(
        vulkan_context: &VkContext,
        queue_family_indices: QueueFamiliesIndices,
    ) -> Result<Self, Box<dyn Error>> {
        let swapchain_setup = create_vulkan_swapchain(vulkan_context, queue_family_indices);
    }
}

fn create_vulkan_swapchain(
    vulkan_context: &VkContext,
    queue_families_index: QueueFamiliesIndices,
) -> Result<SwapchainSetup, Box<dyn Error>> {
    debug!("Creating vulkan swapchain");
    let format = {
        let formats = unsafe {
            vulkan_context
                .surface
                .get_physical_device_surface_formats(
                    vulkan_context.physical_device,
                    vulkan_context.surface_khr,
                )
                .unwrap()
        };

        *formats
            .iter()
            .find(|format| {
                format.format == Format::R8G8B8A8_SRGB
                    && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0])
    };
    debug!("Swapchain format: {format:?}");

    let present_mode = {
        let present_modes = unsafe {
            vulkan_context
                .surface
                .get_physical_device_surface_present_modes(
                    vulkan_context.physical_device,
                    vulkan_context.surface_khr,
                )
                .unwrap()
        };

        // TODO: Check the priority of choosing a surface swapchain present mode
        if present_modes.contains(&PresentModeKHR::MAILBOX) {
            PresentModeKHR::MAILBOX
        } else if present_modes.contains(&PresentModeKHR::FIFO) {
            PresentModeKHR::FIFO
        } else {
            PresentModeKHR::IMMEDIATE
        }
    };

    let capabilities = unsafe {
        vulkan_context
            .surface
            .get_physical_device_surface_capabilities(
                vulkan_context.physical_device,
                vulkan_context.surface_khr,
            )
            .unwrap()
    };

    let extent = {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            let width = WIDTH.min(max.width).max(min.width);
            let height = HEIGHT.min(max.height).max(min.height);
            Extent2D { width, height }
        }
    };
    debug!("Swapchain Extent: {extent:?}");

    let image_count = capabilities.min_image_count;
    debug!("Swapchain image count: {image_count:?}");

    let families_indices = [
        queue_families_index.graphics_index,
        queue_families_index.present_index,
    ];

    let create_info = {
        let mut builder = SwapchainCreateInfoKHR::default()
            .surface(vulkan_context.surface_khr)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT);

        builder = if queue_families_index.graphics_index != queue_families_index.present_index {
            builder
                .image_sharing_mode(SharingMode::CONCURRENT)
                .queue_family_indices(&families_indices)
        } else {
            builder.image_sharing_mode(SharingMode::EXCLUSIVE)
        };

        builder
            .pre_transform(capabilities.current_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
    };

    let swapchain = Device::new(vulkan_context.instance_ref(), vulkan_context.device_ref());
    let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };

    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
    let views = images
        .iter()
        .map(|image| {
            let create_info = ImageViewCreateInfo::default()
                .image(*image)
                .view_type(ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe { vulkan_context.device.create_image_view(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    Ok(SwapchainSetup {
        device: swapchain,
        khr: swapchain_khr,
        extent,
        format: format.format,
        images,
        image_views: views,
    })
}

fn create_render_pass(
    device: &ash::Device,
    swapchain_properties: SwapchainProperties,
    msaa_samples: SampleCountFlags,
    depth_format: Format,
) -> RenderPass {
    let color_attachment_desc = AttachmentDescription::default()
        .format(swapchain_properties.format.format)
        .samples(msaa_samples)
        .load_op(AttachmentLoadOp::CLEAR)
        .store_op(AttachmentStoreOp::STORE)
        .initial_layout(ImageLayout::UNDEFINED)
        .final_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment_desc = AttachmentDescription::default()
        .format(depth_format)
        .samples(msaa_samples)
        .load_op(AttachmentLoadOp::CLEAR)
        .store_op(AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(AttachmentStoreOp::DONT_CARE)
        .initial_layout(ImageLayout::UNDEFINED)
        .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let resolve_attachment_desc = AttachmentDescription::default()
        .format(swapchain_properties.format.format)
        .samples(SampleCountFlags::TYPE_1)
        .load_op(AttachmentLoadOp::DONT_CARE)
        .store_op(AttachmentStoreOp::STORE)
        .stencil_load_op(AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(AttachmentStoreOp::DONT_CARE)
        .initial_layout(ImageLayout::UNDEFINED)
        .final_layout(ImageLayout::PRESENT_SRC_KHR);

    let attachment_descs = [
        color_attachment_desc,
        depth_attachment_desc,
        resolve_attachment_desc,
    ];

    // The first attachment is pretty much a color buffer
    let color_attachment_ref = AttachmentReference::default()
        .attachment(0)
        .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = [color_attachment_ref];

    let depth_attachment_ref = AttachmentReference::default()
        .attachment(1)
        .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let resolve_attachment_ref = AttachmentReference::default()
        .attachment(2)
        .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let resolve_attachment_refs = [resolve_attachment_ref];

    // Every subpass references 1 or more attachment descriptions.
    let subpass_desc = SubpassDescription::default()
        .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .resolve_attachments(&resolve_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref);
    let subpass_descs = [subpass_desc];

    // Subpasses in a render pass automatically take care of image layout transitions.
    // Transitions are controlled by subpass dependencies, which describe the memory layout &
    // execution dependencies between each subpass.
    //
    // There are typically 2 builtin dependencies that take care of the transiation at the
    // start + end of the renderpass.
    //
    // The subpass here uses the COLOR_ATTACHMENT_OUTPUT. Another way is to make the semaphore
    // to PIPELINE_STAGE_TOP_OF_PIPE_BIT instead (TODO: Look into this and remove the subpass).
    let subpass_dep = SubpassDependency::default()
        .src_subpass(SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(AccessFlags::empty())
        .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE);
    let subpass_deps = [subpass_dep];

    let render_pass_info = RenderPassCreateInfo::default()
        .attachments(&attachment_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
}

/// An abstraction of the internals of create_swapchain_image_views. All images are accessed
/// view VkImageView.
pub fn create_image_view(
    device: &ash::Device,
    image: Image,
    mip_levels: u32,
    format: Format,
    aspect_mask: ImageAspectFlags,
) -> ImageView {
    let create_info = ImageViewCreateInfo::default()
        .image(image)
        .view_type(ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe { device.create_image_view(&create_info, None).unwrap() }
}
