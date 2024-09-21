use super::math::{FORWARD, RIGHT, UP};
use nalgebra_glm::Vec3;

pub struct Quad {
    pub offset: Vec3,
    pub width: Vec3,
    pub length: Vec3,
}

impl Quad {
    pub fn new(offset: Vec3, width: Vec3, length: Vec3) -> Self {
        Quad {
            offset,
            width,
            length,
        }
    }

    #[inline]
    pub fn offset_plus_length(&self) -> Vec3 {
        self.offset + self.length
    }

    #[inline]
    pub fn offset_plus_width(&self) -> Vec3 {
        self.offset + self.width
    }

    #[inline]
    pub fn offset_plus_length_width(&self) -> Vec3 {
        self.offset + self.length + self.width
    }

    #[inline]
    pub fn cross(&self) -> Vec3 {
        self.width.cross(&self.length).normalize()
    }
}

pub struct Cube {
    pub position: Vec3,
    pub dimension: Vec3,
    pub center_as_pivot: bool,
}

impl Cube {
    pub fn new(
        position: Vec3,
        length: f32,
        width: f32,
        height: f32,
        center_as_pivot: bool,
    ) -> Self {
        Cube {
            position,
            dimension: Vec3::new(width, length, height),
            center_as_pivot,
        }
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.dimension.x
    }

    #[inline]
    pub fn length(&self) -> f32 {
        self.dimension.y
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.dimension.z
    }

    pub fn corners(&self) -> (Vec3, Vec3) {
        let up = UP * self.height();
        let right = RIGHT * self.width();
        let forward = FORWARD * self.length();

        if self.center_as_pivot {
            ((up + right + forward) / 2.0, -(up + right + forward) / 2.0)
        } else {
            (Vec3::zeros(), up + right + forward)
        }
    }
}

pub struct Sphere {
    pub radius: f32,
    pub radial_segments: i32
}

impl Sphere {
    pub fn new(radius: f32, radial_segments: i32) -> Self {
        Sphere {
            radius,
            radial_segments
        }
    }
}
