use std::ops::Add;

use nalgebra::{Point3, UnitQuaternion};
use nalgebra_glm::{Mat4, Mat4x4, Vec3};

#[allow(dead_code)]
pub const UP : Vec3 = Vec3::new(0.0, 1.0, 0.0);
#[allow(dead_code)]
pub const DOWN : Vec3 = Vec3::new(0.0, -1.0, 0.0);
#[allow(dead_code)]
pub const RIGHT : Vec3 = Vec3::new(1.0, 0.0, 0.0);
#[allow(dead_code)]
pub const LEFT : Vec3 = Vec3::new(-1.0, 0.0, 0.0);
#[allow(dead_code)]
pub const FORWARD: Vec3 = Vec3::new(0.0, 0.0, 1.0);
#[allow(dead_code)]
pub const BACKWARD : Vec3 = Vec3::new(0.0, 0.0, -1.0);

/// Rotates a point about a pivot given the degrees to rotate around the axis.
///
/// # Arguments
///
/// * `point` - The point to rotate
/// * `pivot` - The pivot to rotate the point about
/// * `degrees` - The total degrees to rotate the point about
#[inline]
#[allow(dead_code)]
pub fn rotate_about_point(point: Vec3, pivot: Vec3, degrees: Vec3) -> Vec3 {
    let dir = point - pivot;
    let rotation = UnitQuaternion::from_euler_angles(
        degrees.x.to_radians(),
        degrees.y.to_radians(),
        degrees.z.to_radians(),
    );
    // rotation.mul_vec3(dir) + pivot
    rotation * dir + pivot
}

pub fn trs(translation: &Vec3, angle: f32, axis: &Vec3, scale: &Vec3) -> Mat4x4 {
    let identity = Mat4x4::identity();
    let t = nalgebra_glm::translate(&identity, &translation);
    let r = nalgebra_glm::rotate(&identity, angle, &axis);
    let s = nalgebra_glm::scale(&identity, &scale);
    t * r * s
}

#[inline]
pub fn select<T>(cond: bool, lhs: T, rhs: T) -> T where T: Add<Output = T> + Copy {
    if cond {
        lhs
    } else {
        rhs
    }
}
