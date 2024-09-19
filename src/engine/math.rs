use nalgebra::UnitQuaternion;
use nalgebra_glm::Vec3;

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
