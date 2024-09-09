use nalgebra::UnitQuaternion;
use nalgebra_glm::Vec3;

/// Rotates a point about a pivot given the degrees to rotate around the axis.
///
/// # Arguments
///
/// * `point` - The point to rotate
/// * `pivot` - The pivot to rotate the point about
/// * `degrees` - The total degrees to rotate the point about
pub fn rotate_about_point(point: Vec3, pivot: Vec3, degrees: Vec3) -> Vec3 {
    let dir = point - pivot;
    let rotation = UnitQuaternion::from_euler_angles(degrees.x.to_radians(), degrees.y.to_radians(), degrees.z.to_radians());
    // rotation.mul_vec3(dir) + pivot
    rotation * dir + pivot
}
