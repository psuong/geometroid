use nalgebra::{Quaternion, Vector3, Rotation3};

/// Rotates a point about a pivot given the degrees to rotate around the axis.
///
/// # Arguments
///
/// * `point` - The point to rotate
/// * `pivot` - The pivot to rotate the point about
/// * `degrees` - The total degrees to rotate the point about
pub fn rotate_about_point(point: Vec3, pivot: Vec3, degrees: Vec3) -> Vec3 {
    let dir = point - pivot;
    let rotation = Quat::from_euler(EulerRot::XYZ, degrees.x, degrees.y, degrees.z);
    rotation.mul_vec3(dir) + pivot
}
