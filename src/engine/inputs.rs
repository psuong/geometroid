use nalgebra_glm::IVec2;

#[allow(dead_code)]
pub struct MouseInputs {
    pub cursor_position: IVec2,
    pub cursor_delta: Option<IVec2>,
    pub wheel_delta: Option<f32>
}

#[allow(dead_code)]
impl MouseInputs {
    pub fn new() -> Self {
        MouseInputs { cursor_position: IVec2::zeros(), cursor_delta: None, wheel_delta: None }
    }
}
