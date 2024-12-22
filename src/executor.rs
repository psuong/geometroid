#[derive(Clone, Copy)]
pub struct Executor {}

impl Default for Executor {
    #[inline]
    fn default() -> Self {
        Self {}
    }
}

impl Executor {
    #[inline]
    pub fn execute<F>(&self, f: &mut F) -> &Executor
    where
        F: FnMut(),
    {
        f();
        self
    }
}
