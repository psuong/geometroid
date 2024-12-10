#[inline]
#[deprecated(note = "Use the macro instead")]
pub fn as_array<T>(element: T) -> [T; 1] {
    [element]
}

#[inline]
pub fn empty<T>() -> [T; 0] {
    []
}
