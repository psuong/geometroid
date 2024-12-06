#[macro_export]
macro_rules! to_array {
    // Case for a single value, creating an array of that value
    ($val:expr) => {
        [$val]
    };

    // Case for multiple values, creating an array of those values
    ($($val:expr),*) => {
        [$($val),*]
    };
}
