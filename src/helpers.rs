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

#[macro_export]
macro_rules! unwrap_value {
    ($opt:expr) => {
        match $opt {
            Some(val) => val,
            None => panic!("Stored value in the option was not initialized!"),
        }
    };
}

#[macro_export]
macro_rules! unwrap_read_write_ref {
    ($opt:expr) => {
        match $opt {
            Some(ref mut val) => val,
            None => panic!("Option was None!"),
        }
    };
}

#[macro_export]
macro_rules! unwrap_read_ref {
    ($opt:expr) => {
        match $opt {
            Some(ref val) => val,
            None => panic!("Option was None!"),
        }
    };
}
