// build.rs
use std::{
    ffi::OsStr,
    fs::{read_dir, File},
    io::{Result, Write},
    path::PathBuf,
    process::{Command, Output},
};

struct Source {
    root: PathBuf,
    shader_log: PathBuf,
}

impl Source {
    pub fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let shader_log = root.clone().join("shader.log");
        Source { root, shader_log }
    }

    pub fn shader_src(&self) -> PathBuf {
        let path = self.root.clone().join("src").join("shaders");
        path
    }
}

fn main() {
    let source = Source::new();
    let shader_dir_path = source.shader_src();

    let mut log_messages: Vec<String> = Vec::new();

    println!("Shader directory: {:?}", shader_dir_path);

    read_dir(shader_dir_path.clone())
        .unwrap()
        .map(Result::unwrap)
        .filter(|dir| dir.file_type().unwrap().is_file())
        .filter(|dir| dir.path().extension() != Some(OsStr::new("spv")))
        .for_each(|dir| {
            let path = dir.path();
            let name = path.file_name().unwrap().to_str().unwrap();
            let output_name = format!("{}.spv", &name);

            let msg = format!("Found file {:?}.\nCompiling...", path.as_os_str());
            log_messages.push(msg);

            let result = dbg!(Command::new("glslangValidator"))
                .current_dir(&shader_dir_path)
                .arg("-V")
                .arg(&path)
                .arg("-o")
                .arg(output_name)
                .output();

            handle_shader_result(source.shader_log.clone(), result, &mut log_messages);
        });

    match write_messages_to_file(source.shader_log.clone(), &log_messages) {
        _ => {}
    };
}

fn handle_shader_result(
    path_buffer: PathBuf,
    result: Result<Output>,
    log_messages: &mut Vec<String>,
) {
    match result {
        Ok(output) => {
            if output.status.success() {
                log_messages.push("Shader compilation succeeded".to_string());
                log_messages.push(
                    String::from_utf8(output.stdout)
                        .unwrap_or("Failed to print program to stdout".to_string()),
                );
            } else {
                log_messages.push(format!(
                    "Shader compilation failed. Status: {}",
                    output.status
                ));
                log_messages.push(
                    String::from_utf8(output.stdout)
                        .unwrap_or("Failed to print program stdout".to_string()),
                );
                log_messages.push(
                    String::from_utf8(output.stderr)
                        .unwrap_or("Failed to print program stderr".to_string()),
                );
            }
        }
        Err(error) => {
            log_messages.push(format!("Failed to compile shader due to: {}", error));
            match write_messages_to_file(path_buffer, log_messages) {
                _ => {}
            }
            panic!("Shader compilation failed, please see shader.log in the root directory!");
        }
    }
}

fn write_messages_to_file(path_buffer: PathBuf, log_messages: &Vec<String>) -> Result<()> {
    let mut file = File::create(path_buffer.as_path())?;

    for message in log_messages {
        file.write_all(message.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(())
}
