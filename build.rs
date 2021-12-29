use std::path::{PathBuf, Path};

// build.rs

fn main() {
    println!("Hello from build.rs");
    let glslang_validator_path = get_glslang_exe_path();
    todo!("Execute glslangvalidator and compile the shader!");
}

fn get_glslang_exe_path() -> PathBuf {
    let vulkan_sdk_dir = env!("VK_SDK_PATH");
    let path = Path::new(vulkan_sdk_dir)
        .join("bin")
        .join("glslangValidator.exe");

    println!("glslangValidator path: {:?}", path.as_os_str());
    path
}
