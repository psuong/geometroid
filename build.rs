// build.rs
use std::{
    env,
    ffi::OsStr,
    fs::{self, create_dir_all, read_dir, read_to_string, File},
    io::{Result, Write},
    path::{Path, PathBuf},
    process::{Command, Output},
};

use yaml_rust2::YamlLoader;

/// Defines the Shader Stage to compile.
enum ShaderStage {
    Vertex,
    Fragment,
}

enum AssetType {
    Shaders,
    Models,
    Textures,
}

struct Source {
    root: PathBuf,
    build_log: PathBuf,
    config: PathBuf,
}

impl Source {
    pub fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let build_log = root.clone().join("build.log");
        let config = root.clone().join("build-config.yml");
        Source {
            root,
            build_log,
            config,
        }
    }

    fn read_key(&self, asset_type: AssetType) -> PathBuf {
        let asset = match asset_type {
            AssetType::Shaders => "shaders",
            AssetType::Models => "models",
            AssetType::Textures => "textures",
        };

        let content = read_to_string(self.config.as_path()).unwrap();
        let docs = YamlLoader::load_from_str(&content).unwrap();
        let doc = &docs[0];

        let mut path_buffer = PathBuf::new();
        path_buffer.push(self.root.as_path());
        let trailing_path = doc["source"][asset].as_str().unwrap();
        path_buffer.push(Path::new(&trailing_path));

        path_buffer
    }

    pub fn shader_src(&self) -> PathBuf {
        self.read_key(AssetType::Shaders)
    }

    pub fn model_src(&self) -> PathBuf {
        self.read_key(AssetType::Models)
    }

    pub fn texture_src(&self) -> PathBuf {
        self.read_key(AssetType::Textures)
    }
}

fn main() {
    let source = Source::new();
    let shader_dir_path = source.shader_src();

    let mut log_messages: Vec<String> = Vec::with_capacity(64);

    log_messages.push(format!("Root Directory: {}", source.root.to_str().unwrap()));
    log_messages.push(format!(
        "Shader Directory: {}",
        source.shader_src().to_str().unwrap()
    ));

    read_dir(shader_dir_path.clone())
        .unwrap()
        .map(Result::unwrap)
        .filter(|dir| dir.file_type().unwrap().is_file())
        .filter(|dir| dir.path().extension() == Some(OsStr::new("hlsl")))
        .for_each(|dir| {
            let dir_path = dir.path();
            let file_stem = dir_path.file_stem().unwrap().to_str().unwrap();
            let frag_output_name = format!("{}-frag.spv", &file_stem);
            compile_shader(
                ShaderStage::Fragment,
                &shader_dir_path,
                &dir_path,
                &frag_output_name,
                &source,
                &mut log_messages,
            );

            let vert_output_name = format!("{}-vert.spv", &file_stem);
            compile_shader(
                ShaderStage::Vertex,
                &shader_dir_path,
                &dir_path,
                &vert_output_name,
                &source,
                &mut log_messages,
            );
        });

    let current_dir = std::env::current_dir();
    log_messages.push(format!("Current build dir: {:?}", current_dir));

    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    let mut path_buffer = PathBuf::new();

    path_buffer.push(source.root.to_str().unwrap());
    path_buffer.push(target_dir);
    path_buffer.push(env::var("PROFILE").unwrap_or_else(|_| "debug".to_string()));
    path_buffer.push("assets");

    create_directories(&mut path_buffer, AssetType::Shaders);
    create_directories(&mut path_buffer, AssetType::Models);
    create_directories(&mut path_buffer, AssetType::Textures);

    let _ = create_dir_all(path_buffer.as_path());
    log_messages.push(format!("Target Dir: {}", path_buffer.display()));

    // We have to copy our shaders to our target directory
    read_dir(shader_dir_path.clone())
        .unwrap()
        .map(Result::unwrap)
        .filter(|dir| {
            dir.file_type().unwrap().is_file() && dir.path().extension() == Some(OsStr::new("spv"))
        })
        .for_each(|dir| {
            let mut cloned = path_buffer.clone();
            cloned.push("shaders");
            cloned.push(dir.file_name().to_str().unwrap());
            let _ = fs::copy(dir.path(), cloned.as_path());
        });

    // TODO: Need to figure out asset packs
    // We have to copy out textures to our target directory
    read_dir(source.model_src())
        .unwrap()
        .map(Result::unwrap)
        .filter(|dir| {
            dir.file_type().unwrap().is_file() && dir.path().extension() == Some(OsStr::new("obj"))
        })
        .for_each(|dir| {
            let mut cloned = path_buffer.clone();
            cloned.push("models");
            cloned.push(dir.file_name().to_str().unwrap());
            let _ = fs::copy(dir.path(), cloned.as_path());
            log_messages.push(format!(
                "Models: {}, {}",
                dir.path().display(),
                cloned.as_path().display()
            ));
        });

    read_dir(source.texture_src())
        .unwrap()
        .map(Result::unwrap)
        .filter(|dir| {
            dir.file_type().unwrap().is_file() && dir.path().extension() == Some(OsStr::new("png"))
        })
        .for_each(|dir| {
            let mut cloned = path_buffer.clone();
            cloned.push("textures");
            cloned.push(dir.file_name().to_str().unwrap());
            let _ = fs::copy(dir.path(), cloned.as_path());
            log_messages.push(format!(
                "Textures: {}, {}",
                dir.path().display(),
                cloned.as_path().display()
            ));
        });

    let _ = write_messages_to_file(source.build_log.clone(), &log_messages);
}

fn create_directories(path_buffer: &mut PathBuf, asset_type: AssetType) {
    let asset = match asset_type {
        AssetType::Shaders => "shaders",
        AssetType::Models => "models",
        AssetType::Textures => "textures",
    };

    path_buffer.push(asset);
    let _ = create_dir_all(path_buffer.as_path());
    path_buffer.pop();
}

/// Determines the stage to compile based on the shader\_stage. Stores all messages into the
/// the log\_messages.
///
/// # Arguments
///
/// * `shader_stage` - The ShaderStage to compile
/// * `shader_dir_path` - The current shader's parent directory
/// * `dir_path` - The directory path to output to
/// * `shader_name` - The name of the shader to compile
/// * `source` - The source logger
/// * `log_messages` - The Vec\<String\> that stores all messages done by the shader compiler
fn compile_shader(
    shader_stage: ShaderStage,
    shader_dir_path: &PathBuf,
    dir_path: &PathBuf,
    shader_name: &str,
    source: &Source,
    log_messages: &mut Vec<String>,
) {
    let (stage_entry_point, stage_arg) = match shader_stage {
        ShaderStage::Vertex => ("vert", "vs_6_0"),
        ShaderStage::Fragment => ("frag", "ps_6_0"),
    };

    log_messages.push(format!(
        "Compiling in directory: {}, with shader name: {}, with entry point: {}",
        shader_dir_path.to_str().unwrap(),
        shader_name,
        stage_entry_point
    ));

    let result = dbg!(Command::new("dxc"))
        .current_dir(shader_dir_path)
        .arg("-spirv")
        .arg("-T")
        .arg(stage_arg)
        .arg("-E")
        .arg(stage_entry_point)
        .arg(dir_path)
        .arg("-Fo")
        .arg(shader_name)
        .arg("-fspv-extension=SPV_EXT_descriptor_indexing")
        .output();

    handle_shader_result(source.build_log.clone(), result, log_messages);
}

/// Stores all log messages of the shader compilation pipeline.
///
/// # Arguments
///
/// * `path_buffer` - The file to write logs to
/// * `result` - The state of the shader compiler
/// * `log_messages` - A Vec\<String\> of messages to store into
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
            let _ = write_messages_to_file(path_buffer, log_messages);
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
