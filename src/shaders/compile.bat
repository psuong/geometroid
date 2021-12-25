del shader.verg.spv
del shader.frag.spv

%VK_SDK_PATH%/bin/glslangValidator.exe -V shader.vert
%VK_SDK_PATH%/bin/glslangValidator.exe -V shader.frag

ren vert.spv shader.vert.spv
ren frag.spv shader.frag.spv
