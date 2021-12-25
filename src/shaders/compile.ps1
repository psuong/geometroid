Remove-Item shader.verg.spv
Remove-Item shader.frag.spv

%VK_SDK_PATH%/bin/glslangValidator.exe -V shader.vert
%VK_SDK_PATH%/bin/glslangValidator.exe -V shader.frag

Rename-Item vert.spv shader.vert.spv
Rename-Item frag.spv shader.frag.spv
