#ifndef UNLIT_INCLUDED
#define UNLIT_INCLUDED

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION0;
    [[vk::location(1)]] float3 normal: NORMAL0;
    [[vk::location(2)]] float4 color : COLOR0;
    [[vk::location(3)]] float2 uv0 : TEXCOORD0;
};

struct VSOutput {
    float4 color : COLOR0;
    float2 uv : TEXCOORD0;
    float3 normal: NORMAL0;
    float4 positionWS : SV_POSITION;
};

struct UBO {
    float4x4 projectionMatrix;
    float4x4 modelMatrix;
    float4x4 viewMatrix;
};

cbuffer ubo : register(b0, space0) { UBO ubo; }

Texture2D textureMap : register(t1);
SamplerState sampler_TextureMap : register(s1);

float4 ComputePosition(UBO ubo, float3 vertPos) {
    return mul(ubo.projectionMatrix, mul(ubo.viewMatrix, mul(ubo.modelMatrix, float4(vertPos, 1.0))));
}

// dxc -spirv -T vs_6_0 -E main .\shader-vert.hlsl -Fo shader-vert-hlsl.spv -fspv-extension=SPV_EXT_descriptor_indexing
// VertexIndex is the same as gl_VertexIndex
VSOutput vert(VSInput input) {
    VSOutput output = (VSOutput)0;
    output.positionWS = ComputePosition(ubo, input.position);
    output.color = input.color;
    output.uv = input.uv0;
    output.normal = input.normal;
    return output;
}

half4 frag(in VSOutput input) : SV_TARGET {
    float4 textureColor =  textureMap.Sample(sampler_TextureMap, input.uv);
    textureColor = 1.0;
    float4 transformedColor = textureColor * input.color;
    return transformedColor;
}

#endif
