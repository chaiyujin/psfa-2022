struct Flags {
    texture_albedo: i32,
    shading: i32,
    COOK_BLINN: i32,
    COOK_BECKMANN: i32,
    COOK_GGX: i32,
};

struct Transform {
    mat_model: mat4x4<f32>,
    mat_view: mat4x4<f32>,
    mat_proj: mat4x4<f32>,
    mat_norm: mat4x4<f32>,
};

struct Material {
    metallic: f32,
    roughness: f32,
    rim: f32,
};

struct PointLight {
    misc: vec4<f32>,
    position: vec4<f32>,
    color: vec4<f32>,
};

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec3<f32>,
};

struct VertexOutput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) v_pos: vec3<f32>,
    @location(3) v_normal: vec3<f32>,
    @builtin(position) clip_position: vec4<f32>,
};

let PI: f32 = 3.1415926;

// Visible in vs and fs.
@group(0) @binding(0)
var<uniform> r_flags: Flags;
@group(0) @binding(1)
var<uniform> r_transform: Transform;
@group(0) @binding(2)
var<uniform> r_material: Material;
@group(0) @binding(3)
var<storage, read> r_point_lights: array<PointLight>;

// Only visible in fs.
@group(1) @binding(0)
var r_sampler: sampler;
@group(1) @binding(1)
var r_texture_albedo: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = (
        r_transform.mat_proj
        * r_transform.mat_view
        * r_transform.mat_model
        * vec4<f32>(in.pos, 1.0)
    );
    out.v_pos = (
        r_transform.mat_view
        * r_transform.mat_model
        * vec4<f32>(in.pos, 1.0)
    ).xyz;
    out.v_normal = (
        r_transform.mat_norm
        * vec4<f32>(in.normal, 1.0)
    ).xyz;
    out.uv = in.uv;
    out.color = vec4<f32>(in.color, 1.0);
    return out;
}

fn num_point_lights() -> u32 {
    return arrayLength(&r_point_lights);
}


// handy value clamping to 0 - 1 range
fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

// phong (lambertian) diffuse term
fn phong_diffuse() -> f32 {
    return (1.0 / PI);
}

// compute fresnel specular factor for given base specular and product
// product could be NdV or VdH depending on used technique
fn fresnel_factor(f0: vec3<f32>, product: f32) -> vec3<f32> {
    return mix(f0, vec3(1.0), pow(1.01 - product, 5.0));
}

// following functions are copies of UE4
// for computing cook-torrance specular lighting terms

fn D_blinn(roughness: f32, NdH: f32) -> f32 {
    let m = roughness * roughness;
    let m2 = m * m;
    let n = 2.0 / m2 - 2.0;
    return (n + 2.0) / (2.0 * PI) * pow(NdH, n);
}

fn D_beckmann(roughness: f32, NdH: f32) -> f32 {
    let m = roughness * roughness;
    let m2 = m * m;
    let NdH2 = NdH * NdH;
    return exp((NdH2 - 1.0) / (m2 * NdH2)) / (PI * m2 * NdH2 * NdH2);
}

fn D_GGX(roughness: f32, NdH: f32) -> f32 {
    let m = roughness * roughness;
    let m2 = m * m;
    let d = (NdH * m2 - NdH) * NdH + 1.0;
    return m2 / (PI * d * d);
}

fn G_schlick(roughness: f32, NdV: f32, NdL: f32) -> f32 {
    let k: f32 = roughness * roughness * 0.5;
    let V: f32 = NdV * (1.0 - k) + k;
    let L: f32 = NdL * (1.0 - k) + k;
    return 0.25 / (V * L);
}

// simple phong specular calculation with normalization
fn phong_specular(
    V: vec3<f32>,
    L: vec3<f32>,
    N: vec3<f32>,
    specular: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let R: vec3<f32> = reflect(-L, N);
    let spec: f32 = max(0.0, dot(V, R));
    let k: f32 = 1.999 / (roughness * roughness);
    return min(1.0, 3.0 * 0.0398 * k) * pow(spec, min(10000.0, k)) * specular;
}

// simple blinn specular calculation with normalization
fn blinn_specular(NdH: f32, specular: vec3<f32>, roughness: f32) -> vec3<f32> {
    let k: f32 = 1.999 / (roughness * roughness);
    return min(1.0, 3.0 * 0.0398 * k) * pow(NdH, min(10000.0, k)) * specular;
}

// cook-torrance specular calculation
fn cooktorrance_specular(
    NdL: f32,
    NdV: f32,
    NdH: f32,
    specular: vec3<f32>,
    roughness: f32,
    rim: f32,
) -> vec3<f32> {
    var D: f32;
    if r_flags.COOK_BLINN > 0 {
        D = D_blinn(roughness, NdH);
    }
    else if r_flags.COOK_BECKMANN > 0 {
        D = D_beckmann(roughness, NdH);
    }
    else if r_flags.COOK_GGX > 0 {
        D = D_GGX(roughness, NdH);
    }

    let G = G_schlick(roughness, NdV, NdL);
    let den_rim = mix(1.0 - roughness * rim * 0.9, 1.0, NdV);
    return (1.0 / den_rim) * specular * G * D;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // NOTE: base color.
    var base: vec3<f32>;
    if r_flags.texture_albedo > 0 {
        base = textureSample(r_texture_albedo, r_sampler, in.uv).rgb;
        base = pow(base.rgb, vec3<f32>(2.2));  // gamma correct
    } else {
        base = in.color.rgb;
    }

    if r_flags.shading <= 0 {
        return vec4<f32>(base, 1.0);
    }

    let v_pos: vec3<f32> = in.v_pos;
    let v_normal: vec3<f32> = in.v_normal;
    var reflected_light = vec3<f32>(0.0);
    var diffuse_light = vec3<f32>(0.0);

    for (var i = 0; i < i32(num_point_lights()); i++) {
        // Guard: if the point light is not open, continue.
        if r_point_lights[i].misc.x <= 0.0 { continue; }
        let light_pos: vec3<f32> = r_point_lights[i].position.xyz;
        let light_color: vec3<f32> = r_point_lights[i].color.rgb;
        let light_strength: f32 = r_point_lights[i].misc.x;

        let local_light_pos = (r_transform.mat_view * vec4<f32>(light_pos, 1.0)).xyz;

        // light attenuation
        let A: f32 = light_strength / dot(local_light_pos - v_pos, local_light_pos - v_pos);

        // L, V, H vectors
        let L = normalize(local_light_pos - v_pos);
        let V = normalize(-v_pos);
        let H = normalize(L + V);
        let nn = normalize(v_normal);

        // let nb = normalize(v_binormal);
        // let tbn = mat3x3(nb, cross(nn, nb), nn);

        let N = nn;

        let roughness = r_material.roughness;

        let metallic = r_material.metallic;

        // mix between metal and non-metal material, for non-metal
        // constant base specular factor of 0.04 grey is used
        let specular = mix(vec3<f32>(0.04), base.xyz, metallic);

        // compute material reflectance
        let NdL = max(0.0, dot(N, L));
        let NdV = max(0.001, dot(N, V));
        let NdH = max(0.001, dot(N, H));
        let HdV = max(0.001, dot(H, V));
        let LdV = max(0.001, dot(L, V));

        // Phong
        let specfresnel = fresnel_factor(specular, NdV);
        let specref_ = phong_specular(V, L, N, specfresnel, roughness);

// #ifdef BLINN
//         // specular reflectance with BLINN
//         vec3 specfresnel = fresnel_factor(specular, HdV);
//         vec3 specref_ = blinn_specular(NdH, specfresnel, roughness);
// #endif

// #ifdef COOK
//         // specular reflectance with COOK-TORRANCE
//         vec3 specfresnel = fresnel_factor(specular, HdV);
//         vec3 specref_ = cooktorrance_specular(NdL, NdV, NdH, specfresnel, roughness);
// #endif

        let specref: vec3<f32> = specref_ * vec3<f32>(NdL);

        // diffuse is common for any model
        let diffref: vec3<f32> = (vec3<f32>(1.0) - specfresnel) * phong_diffuse() * NdL;

        // point light
        var scaled_light_color = light_color * A;
        reflected_light += specref * scaled_light_color;
        diffuse_light += diffref * scaled_light_color;
    }

    let value = (
        diffuse_light * mix(base.rgb, vec3<f32>(0.0), r_material.metallic)
        + reflected_light
    );
    return vec4<f32>(value.rgb, 1.0);
}
