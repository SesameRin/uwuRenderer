#include <iostream>
#include <cstdlib>
#include "model.h"
#include "our_gl.h"
#include <limits>
#include "tgaimage.h"

constexpr double PI = 3.14159265358979323846;
constexpr int width = 800;
constexpr int height = 800;

// 支持 UV 插值和法线贴图的光照模型
struct PhongShader : public IShader
{
    Model &model;
    vec3 l;
    mat<4, 4> uniform_M_inv_T;
    double e = 35.0;

    // --- 阴影相关数据 ---
    mat<4, 4> LightMatrix;           // 从 世界空间 -> 光源屏幕空间 的变换矩阵
    std::vector<double> &shadow_map;
    int shadow_w, shadow_h;          // Shadow Map 的分辨率
    double sm_bias = 0.001; // shadow map的bias

    // --- Shader 内部的数据通道 ---
    vec2 varying_uv[3];
    vec3 varying_nrm[3];
    vec3 tri[3];               // 存 View Space 坐标 (算 TBN 用)
    vec3 varying_world_pos[3]; // 存 World Space 坐标 (算阴影用)

    PhongShader(vec3 light_dir, Model &m, mat<4, 4> light_mat, std::vector<double> &s_map, int sw, int sh)
        : model(m), LightMatrix(light_mat), shadow_map(s_map), shadow_w(sw), shadow_h(sh)
    {
        vec4 l_view = ViewMatrix * vec4{light_dir.x, light_dir.y, light_dir.z, 0.0};
        l = normalized(vec3{l_view.x, l_view.y, l_view.z});
        uniform_M_inv_T = (ViewMatrix * ModelMatrix).invert_transpose();
    }

    virtual vec4 vertex(int iface, int nthvert) override
    {
        varying_uv[nthvert] = model.uv(iface, nthvert);
        vec3 v = model.vert(iface, nthvert);
        vec4 gl_Vertex = vec4{v.x, v.y, v.z, 1.0};

        // 保存世界坐标 (World Space) 留给阴影计算用
        vec4 world_pos = ModelMatrix * gl_Vertex;
        varying_world_pos[nthvert] = vec3{world_pos.x, world_pos.y, world_pos.z};

        // 正常算相机空间坐标和法线
        vec4 v_view = ViewMatrix * world_pos;
        tri[nthvert] = vec3{v_view.x, v_view.y, v_view.z};

        vec3 n = model.normal(iface, nthvert);
        vec4 n_view = uniform_M_inv_T * vec4{n.x, n.y, n.z, 0.0};
        varying_nrm[nthvert] = normalized(vec3{n_view.x, n_view.y, n_view.z});

        return ProjMatrix * v_view;
    }

    virtual std::pair<bool, TGAColor> fragment(vec3 bar) override
    {
        vec2 uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;

        // 利用重心坐标，算出当前像素平滑的 View Space 法线 (Normal)
        vec3 bn = normalized(varying_nrm[0] * bar.x + varying_nrm[1] * bar.y + varying_nrm[2] * bar.z);

        // 构建 3D 边矩阵 E (2行3列) 和 UV 边矩阵 U (2行2列)
        mat<2, 3> E;
        E[0] = tri[1] - tri[0];
        E[1] = tri[2] - tri[0];
        mat<2, 2> U;
        U[0] = varying_uv[1] - varying_uv[0];
        U[1] = varying_uv[2] - varying_uv[0];

        mat<2, 3> T = U.invert() * E;

        // 构建 TBN 矩阵 (Darboux frame)。将 T, B, N 作为行向量塞进去
        mat<3, 3> D;
        D[0] = normalized(T[0]); // 切线 Tangent
        D[1] = normalized(T[1]); // 副切线 Bitangent
        D[2] = bn;               // 法线 Normal

        // 获取切线空间里的扰动法线 (即蓝色贴图里的相对倾斜角)
        vec3 n_tex = model.normal(uv);

        // 将切线空间法线转换为 View Space 法线
        vec3 n = normalized(D.transpose() * n_tex);

        /// Shadow Test
        // 重建当前像素的世界坐标
        vec3 world_pos = varying_world_pos[0] * bar.x + varying_world_pos[1] * bar.y + varying_world_pos[2] * bar.z;
        // 把世界坐标变换到光源的屏幕空间
        vec4 frag_light_space = LightMatrix * vec4{world_pos.x, world_pos.y, world_pos.z, 1.0};
        // 透视除法，得到真正的坐标和深度
        vec3 p = vec3{frag_light_space.x / frag_light_space.w, frag_light_space.y / frag_light_space.w, frag_light_space.z / frag_light_space.w};

        double shadow_coeff = 1.0;
        int idx = int(p.x) + int(p.y) * shadow_w;
        // 检查坐标是否在 Shadow Map 的范围内
        if (p.x >= 0 && p.x < shadow_w && p.y >= 0 && p.y < shadow_h)
        {
            double current_depth = p.z - 1.0;
            // std::cout << "cur depth: " << current_depth << std::endl;
            // std::cout << "shadow map : " << shadow_map[idx] << std::endl;
            // 加一个极小的 bias, 防止产生 Z-fighting 自遮挡问题
            if (current_depth > shadow_map[idx] + sm_bias)
            {
                shadow_coeff = 0.3; // 处于阴影中，亮度降为 30%
            }
        }

        /// 光照计算
        vec3 r = normalized(n * (n * l) * 2.0 - l);

        // 光照参数设置
        double ambient = 0.4;
        double diff = std::max(0.0, n * l);

        // 从贴图读取高光权重 (灰度图，读 [0] 通道即可)
        double spec_weight = IShader::sample2D(model.specular(), uv)[0] / 255.0;
        double spec = (3.0 * spec_weight) * std::pow(std::max(r.z, 0.0), e);

        // 从贴图读取基础颜色
        TGAColor base_color = IShader::sample2D(model.diffuse(), uv);

        for (int i = 0; i < 3; i++)
        {
            base_color[i] = std::min<int>(255, base_color[i] * (ambient + (diff + spec) * shadow_coeff));
        }

        return {false, base_color};
    }
};

// shadow mapping中第一遍光源视角使用
struct DepthShader : public IShader
{
    Model &model;
    DepthShader(Model &m) : model(m) {}

    virtual vec4 vertex(int iface, int nthvert) override
    {
        vec3 v = model.vert(iface, nthvert);
        vec4 gl_Vertex = vec4{v.x, v.y, v.z, 1.0};
        // 算出光源视角下的裁剪坐标
        return ProjMatrix * ViewMatrix * ModelMatrix * gl_Vertex;
    }

    virtual std::pair<bool, TGAColor> fragment(vec3 bar) override
    {
        return {false, TGAColor{255, 255, 255, 255}};
    }
};

int main(int argc, char **argv)
{
    // if (argc != 2)
    // {
    //     std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
    //     return 1;
    // }

    // Model model(argv[1]);

    // for convenient testing
    Model diablo("obj/diablo3_pose/diablo3_pose.obj");
    Model floor("obj/floor.obj");

    TGAImage framebuffer(width, height, TGAImage::RGB);

    // 基础参数设置
    vec3 light_dir{1.2, 0.6, 0.8};
    vec3 eye{0, 0.5, 2.0}; // 稍微抬高一点相机，俯视能更好地看清地板阴影
    vec3 center{0, 0, 0};
    vec3 up{0, 1, 0};

    int shadow_res = 1024; // Shadow Map 分辨率，1024 足够清晰了，不用像作者那样开 8000
    std::vector<double> shadow_buffer;
    mat<4, 4> LightMatrix;

    // ==========================================
    // Pass 1: 生成 Shadow Map
    // ==========================================
    {
        // 假设光源在远处
        vec3 light_pos = light_dir * 3.0;

        set_view_matrix(light_pos, center, up); // 相机移到光源位置
        set_projection_matrix(60.0, 1.0, 0.1, 15.0);
        set_viewport_matrix(0, 0, shadow_res, shadow_res); // 使用阴影分辨率
        init_zbuffer(shadow_res, shadow_res);
        TGAImage trash(shadow_res, shadow_res, TGAImage::RGB); // 我们不需要颜色，随便建一个

        // diablo
        set_model_matrix(PI / 6.0, {0.2, 0, 0.3});
        DepthShader depth_shader_diablo(diablo);
        for (int i = 0; i < diablo.nfaces(); i++)
        {
            vec4 clip_coords[3];
            for (int j = 0; j < 3; j++)
            clip_coords[j] = depth_shader_diablo.vertex(i, j);
            rasterize(clip_coords, depth_shader_diablo, trash);
        }
        
        // floor
        set_model_matrix(0.0, {0, 0, 0});
        DepthShader depth_shader_floor(floor);
        for (int i = 0; i < floor.nfaces(); i++)
        {
            vec4 clip_coords[3];
            for (int j = 0; j < 3; j++)
                clip_coords[j] = depth_shader_floor.vertex(i, j);
            rasterize(clip_coords, depth_shader_floor, trash);
        }

        shadow_buffer = zbuffer;
        LightMatrix = VpMatrix * ProjMatrix * ViewMatrix;
    }

    // ==========================================
    // Pass 2: 常规相机渲染 (带阴影测试)
    // ==========================================
    {
        set_model_matrix(PI / 6.0);
        set_view_matrix(eye, center, up); // 恢复正常的相机位置
        set_projection_matrix(60.0, (double)width / height, 0.1, 15.0);
        set_viewport_matrix(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
        init_zbuffer(width, height); // 清空 zbuffer 给真正的画面用

        // 实例化主 Shader
        
        // diablo
        set_model_matrix(PI / 6.0, {0.2, 0, 0.3});
        PhongShader shader_diablo(light_dir, diablo, LightMatrix, shadow_buffer, shadow_res, shadow_res);
        for (int i = 0; i < diablo.nfaces(); i++)
        {
            vec4 clip_coords[3];
            for (int j = 0; j < 3; j++)
            clip_coords[j] = shader_diablo.vertex(i, j);
            rasterize(clip_coords, shader_diablo, framebuffer);
        }
        
        // floor
        set_model_matrix(0.0, {0, 0, 0});
        PhongShader shader_floor(light_dir, floor, LightMatrix, shadow_buffer, shadow_res, shadow_res);
        for (int i = 0; i < floor.nfaces(); i++)
        {
            vec4 clip_coords[3];
            for (int j = 0; j < 3; j++)
                clip_coords[j] = shader_floor.vertex(i, j);
            rasterize(clip_coords, shader_floor, framebuffer);
        }
    }

    framebuffer.write_tga_file("framebuffer.tga");
    return 0;

    // // 可视化 z-buffer (直接使用外部全局变量 zbuffer)
    // TGAImage zbuffer_img(width, height, TGAImage::RGB);
    // double min_z = std::numeric_limits<double>::max();
    // double max_z = -std::numeric_limits<double>::max();

    // for (int i = 0; i < width * height; i++)
    // {
    //     if (zbuffer[i] < 100.0)
    //     {
    //         min_z = std::min(min_z, zbuffer[i]);
    //         max_z = std::max(max_z, zbuffer[i]);
    //     }
    // }

    // for (int px = 0; px < width; px++)
    // {
    //     for (int py = 0; py < height; py++)
    //     {
    //         double z = zbuffer[px + py * width];
    //         if (z < 100.0)
    //         {
    //             unsigned char c = 255 * (max_z - z) / (max_z - min_z);
    //             zbuffer_img.set(px, py, TGAColor{c, c, c, 255});
    //         }
    //     }
    // }
    // zbuffer_img.write_tga_file("zbuffer.tga");

    return 0;
}