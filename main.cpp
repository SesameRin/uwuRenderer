#include <iostream>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <random>
#include <cmath>
#include "model.h"
#include "our_gl.h"
#include "tgaimage.h"

constexpr double PI = 3.14159265358979323846;
constexpr int width = 800;
constexpr int height = 800;

// ==========================================
// Shaders 定义区
// ==========================================

// DepthShader: 用于 Pass 1 生成 Shadow Map
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

// PhongShader: 用于 Pass 2 生成彩色画面与 PCSS 阴影
struct PhongShader : public IShader
{
    Model &model;
    vec3 l;
    mat<4, 4> uniform_M_inv_T;
    double e = 35.0;

    mat<4, 4> LightMatrix;
    std::vector<double> &shadow_map;
    int shadow_w, shadow_h;
    double sm_bias = 0.001;

    vec2 varying_uv[3];
    vec3 varying_nrm[3];
    vec3 tri[3];
    vec3 varying_world_pos[3];

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

        vec4 world_pos = ModelMatrix * gl_Vertex;
        varying_world_pos[nthvert] = vec3{world_pos.x, world_pos.y, world_pos.z};

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
        vec3 bn = normalized(varying_nrm[0] * bar.x + varying_nrm[1] * bar.y + varying_nrm[2] * bar.z);

        mat<2, 3> E;
        E[0] = tri[1] - tri[0];
        E[1] = tri[2] - tri[0];
        mat<2, 2> U;
        U[0] = varying_uv[1] - varying_uv[0];
        U[1] = varying_uv[2] - varying_uv[0];
        mat<2, 3> T = U.invert() * E;

        mat<3, 3> D;
        D[0] = normalized(T[0]);
        D[1] = normalized(T[1]);
        D[2] = bn;

        vec3 n_tex = model.normal(uv);
        vec3 n = normalized(D.transpose() * n_tex);

        // --- PCSS 阴影 ---
        vec3 world_pos = varying_world_pos[0] * bar.x + varying_world_pos[1] * bar.y + varying_world_pos[2] * bar.z;
        vec4 frag_light_space = LightMatrix * vec4{world_pos.x, world_pos.y, world_pos.z, 1.0};
        vec3 p = vec3{frag_light_space.x / frag_light_space.w, frag_light_space.y / frag_light_space.w, frag_light_space.z / frag_light_space.w};

        double shadow_coeff = 1.0;
        int px = int(p.x);
        int py = int(p.y);

        if (frag_light_space.w > 0.0 && px >= 0 && px < shadow_w && py >= 0 && py < shadow_h)
        {
            double current_depth = p.z - 1.0;
            int blocker_search_radius = 2;
            int blocker_count = 0;
            double blocker_depth_sum = 0.0;

            for (int i = -blocker_search_radius; i <= blocker_search_radius; i++)
            {
                for (int j = -blocker_search_radius; j <= blocker_search_radius; j++)
                {
                    int nx = px + i;
                    int ny = py + j;
                    if (nx < 0 || nx >= shadow_w || ny < 0 || ny >= shadow_h)
                        continue;

                    double sample_depth = shadow_map[nx + ny * shadow_w];
                    if (current_depth > sample_depth + sm_bias)
                    {
                        blocker_count++;
                        blocker_depth_sum += sample_depth;
                    }
                }
            }

            if (blocker_count > 0)
            {
                double avg_blocker_depth = blocker_depth_sum / blocker_count;
                double d_receiver = (current_depth + 1.0) / 2.0;
                double d_blocker = (avg_blocker_depth + 1.0) / 2.0;
                double w_light = 12.0;

                double penumbra_ratio = std::max(0.0, (d_receiver - d_blocker)) / d_blocker;
                double filter_radius_float = penumbra_ratio * w_light;
                int filter_radius = std::min(6, (int)std::ceil(filter_radius_float));

                double visibility = 0.0;
                int sample_count = 0;
                for (int i = -filter_radius; i <= filter_radius; i++)
                {
                    for (int j = -filter_radius; j <= filter_radius; j++)
                    {
                        int nx = px + i;
                        int ny = py + j;
                        if (nx < 0 || nx >= shadow_w || ny < 0 || ny >= shadow_h)
                            continue;

                        if (current_depth <= shadow_map[nx + ny * shadow_w] + sm_bias)
                            visibility += 1.0;
                        sample_count++;
                    }
                }
                if (sample_count > 0)
                    visibility /= sample_count;
                shadow_coeff = visibility;
            }
        }

        // --- 光照计算 ---
        vec3 r = normalized(n * (n * l) * 2.0 - l);
        double ambient = 0.4;
        double diff = std::max(0.0, n * l);
        double spec_weight = IShader::sample2D(model.specular(), uv)[0] / 255.0;
        double spec = (3.0 * spec_weight) * std::pow(std::max(r.z, 0.0), e);

        TGAColor base_color = IShader::sample2D(model.diffuse(), uv);
        for (int i = 0; i < 3; i++)
        {
            base_color[i] = std::min<int>(255, base_color[i] * (ambient + (diff + spec) * shadow_coeff));
        }

        return {false, base_color};
    }
};


int main(int argc, char **argv)
{
    Model diablo("obj/diablo3_pose/diablo3_pose.obj");
    Model floor("obj/floor.obj");

    TGAImage framebuffer(width, height, TGAImage::RGB);

    vec3 light_dir{1.2, 0.6, 0.8};
    vec3 eye{0, 0.5, 2.0};
    vec3 center{0, 0, 0};
    vec3 up{0, 1, 0};

    int shadow_res = 1024;
    std::vector<double> shadow_buffer;
    mat<4, 4> LightMatrix;

    // ---------------------------------------------------------
    // Pass 1: 生成 Shadow Map (光源视角)
    // ---------------------------------------------------------
    std::cout << "Rendering Pass 1: Shadow Map..." << std::endl;
    {
        vec3 light_pos = light_dir * 3.0;
        set_view_matrix(light_pos, center, up);
        set_projection_matrix(60.0, 1.0, 0.1, 25.0);
        set_viewport_matrix(0, 0, shadow_res, shadow_res);
        init_zbuffer(shadow_res, shadow_res);
        TGAImage trash(shadow_res, shadow_res, TGAImage::RGB);

        // 渲染恶魔深度
        set_model_matrix(PI / 6.0, {0.2, 0, 0.3});
        DepthShader depth_shader_diablo(diablo);
        for (int i = 0; i < diablo.nfaces(); i++)
        {
            vec4 clip_coords[3];
            for (int j = 0; j < 3; j++)
                clip_coords[j] = depth_shader_diablo.vertex(i, j);
            rasterize(clip_coords, depth_shader_diablo, trash);
        }

        // 渲染地板深度
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

    // ---------------------------------------------------------
    // Pass 2: 场景渲染 (相机视角 + 阴影着色)
    // ---------------------------------------------------------
    std::cout << "Rendering Pass 2: Scene Shading..." << std::endl;
    {
        set_view_matrix(eye, center, up);
        set_projection_matrix(60.0, (double)width / height, 0.1, 25.0);
        set_viewport_matrix(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
        init_zbuffer(width, height);

        // 渲染恶魔
        set_model_matrix(PI / 6.0, {0.2, 0, 0.3});
        PhongShader shader_diablo(light_dir, diablo, LightMatrix, shadow_buffer, shadow_res, shadow_res);
        for (int i = 0; i < diablo.nfaces(); i++)
        {
            vec4 clip_coords[3];
            for (int j = 0; j < 3; j++)
                clip_coords[j] = shader_diablo.vertex(i, j);
            rasterize(clip_coords, shader_diablo, framebuffer);
        }

        // 渲染地板
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

    // ----------------------------
    // Pass 3: SSAO 屏幕空间后处理 
    // ----------------------------

    // 这里是简化的ssao，在ndc空间采样球体，正常来说应该在真实的view space 或是 world space中进行采样，
    // 然后通过管线得到屏幕上对应的像素再查询深度比较
    std::cout << "Rendering Pass 3: Screen Space Ambient Occlusion (SSAO)..." << std::endl;
    {
        constexpr double ao_radius = 0.05; // 采样半径 (NDC 空间内衡量)
        constexpr int nsamples = 64;       // 采样数，保证画面不会太噪

        auto smoothstep = [](double edge0, double edge1, double x)
        {
            double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        };

        // 避免多线程竞争，把随机生成器丢进内层，或者不使用 omp。此处用普通的单线程遍历
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                int idx = x + y * width;
                double z = zbuffer[idx];

                // 跳过背景 (zbuffer 的初始极大值)
                if (z > 1.0)
                    continue;

                // 核心安全映射：将屏幕坐标安全退回 NDC 空间
                double ndc_x = (x - VpMatrix[0][3]) / VpMatrix[0][0];
                double ndc_y = (y - VpMatrix[1][3]) / VpMatrix[1][1];
                vec4 fragment_ndc = {ndc_x, ndc_y, z, 1.0};

                double vote = 0.0;
                double voters = 0.0;

                // 每次循环初始化随机数，保证纯粹的随机分布
                static std::mt19937 gen(1337);
                std::uniform_real_distribution<double> dist(-ao_radius, ao_radius);

                for (int i = 0; i < nsamples; i++)
                {
                    // 在 NDC 空间的小球内撒点
                    vec4 sample_ndc = fragment_ndc + vec4{dist(gen), dist(gen), dist(gen), 0.0};

                    // 将采样点投射回屏幕 2D 坐标系去查字典
                    double sx = sample_ndc.x * VpMatrix[0][0] + VpMatrix[0][3];
                    double sy = sample_ndc.y * VpMatrix[1][1] + VpMatrix[1][3];

                    int px = int(sx);
                    int py = int(sy);

                    if (px < 0 || px >= width || py < 0 || py >= height)
                        continue;

                    double sample_depth = zbuffer[px + py * width];
                    if (sample_depth > 1.0)
                        continue; // 采样到了背景，忽略

                    // Range check: 防止远方物体飞跃虚空产生错误的遮挡黑边 (Halo)
                    if (std::abs(z - sample_depth) > 5.0 * ao_radius)
                        continue;

                    voters++;
                    // 深度测试：在你的光栅化器中，Z 值越小越靠近相机 (-1近，1远)
                    // 如果查到的深度比我的采样点还小，说明有物体挡在了我的采样点前面！
                    if (sample_depth < sample_ndc.z - 0.001)
                    {
                        vote += 1.0;
                    }
                }

                // 乘以 0.4 是神来之笔：因为我们用的是整球采样，平地上必然有 50% 采样点在地下
                // 补偿后，平地就不会变得黑乎乎的。
                double ao_ratio = 1.0;
                if (voters > 0)
                    ao_ratio = 1.0 - (vote / voters) * 0.4;

                double ssao_coeff = smoothstep(0.0, 1.0, ao_ratio);

                // 最后，读出 Pass 2 画好的颜色，乘以计算出来的 SSAO 系数
                TGAColor c = framebuffer.get(x, y);
                framebuffer.set(x, y, TGAColor(c[0] * ssao_coeff, c[1] * ssao_coeff, c[2] * ssao_coeff, c[3]));
            }
        }
    }

    std::cout << "Done! Saving framebuffer.tga..." << std::endl;
    framebuffer.write_tga_file("framebuffer.tga");
    return 0;
}