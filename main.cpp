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

    // --- 新增：专门用来存放并传递 UV 坐标的数组 ---
    vec2 varying_uv[3];

    PhongShader(vec3 light_dir, Model &m) : model(m)
    {
        vec4 l_view = ViewMatrix * vec4{light_dir.x, light_dir.y, light_dir.z, 0.0};
        l = normalized(vec3{l_view.x, l_view.y, l_view.z});
        uniform_M_inv_T = (ViewMatrix * ModelMatrix).invert_transpose();
    }

    virtual vec4 vertex(int iface, int nthvert) override
    {
        // 1. 读取并保存当前顶点的 UV 坐标，供后续片元着色器插值使用
        varying_uv[nthvert] = model.uv(iface, nthvert);

        // 2. 处理顶点坐标 (保持不变)
        vec3 v = model.vert(iface, nthvert);
        vec4 gl_Vertex = vec4{v.x, v.y, v.z, 1.0};
        vec4 v_view = ViewMatrix * ModelMatrix * gl_Vertex;

        return ProjMatrix * v_view;
    }

    virtual std::pair<bool, TGAColor> fragment(vec3 bar) override
    {
        // 1. 算 UV 和法线
        vec2 uv = varying_uv[0] * bar.x + varying_uv[1] * bar.y + varying_uv[2] * bar.z;
        vec3 n_tex = model.normal(uv);
        vec4 n_view = uniform_M_inv_T * vec4{n_tex.x, n_tex.y, n_tex.z, 0.0};
        vec3 n = normalized(vec3{n_view.x, n_view.y, n_view.z});
        vec3 r = normalized(n * (n * l) * 2.0 - l);

        // 2. 光照参数设置
        double ambient = 0.4;
        double diff = std::max(0.0, n * l);

        // 3. 从贴图读取高光权重 (灰度图，读 [0] 通道即可)
        double spec_weight = IShader::sample2D(model.specular(), uv)[0] / 255.0;
        double spec = (3.0 * spec_weight) * std::pow(std::max(r.z, 0.0), e);

        // 4. 从贴图读取基础颜色
        TGAColor base_color = IShader::sample2D(model.diffuse(), uv);

        // 5. 颜色混合输出
        for (int i = 0; i < 3; i++)
        {
            base_color[i] = std::min<int>(255, base_color[i] * (ambient + diff + spec));
        }

        return {false, base_color};
    }
};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
        return 1;
    }

    Model model(argv[1]);
    TGAImage framebuffer(width, height, TGAImage::RGB);

    // 1. 设置管线全局状态 (替代你原来在 main 里的局部变量)
    vec3 eye{0, 0, 1.8};
    vec3 center{0, 0, 0};
    vec3 up{0, 1, 0};

    set_model_matrix(PI / 6.0);
    set_view_matrix(eye, center, up);
    set_projection_matrix(60.0, (double)width / height, 0.1, 100.0);
    set_viewport_matrix(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    init_zbuffer(width, height);

    // 2. 实例化着色器
    vec3 light_dir{1.0, 1.0, 1.0};
    PhongShader shader(light_dir, model);

    // 3. 渲染主循环 (Primitive Assembly)
    for (int i = 0; i < model.nfaces(); i++)
    {
        vec4 clip_coords[3];
        for (int j = 0; j < 3; j++)
        {
            // 调用 Vertex Shader，获取裁剪坐标
            clip_coords[j] = shader.vertex(i, j);
        }

        // 把组装好的三角形丢进硬件 (光栅化)
        rasterize(clip_coords, shader, framebuffer);
    }

    framebuffer.write_tga_file("framebuffer.tga");

    // 4. 可视化 z-buffer (直接使用外部全局变量 zbuffer)
    TGAImage zbuffer_img(width, height, TGAImage::RGB);
    double min_z = std::numeric_limits<double>::max();
    double max_z = -std::numeric_limits<double>::max();

    for (int i = 0; i < width * height; i++)
    {
        if (zbuffer[i] < 100.0)
        {
            min_z = std::min(min_z, zbuffer[i]);
            max_z = std::max(max_z, zbuffer[i]);
        }
    }

    for (int px = 0; px < width; px++)
    {
        for (int py = 0; py < height; py++)
        {
            double z = zbuffer[px + py * width];
            if (z < 100.0)
            {
                unsigned char c = 255 * (max_z - z) / (max_z - min_z);
                zbuffer_img.set(px, py, TGAColor{c, c, c, 255});
            }
        }
    }
    zbuffer_img.write_tga_file("zbuffer.tga");

    return 0;
}