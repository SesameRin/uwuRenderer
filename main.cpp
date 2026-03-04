#include <iostream>
#include <cstdlib>
#include "model.h"
#include "our_gl.h"
#include <limits>
#include "tgaimage.h"

constexpr double PI = 3.14159265358979323846;
constexpr int width = 800;
constexpr int height = 800;

// 平滑着色的 Phong 光照模型
struct PhongShader : public IShader
{
    Model &model;
    vec3 l;              // View Space 下的光照方向
    vec3 tri[3];         // View Space 下的三角形顶点坐标
    vec3 varying_nrm[3]; // 传递给片元着色器的顶点法线 (View Space)

    mat<4, 4> uniform_M_inv_T; // 预计算法线的变换矩阵：(View * Model) 的逆转置
    double e = 35.0;

    PhongShader(vec3 light_dir, Model &m) : model(m)
    {
        vec4 l_view = ViewMatrix * vec4{light_dir.x, light_dir.y, light_dir.z, 0.0};
        l = normalized(vec3{l_view.x, l_view.y, l_view.z});
        uniform_M_inv_T = (ViewMatrix * ModelMatrix).invert_transpose();
    }

    virtual vec4 vertex(int iface, int nthvert) override
    {
        // 1. 处理顶点坐标
        vec3 v = model.vert(iface, nthvert);
        vec4 gl_Vertex = vec4{v.x, v.y, v.z, 1.0};
        vec4 v_view = ViewMatrix * ModelMatrix * gl_Vertex;
        tri[nthvert] = vec3{v_view.x, v_view.y, v_view.z};

        // 2. 处理顶点法线
        vec3 n = model.normal(iface, nthvert);
        vec4 n_view = uniform_M_inv_T * vec4{n.x, n.y, n.z, 0.0};
        varying_nrm[nthvert] = normalized(vec3{n_view.x, n_view.y, n_view.z});

        // 3. 投影到裁剪空间
        return ProjMatrix * v_view;
    }

    virtual std::pair<bool, TGAColor> fragment(vec3 bar) override
    {
        TGAColor base_color = {255, 255, 255, 255};

        // 利用重心坐标进行法线插值
        vec3 n = normalized(varying_nrm[0] * bar.x +varying_nrm[1] * bar.y +varying_nrm[2] * bar.z);
        // R = 2 * (N·L) * N - L
        vec3 r = normalized(n * (n * l) * 2.0 - l);

        // --- Phong Reflection Model ---
        double ambient = 0.3;               // 环境光系数
        double diff = std::max(0.0, n * l); // 漫反射：法线和光照方向的夹角余弦

        // 指向相机的视线向量 V : (0, 0, 1)。
        // 高光dot(V, R)，即 (0, 0, 1) · (r.x, r.y, r.z) = r.z
        double spec = std::pow(std::max(r.z, 0.0), e);

        for (int i = 0; i < 3; i++)
        {
            // 系数可调整
            base_color[i] = std::min<int>(255, base_color[i] * (ambient + 0.4 * diff + 0.9 * spec));
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