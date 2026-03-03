#include <cmath>
#include <tuple>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include "geometry.h"
#include "model.h"
#include "tgaimage.h"

constexpr double PI = 3.14159265358979323846;

constexpr int width = 800;
constexpr int height = 800;

// Model 矩阵 (局部坐标 -> 世界坐标)
mat<4, 4> get_model_matrix(double angle_y)
{
    // 一个绕y轴的旋转
    mat<4, 4> m;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m[i][j] = (i == j ? 1. : 0.);
    m[0][0] = std::cos(angle_y);
    m[0][2] = std::sin(angle_y);
    m[2][0] = -std::sin(angle_y);
    m[2][2] = std::cos(angle_y);
    return m; 
}

// View 矩阵 (世界坐标 -> 视图/相机坐标)
mat<4, 4> get_view_matrix(vec3 eye, vec3 center, vec3 up)
{
    // M_cam = T_cam * R_cam (相机变换矩阵，先旋转再平移)
    // View = inv(M_cam) = inv(R_cam) * inv(T_cam)
    // 旋转矩阵是正交矩阵，其逆矩阵就是它的转置

    // f,u,s 是相机坐标系的basis
    vec3 f = normalized(center - eye); // 前向向量 (指向目标)
    vec3 u = normalized(up);
    vec3 s = normalized(cross(f, u)); // 右向量
    u = cross(s, f);                  // 重新正交化的上向量

    mat<4, 4> Minv; 
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            Minv[i][j] = (i == j ? 1. : 0.);
    Minv[0][0] = s.x;
    Minv[0][1] = s.y;
    Minv[0][2] = s.z;
    Minv[1][0] = u.x;
    Minv[1][1] = u.y;
    Minv[1][2] = u.z;
    Minv[2][0] = -f.x;
    Minv[2][1] = -f.y;
    Minv[2][2] = -f.z;

    mat<4, 4> Tr;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            Tr[i][j] = (i == j ? 1. : 0.);
    Tr[0][3] = -eye.x;
    Tr[1][3] = -eye.y;
    Tr[2][3] = -eye.z;

    return Minv * Tr;
}

// Projection 矩阵 (视图坐标 -> 裁剪坐标)
mat<4, 4> get_projection_matrix(double fov_deg, double aspect, double near, double far) // aspect: 宽高比
{
    // P matrix:
    // n/r 0   0   0
    // 0   n/t 0   0
    // 0   0   A   B
    // 0   0   -1  0
    mat<4, 4> P;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            P[i][j] = 0.;
    double f = 1.0 / std::tan(fov_deg * PI / 180.0 / 2.0);

    P[0][0] = f / aspect;
    P[1][1] = f;
    P[2][2] = -(far + near) / (far - near);       // A
    P[2][3] = -(2.0 * far * near) / (far - near); // B
    P[3][2] = -1.0;                              
    return P;
}

// Viewport 矩阵 (NDC -> 屏幕像素坐标)
mat<4, 4> get_viewport_matrix(int x, int y, int w, int h)
{
    // [-1,1] * [-1,1] -> [0,w] * [0,h]

    // w/2    0     0    x + w/2
    //  0    h/2    0    y + h/2 
    //  0     0     1        1    
    //  0     0     0        1    
    mat<4, 4> V;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            V[i][j] = (i == j ? 1. : 0.);
    V[0][3] = x + w / 2.0;
    V[1][3] = y + h / 2.0;
    V[2][3] = 1.0;
    V[0][0] = w / 2.0;
    V[1][1] = h / 2.0;
    V[2][2] = 1.0;
    return V;
}

double signed_triangle_area(double ax, double ay, double bx, double by, double cx, double cy)
{
    return 0.5 * ((by - ay) * (bx + ax) + (cy - by) * (cx + bx) + (ay - cy) * (ax + cx));
}


void rasterize(vec3 v0, vec3 v1, vec3 v2, std::vector<double> &zbuffer, TGAImage &framebuffer, TGAColor color)
{
    int bbminx = std::max(0, static_cast<int>(std::min({v0.x, v1.x, v2.x})));
    int bbminy = std::max(0, static_cast<int>(std::min({v0.y, v1.y, v2.y})));
    int bbmaxx = std::min(width - 1, static_cast<int>(std::max({v0.x, v1.x, v2.x})));
    int bbmaxy = std::min(height - 1, static_cast<int>(std::max({v0.y, v1.y, v2.y})));

    double total_area = signed_triangle_area(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
    if (total_area < 1e-5)
        return; // 背面剔除 (Backface Culling)

    for (int x = bbminx; x <= bbmaxx; x++)
    {
        for (int y = bbminy; y <= bbmaxy; y++)
        {
            double alpha = signed_triangle_area(x, y, v1.x, v1.y, v2.x, v2.y) / total_area;
            double beta = signed_triangle_area(x, y, v2.x, v2.y, v0.x, v0.y) / total_area;
            double gamma = signed_triangle_area(x, y, v0.x, v0.y, v1.x, v1.y) / total_area;

            if (alpha < 0 || beta < 0 || gamma < 0)
                continue;

            // 插值 Z_ndc
            double z = alpha * v0.z + beta * v1.z + gamma * v2.z;

            int idx = x + y * width;

            // 深度测试。对于标准的 OpenGL 投影矩阵，算出的 Z_ndc 越小 (靠近 -1) 代表越近。
            if (z < zbuffer[idx])
            {
                zbuffer[idx] = z;
                framebuffer.set(x, y, color);
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
        return 1;
    }

    Model model(argv[1]);
    TGAImage framebuffer(width, height, TGAImage::RGB);

    std::vector<double> zbuffer(width * height, std::numeric_limits<double>::max());

    // 设定相机参数
    vec3 eye{0, 0, 3};    // 相机位置
    vec3 center{0, 0, 0}; // 看向原点
    vec3 up{0, 1, 0};     // 向上向量

    // Viewport * P * V * M * vertex
    mat<4, 4> ModelMatrix = get_model_matrix(PI / 6.0);
    mat<4, 4> ViewMatrix = get_view_matrix(eye, center, up);
    mat<4, 4> ProjMatrix = get_projection_matrix(60.0, (double)width / height, 0.1, 100.0);
    mat<4, 4> VpMatrix = get_viewport_matrix(width / 8, height / 8, width * 3 / 4, height * 3 / 4);

    for (int i = 0; i < model.nfaces(); i++)
    {
        vec3 screen_coords[3];

        for (int j = 0; j < 3; j++)
        {
            vec3 v = model.vert(i, j);
            vec4 gl_Vertex = vec4{v.x, v.y, v.z, 1.0};

            // clipped = P * V * M * vertex
            vec4 clip = ProjMatrix * ViewMatrix * ModelMatrix * gl_Vertex;

            // 透视除法 (Perspective Divide) -> 得到 NDC 坐标 [-1, 1]
            vec3 ndc = vec3{clip.x / clip.w, clip.y / clip.w, clip.z / clip.w};

            // 视口变换 -> 映射到屏幕真实像素 [w,h]
            vec4 screen_homo = VpMatrix * vec4{ndc.x, ndc.y, ndc.z, 1.0};
            screen_coords[j] = vec3{screen_homo.x, screen_homo.y, screen_homo.z};
        }

        TGAColor rnd;
        for (int c = 0; c < 3; c++)
            rnd[c] = std::rand() % 255;

        rasterize(screen_coords[0], screen_coords[1], screen_coords[2], zbuffer, framebuffer, rnd);
    }

    framebuffer.write_tga_file("framebuffer.tga");

    /// 可视化 z-buffer
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