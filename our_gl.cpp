#include "our_gl.h"
#include <cmath>
#include <algorithm>
#include <limits>

constexpr double PI = 3.14159265358979323846;

mat<4, 4> ModelMatrix;
mat<4, 4> ViewMatrix;
mat<4, 4> ProjMatrix;
mat<4, 4> VpMatrix;
std::vector<double> zbuffer;
int screen_width = 0;
int screen_height = 0;

// Model 矩阵 (局部坐标 -> 世界坐标)
void set_model_matrix(double angle_y)
{
    // 一个绕y轴的旋转
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ModelMatrix[i][j] = (i == j ? 1. : 0.);
    ModelMatrix[0][0] = std::cos(angle_y);
    ModelMatrix[0][2] = std::sin(angle_y);
    ModelMatrix[2][0] = -std::sin(angle_y);
    ModelMatrix[2][2] = std::cos(angle_y);
}

// View 矩阵 (世界坐标 -> 视图/相机坐标)
void set_view_matrix(vec3 eye, vec3 center, vec3 up)
{
    // M_cam = T_cam * R_cam (相机变换矩阵，先旋转再平移)
    // View = inv(M_cam) = inv(R_cam) * inv(T_cam)
    // 旋转矩阵是正交矩阵，其逆矩阵就是它的转置

    // f,u,s 是相机坐标系的basis
    vec3 f = normalized(center - eye); // 前向向量，指向目标
    vec3 u = normalized(up); // 上
    vec3 s = normalized(cross(f, u)); // 叉乘得到另一个坐标轴
    u = cross(s, f);

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

    ViewMatrix = Minv * Tr;
}

// Projection 矩阵 (视图坐标 -> 裁剪坐标)
void set_projection_matrix(double fov_deg, double aspect, double near, double far)
{
    // P matrix:
    // n/r 0   0   0
    // 0   n/t 0   0
    // 0   0   A   B
    // 0   0   -1  0
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ProjMatrix[i][j] = 0.;
    double f = 1.0 / std::tan(fov_deg * PI / 180.0 / 2.0);
    ProjMatrix[0][0] = f / aspect;
    ProjMatrix[1][1] = f;
    ProjMatrix[2][2] = -(far + near) / (far - near); // A
    ProjMatrix[2][3] = -(2.0 * far * near) / (far - near); // B
    ProjMatrix[3][2] = -1.0;
}

// Viewport 矩阵 (NDC -> 屏幕像素坐标)
void set_viewport_matrix(int x, int y, int w, int h)
{
    // [-1,1] * [-1,1] -> [0,w] * [0,h]

    // w/2    0     0    x + w/2
    //  0    h/2    0    y + h/2
    //  0     0     1        1
    //  0     0     0        1
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            VpMatrix[i][j] = (i == j ? 1. : 0.);
    VpMatrix[0][3] = x + w / 2.0;
    VpMatrix[1][3] = y + h / 2.0;
    VpMatrix[2][3] = 1.0;
    VpMatrix[0][0] = w / 2.0;
    VpMatrix[1][1] = h / 2.0;
    VpMatrix[2][2] = 1.0;
}

void init_zbuffer(int w, int h)
{
    screen_width = w;
    screen_height = h;
    zbuffer.assign(w * h, std::numeric_limits<double>::max());
}

double signed_triangle_area(double ax, double ay, double bx, double by, double cx, double cy)
{
    return 0.5 * ((by - ay) * (bx + ax) + (cy - by) * (cx + bx) + (ay - cy) * (ax + cx));
}

// 核心光栅化器：现在它负责透视除法和调用 Shader
void rasterize(vec4 clip_coords[3], IShader &shader, TGAImage &framebuffer)
{
    vec3 ndc[3];
    vec3 screen[3];

    // 1. 透视除法与视口变换 (从 Clip Space 到 Screen Space)
    for (int i = 0; i < 3; i++)
    {
        ndc[i] = vec3{clip_coords[i].x / clip_coords[i].w,
                      clip_coords[i].y / clip_coords[i].w,
                      clip_coords[i].z / clip_coords[i].w};
        vec4 screen_homo = VpMatrix * vec4{ndc[i].x, ndc[i].y, ndc[i].z, 1.0};
        screen[i] = vec3{screen_homo.x, screen_homo.y, screen_homo.z};
    }

    vec3 v0 = screen[0], v1 = screen[1], v2 = screen[2];

    int bbminx = std::max(0, static_cast<int>(std::min({v0.x, v1.x, v2.x})));
    int bbminy = std::max(0, static_cast<int>(std::min({v0.y, v1.y, v2.y})));
    int bbmaxx = std::min(screen_width - 1, static_cast<int>(std::max({v0.x, v1.x, v2.x})));
    int bbmaxy = std::min(screen_height - 1, static_cast<int>(std::max({v0.y, v1.y, v2.y})));

    double total_area = signed_triangle_area(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
    if (total_area < 1e-5)
        return; // 背面剔除

    // 2. 遍历包围盒生成片元
    for (int x = bbminx; x <= bbmaxx; x++)
    {
        for (int y = bbminy; y <= bbmaxy; y++)
        {
            double alpha = signed_triangle_area(x, y, v1.x, v1.y, v2.x, v2.y) / total_area;
            double beta = signed_triangle_area(x, y, v2.x, v2.y, v0.x, v0.y) / total_area;
            double gamma = signed_triangle_area(x, y, v0.x, v0.y, v1.x, v1.y) / total_area;

            if (alpha < 0 || beta < 0 || gamma < 0)
                continue;

            // 这里使用 NDC 坐标的 Z 进行插值测试
            double z = alpha * ndc[0].z + beta * ndc[1].z + gamma * ndc[2].z;
            int idx = x + y * screen_width;

            if (z < zbuffer[idx])
            {
                vec3 barycentric_coords = {alpha, beta, gamma};
                // 3. 调用片元着色器计算颜色
                auto [discard, color] = shader.fragment(barycentric_coords);

                if (!discard)
                {
                    zbuffer[idx] = z;
                    framebuffer.set(x, y, color);
                }
            }
        }
    }
}