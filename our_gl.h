#pragma once
#include <vector>
#include <utility>
#include "geometry.h"
#include "tgaimage.h"

extern mat<4, 4> ModelMatrix;
extern mat<4, 4> ViewMatrix;
extern mat<4, 4> ProjMatrix;
extern mat<4, 4> VpMatrix;

extern std::vector<double> zbuffer;
extern int screen_width;
extern int screen_height;

void set_model_matrix(double angle_y);
void set_view_matrix(vec3 eye, vec3 center, vec3 up);
void set_projection_matrix(double fov_deg, double aspect, double near, double far);
void set_viewport_matrix(int x, int y, int w, int h);
void init_zbuffer(int w, int h);

// 可编程管线接口
struct IShader
{
    virtual ~IShader() = default;

    // 顶点着色器：输入面索引和顶点索引，返回裁剪空间坐标 (Clip Coordinates)
    virtual vec4 vertex(int iface, int nthvert) = 0;
    
    // 片元着色器：输入重心坐标，返回 {是否丢弃该片元, 片元颜色}
    virtual std::pair<bool, TGAColor> fragment(vec3 bar) = 0;
};

// 光栅化器核心 (接收组装好的三角形裁剪坐标)
void rasterize(vec4 clip_coords[3], IShader &shader, TGAImage &framebuffer);