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

    // 通用 2D 贴图采样器
    static TGAColor sample2D(const TGAImage &img, const vec2 &uvf)
    {
        return img.get(uvf.x * img.width(), uvf.y * img.height());
    }

    virtual vec4 vertex(int iface, int nthvert) = 0;
    virtual std::pair<bool, TGAColor> fragment(vec3 bar) = 0;
};

void rasterize(vec4 clip_coords[3], IShader &shader, TGAImage &framebuffer);