#pragma once
#include <vector>
#include <string>
#include "geometry.h"
#include "tgaimage.h"

class Model
{
    std::vector<vec3> verts = {};
    std::vector<vec3> norms = {};
    std::vector<vec2> tex = {};
    std::vector<int> facet_vrt = {};
    std::vector<int> facet_nrm = {};
    std::vector<int> facet_tex = {};
    TGAImage diffusemap = {};  // diffuse color texture
    TGAImage normalmap = {};   // normal map texture
    TGAImage specularmap = {}; // specular texture

public:
    Model(const std::string filename);
    int nverts() const;
    int nfaces() const;
    vec3 vert(const int i) const;
    vec3 vert(const int iface, const int nthvert) const;
    vec3 normal(const int iface, const int nthvert) const;

    vec3 normal(const vec2 &uv) const;
    vec2 uv(const int iface, const int nthvert) const;

    const TGAImage &diffuse() const;
    const TGAImage &specular() const;
};