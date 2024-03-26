#ifndef NERFSTUDIO_H
#define NERFSTUDIO_H

#include <iostream>
#include <string>
#include <fstream>
#include <optional>
#include <torch/torch.h>
#include "input_data.hpp"
#include "vendor/json/json_fwd.hpp"

using json = nlohmann::json;

namespace ns{
    typedef std::vector<std::vector<float>> Mat4;

    struct Frame{
        std::string filePath = "";
        int width = 0;
        int height = 0;
        double fx = 0;
        double fy = 0;
        double cx = 0;
        double cy = 0;
        double k1 = 0;
        double k2 = 0;
        double p1 = 0;
        double p2 = 0;
        double k3 = 0;
        Mat4 transformMatrix;
    };
    void to_json(json &j, const Frame &f);
    void from_json(const json& j, Frame &f);

    struct Transforms{
        std::string cameraModel;
        std::vector<Frame> frames;
        std::string plyFilePath;
        std::array<float, 3> backgroundColor;
    };
    void to_json(json &j, const Transforms &t);
    void from_json(const json& j, Transforms &t);

    Transforms readTransforms(const std::string &filename);
    torch::Tensor posesFromTransforms(const Transforms &t);

    InputData inputDataFromNerfStudio(const std::string &projectRoot, const std::string& meshInput);
}   



#endif