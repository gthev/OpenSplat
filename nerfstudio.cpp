#include <filesystem>
#include <cstdlib>
#include "vendor/json/json.hpp"
#include "nerfstudio.hpp"
#include "point_io.hpp"
#include "cv_utils.hpp"
#include "tensor_math.hpp"

namespace fs = std::filesystem;

using json = nlohmann::json;
using namespace torch::indexing;

namespace ns
{

    void to_json(json &j, const Frame &f)
    {
        j = json{
            {"file_path", f.filePath},
            {"w", f.width},
            {"h", f.height},
            {"fl_x", f.fx},
            {"fl_y", f.fy},
            {"cx", f.cx},
            {"cy", f.cy},
            {"k1", f.k1},
            {"k2", f.k2},
            {"p1", f.p1},
            {"p2", f.p2},
            {"k3", f.k3},
            {"transform_matrix", f.transformMatrix},

        };
    }

    void from_json(const json &j, Frame &f)
    {
        j.at("file_path").get_to(f.filePath);
        j.at("transform_matrix").get_to(f.transformMatrix);
        if (j.contains("w"))
            j.at("w").get_to(f.width);
        if (j.contains("h"))
            j.at("h").get_to(f.height);
        if (j.contains("fl_x"))
            j.at("fl_x").get_to(f.fx);
        if (j.contains("fl_y"))
            j.at("fl_y").get_to(f.fy);
        if (j.contains("cx"))
            j.at("cx").get_to(f.cx);
        if (j.contains("cy"))
            j.at("cy").get_to(f.cy);
        if (j.contains("k1"))
            j.at("k1").get_to(f.k1);
        if (j.contains("k2"))
            j.at("k2").get_to(f.k2);
        if (j.contains("p1"))
            j.at("p1").get_to(f.p1);
        if (j.contains("p2"))
            j.at("p2").get_to(f.p2);
        if (j.contains("k3"))
            j.at("k3").get_to(f.k3);
    }

    void to_json(json &j, const Transforms &t)
    {
        j = json{
            {"camera_model", t.cameraModel},
            {"frames", t.frames},
            {"ply_file_path", t.plyFilePath},
            {"background_color", t.backgroundColor},
        };
    }

    void from_json(const json &j, Transforms &t)
    {
        j.at("camera_model").get_to(t.cameraModel);
        j.at("frames").get_to(t.frames);
        if (j.contains("ply_file_path"))
            j.at("ply_file_path").get_to(t.plyFilePath);

        if (j.contains("background_color"))
        {
            j.at("background_color").get_to(t.backgroundColor);
        }
        else
        {
            t.backgroundColor = {0.6130f, 0.0101f, 0.3984f};
        }

        // Globals
        int width = 0;
        int height = 0;
        float fx = 0;
        float fy = 0;
        float cx = 0;
        float cy = 0;
        float k1 = 0;
        float k2 = 0;
        float k3 = 0;
        float p1 = 0;
        float p2 = 0;

        if (j.contains("w"))
            j.at("w").get_to(width);
        if (j.contains("h"))
            j.at("h").get_to(height);
        if (j.contains("fl_x"))
            j.at("fl_x").get_to(fx);
        if (j.contains("fl_y"))
            j.at("fl_y").get_to(fy);
        if (j.contains("cx"))
            j.at("cx").get_to(cx);
        if (j.contains("cy"))
            j.at("cy").get_to(cy);
        if (j.contains("k1"))
            j.at("k1").get_to(k1);
        if (j.contains("k2"))
            j.at("k2").get_to(k2);
        if (j.contains("p1"))
            j.at("p1").get_to(p1);
        if (j.contains("p2"))
            j.at("p2").get_to(p2);
        if (j.contains("k3"))
            j.at("k3").get_to(k3);

        // Assign per-frame intrinsics if missing
        for (Frame &f : t.frames)
        {
            if (!f.width && width)
                f.width = width;
            if (!f.height && height)
                f.height = height;
            if (!f.fx && fx)
                f.fx = fx;
            if (!f.fy && fy)
                f.fy = fy;
            if (!f.cx && cx)
                f.cx = cx;
            if (!f.cy && cy)
                f.cy = cy;
            if (!f.k1 && k1)
                f.k1 = k1;
            if (!f.k2 && k2)
                f.k2 = k2;
            if (!f.p1 && p1)
                f.p1 = p1;
            if (!f.p2 && p2)
                f.p2 = p2;
            if (!f.k3 && k3)
                f.k3 = k3;
        }

        std::sort(t.frames.begin(), t.frames.end(),
                  [](Frame const &a, Frame const &b)
                  {
                      return a.filePath < b.filePath;
                  });
    }

    Transforms readTransforms(const std::string &filename)
    {
        std::ifstream f(filename);
        json data = json::parse(f);
        return data.template get<Transforms>();
    }

    torch::Tensor posesFromTransforms(const Transforms &t)
    {
        torch::Tensor poses = torch::zeros({static_cast<long int>(t.frames.size()), 4, 4}, torch::kFloat32);
        for (size_t c = 0; c < t.frames.size(); c++)
        {
            for (size_t i = 0; i < 4; i++)
            {
                for (size_t j = 0; j < 4; j++)
                {
                    poses[c][i][j] = t.frames[c].transformMatrix[i][j];
                }
            }
        }
        return poses;
    }

    struct PointSetConstrRes
    {
        PointSet *ps;
        std::shared_ptr<MeshConstraint> mc;
    };

    PointSetConstrRes pointSetFromMeshConstraint(std::shared_ptr<MeshConstraintRaw> mcr)
    {
        PointSet *r = new PointSet();
        std::shared_ptr<MeshConstraint> mc = std::make_shared<MeshConstraint>();
        mc->_impl = mcr;
        unsigned int nrgauss = mcr->means.size() / 3;
        r->points.resize(nrgauss);
        r->colors.resize(nrgauss);

        for (int i = 0; i < nrgauss; i++)
        {
            // conversion from SH to u8 rgb: a bit useless, as it will be converted
            // back to SH during model initialization
            float rr = mcr->colors[3 * i], g = mcr->colors[3 * i + 1], b = mcr->colors[3 * i + 2];
            const float C0 = 0.2820947f;
            r->colors[i] = {static_cast<unsigned char>(((rr * C0) + 0.5) * 254),
                            static_cast<unsigned char>(((g * C0) + 0.5) * 254),
                            static_cast<unsigned char>(((b * C0) + 0.5) * 254)};
        }

        memcpy(r->points.data(), mcr->means.data(), 3 * nrgauss * sizeof(float));
        mc->scales = torch::from_blob(mcr->scales.data(), {static_cast<long int>(nrgauss), 3}, torch::kFloat32);
        mc->quats = torch::from_blob(mcr->quats.data(), {static_cast<long int>(nrgauss), 4}, torch::kFloat32);

        return {r, std::move(mc)};
    }

    InputData inputDataFromNerfStudio(const std::string &projectRoot, const std::string &meshInput)
    {
        InputData ret;
        fs::path nsRoot(projectRoot);
        fs::path transformsPath = nsRoot / "transforms.json";
        if (!fs::exists(transformsPath))
            throw std::runtime_error(transformsPath.string() + " does not exist");

        bool hasMeshInput = meshInput.size() > 0;

        Transforms t = readTransforms(transformsPath.string());
        if (t.plyFilePath.empty() && !hasMeshInput)
            throw std::runtime_error("ply_file_path is empty (and no mesh input)");

        ret.backgroundColor = t.backgroundColor;

        PointSet *pSet = nullptr;
        std::shared_ptr<MeshConstraint> meshctr = nullptr;
        if (!hasMeshInput)
            pSet = readPointSet((nsRoot / t.plyFilePath).string());
        else
        {
            std::shared_ptr<MeshConstraintRaw> mc = loadMeshConstraint(meshInput);
            auto res = pointSetFromMeshConstraint(mc);
            pSet = res.ps;
            meshctr = res.mc;
        }

        torch::Tensor unorientedPoses = posesFromTransforms(t);

        auto r = autoScaleAndCenterPoses(unorientedPoses);
        torch::Tensor poses = std::get<0>(r);
        ret.translation = std::get<1>(r);
        ret.scale = std::get<2>(r);

        // aabbScale = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]

        for (size_t i = 0; i < t.frames.size(); i++)
        {
            Frame f = t.frames[i];

            ret.cameras.emplace_back(Camera(f.width, f.height,
                                            static_cast<float>(f.fx), static_cast<float>(f.fy),
                                            static_cast<float>(f.cx), static_cast<float>(f.cy),
                                            static_cast<float>(f.k1), static_cast<float>(f.k2), static_cast<float>(f.k3),
                                            static_cast<float>(f.p1), static_cast<float>(f.p2),

                                            poses[i], (nsRoot / f.filePath).string()));
        }

        torch::Tensor points = pSet->pointsTensor().clone();
        if (hasMeshInput)
            ret.points.mesh = std::move(meshctr);

        ret.points.xyz = (points - ret.translation) * ret.scale;
        ret.points.rgb = pSet->colorsTensor().clone();

        RELEASE_POINTSET(pSet);

        return ret;
    }

}
