#include <filesystem>
#include "input_data.hpp"
#include "cv_utils.hpp"
#include <tinyply.h>
#include <memory>
#include <fstream>
#include <cstring>

namespace fs = std::filesystem;
using namespace torch::indexing;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot, const std::string& meshInput); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot); }

std::unique_ptr<MeshConstraintRaw> loadMeshConstraint(const std::string& fileName) {

    std::printf("Opening mesh gaussians file %s... ", fileName.c_str());
    // Load the file into a file stream
    std::unique_ptr<std::istream> file_stream;
    file_stream.reset(new std::ifstream(fileName, std::ios::binary));
    if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + fileName);
    file_stream->seekg(0, std::ios::end);
    const float size_mb = file_stream->tellg() * float(1e-6);
    file_stream->seekg(0, std::ios::beg);

    std::printf("OK\n");

    std::printf("Checking required elements and properties... ");

    tinyply::PlyFile meshply;
    meshply.parse_header(*file_stream);

    if(meshply.get_elements().size() != 1 || meshply.get_elements()[0].name != "vertex") {
        printf("Incorrect elements format in mesh input file\n");
        return nullptr;
    }
    // Verification of required fields in input ply file
    {
        const std::array<std::string, 14> requiredFields = {
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
        };

        const auto element = meshply.get_elements()[0];

        for(const auto& f: requiredFields) {
            bool found = false;
            for(const auto& p: element.properties) {
                if(p.name == f) {
                    found = true;
                    break;
                }
            }
            if(!found) {
                printf("Required field %s was not found in mesh input file\n", f.c_str());
                return nullptr;
            }
        }
    }

    printf("OK\n");

    unsigned int nrgauss = meshply.get_elements()[0].size;

    std::unique_ptr<MeshConstraintRaw> mc = std::make_unique<MeshConstraintRaw>();
    mc->means = std::vector<float>(nrgauss*3);
    mc->colors = std::vector<float>(nrgauss*3);
    mc->scales = std::vector<float>(nrgauss*3);
    mc->quats = std::vector<float>(nrgauss*4);

    // read from file here...
    std::shared_ptr<tinyply::PlyData> means;
    std::shared_ptr<tinyply::PlyData> colors;
    std::shared_ptr<tinyply::PlyData> scales;
    std::shared_ptr<tinyply::PlyData> quats;

    means = meshply.request_properties_from_element("vertex", {"x", "y", "z"});
    colors = meshply.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
    scales = meshply.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
    quats = meshply.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

    printf("Loading mesh data... ");

    // read the file
    meshply.read(*file_stream);

    // copy into our struct
    {
        size_t num_b = means->buffer.size_bytes();
        memcpy(mc->means.data(), means->buffer.get(), num_b);
        
        num_b = colors->buffer.size_bytes();
        memcpy(mc->colors.data(), colors->buffer.get(), num_b);

        num_b = scales->buffer.size_bytes();
        memcpy(mc->scales.data(), scales->buffer.get(), num_b);

        num_b = quats->buffer.size_bytes();
        memcpy(mc->quats.data(), quats->buffer.get(), num_b);
    }

    printf("OK\n");

    return mc;
}

InputData inputDataFromX(const std::string &projectRoot, const std::string& meshInput){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot, meshInput);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot);
    }else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio project folder)");
    }
}

torch::Tensor Camera::getIntrinsicsMatrix(){
    return torch::tensor({{fx, 0.0f, cx},
                          {0.0f, fy, cy},
                          {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
}

void Camera::loadImage(float downscaleFactor){
    // Populates image and K, then updates the camera parameters
    // Caution: this function has destructive behaviors
    // and should be called only once
    if (image.numel()) std::runtime_error("loadImage already called");
    std::cout << "Loading " << filePath << std::endl;

    float scaleFactor = 1.0f / downscaleFactor;
    cv::Mat cImg = imreadRGB(filePath);
    
    float rescaleF = 1.0f;
    // If camera intrinsics don't match the image dimensions 
    if (cImg.rows != height || cImg.cols != width){
        rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(height);
    }
    fx *= scaleFactor * rescaleF;
    fy *= scaleFactor * rescaleF;
    cx *= scaleFactor * rescaleF;
    cy *= scaleFactor * rescaleF;

    if (downscaleFactor > 1.0f){
        float f = 1.0f / downscaleFactor;
        cv::resize(cImg, cImg, cv::Size(), f, f, cv::INTER_AREA);
    }

    K = getIntrinsicsMatrix();
    cv::Rect roi;

    if (hasDistortionParameters()){
        // Undistort
        std::vector<float> distCoeffs = undistortionParameters();
        cv::Mat cK = floatNxNtensorToMat(K);
        cv::Mat newK = cv::getOptimalNewCameraMatrix(cK, distCoeffs, cv::Size(cImg.cols, cImg.rows), 0, cv::Size(), &roi);

        cv::Mat undistorted = cv::Mat::zeros(cImg.rows, cImg.cols, cImg.type());
        cv::undistort(cImg, undistorted, cK, distCoeffs, newK);
        
        image = imageToTensor(undistorted);
        K = floatNxNMatToTensor(newK);
    }else{
        roi = cv::Rect(0, 0, cImg.cols, cImg.rows);
        image = imageToTensor(cImg);
    }

    // Crop to ROI
    image = image.index({Slice(roi.y, roi.y + roi.height), Slice(roi.x, roi.x + roi.width), Slice()});

    // Update parameters
    height = image.size(0);
    width = image.size(1);
    fx = K[0][0].item<float>();
    fy = K[1][1].item<float>();
    cx = K[0][2].item<float>();
    cy = K[1][2].item<float>();
}

torch::Tensor Camera::getImage(int downscaleFactor){
    if (downscaleFactor <= 1) return image;
    else{

        // torch::jit::script::Module container = torch::jit::load("gt.pt");
        // return container.attr("val").toTensor();

        if (imagePyramids.find(downscaleFactor) != imagePyramids.end()){
            return imagePyramids[downscaleFactor];
        }

        // Rescale, store and return
        cv::Mat cImg = tensorToImage(image);
        cv::resize(cImg, cImg, cv::Size(cImg.cols / downscaleFactor, cImg.rows / downscaleFactor), 0.0, 0.0, cv::INTER_AREA);
        torch::Tensor t = imageToTensor(cImg);
        imagePyramids[downscaleFactor] = t;
        return t;
    }
}

bool Camera::hasDistortionParameters(){
    return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
}

std::vector<float> Camera::undistortionParameters(){
    std::vector<float> p = { k1, k2, p1, p2, k3, 0.0f, 0.0f, 0.0f };
    return p;
}

std::tuple<std::vector<Camera>, Camera *> InputData::getCameras(bool validate, const std::string &valImage){
    if (!validate) return std::make_tuple(cameras, nullptr);
    else{
        size_t valIdx = -1;
        std::srand(42);

        if (valImage == "random"){
            valIdx = std::rand() % cameras.size();
        }else{
            for (size_t i = 0; i < cameras.size(); i++){
                if (fs::path(cameras[i].filePath).filename().string() == valImage){
                    valIdx = i;
                    break;
                }
            }
            if (valIdx == -1) throw std::runtime_error(valImage + " not in the list of cameras");
        }

        std::vector<Camera> cams;
        Camera *valCam = nullptr;

        for (size_t i = 0; i < cameras.size(); i++){
            if (i != valIdx) cams.push_back(cameras[i]);
            else valCam = &cameras[i];
        }

        return std::make_tuple(cams, valCam);
    }
}
