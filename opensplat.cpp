#include <filesystem>
#include "vendor/json/json.hpp"
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "vendor/cxxopts.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator");
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("val-every", "Dump evaluation images every this amount of iterations", cxxopts::value<int>()->default_value("50"))
        ("cpu", "Force CPU execution")
        
        ("mesh-file", "Filename of a .ply file specifying the gaussians defining the structure of input", cxxopts::value<std::string>()->default_value(""))
        ("fixed", "No spliting/duplicating/pruning of gaussians")

        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("2"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Split/duplicate/prune gaussians every these many steps", cxxopts::value<int>()->default_value("100"))
        ("warmup-length", "Split/duplicate/prune gaussians only after these many steps", cxxopts::value<int>()->default_value("500"))
        ("reset-alpha-every", "Reset the opacity values of gaussians after these many refinements (not steps)", cxxopts::value<int>()->default_value("30"))
        ("stop-split-at", "Stop splitting/duplicating gaussians after these many steps", cxxopts::value<int>()->default_value("15000"))
        ("densify-grad-thresh", "Threshold of the positional gradient norm (magnitude of the loss function) which when exceeded leads to a gaussian split/duplication", cxxopts::value<float>()->default_value("0.0002"))
        ("densify-size-thresh", "Gaussians' scales below this threshold are duplicated, otherwise split", cxxopts::value<float>()->default_value("0.01"))
        ("stop-screen-size-at", "Stop splitting gaussians that are larger than [split-screen-size] after these many steps", cxxopts::value<int>()->default_value("4000"))
        ("split-screen-size", "Split gaussians that are larger than this percentage of screen space", cxxopts::value<float>()->default_value("0.05"))

        ("h,help", "Print usage")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[colmap or nerfstudio project path]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const bool fixedPoints = result.count("fixed") > 0;
    const std::string projectRoot = result["input"].as<std::string>();
    const std::string outputScene = result["output"].as<std::string>();
    const int saveEvery = result["save-every"].as<int>(); 
    const bool validate = result.count("val") > 0 || result.count("val-render") > 0;
    const std::string valImage = result["val-image"].as<std::string>();
    const std::string valRender = result["val-render"].as<std::string>();
    const int valEvery = result["val-every"].as<int>();
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);

    const float downScaleFactor = (std::max)(result["downscale-factor"].as<float>(), 1.0f);
    const int numIters = result["num-iters"].as<int>();
    const int numDownscales = result["num-downscales"].as<int>();
    const int resolutionSchedule = result["resolution-schedule"].as<int>();
    const int shDegree = result["sh-degree"].as<int>();
    const int shDegreeInterval = result["sh-degree-interval"].as<int>();
    const float ssimWeight = result["ssim-weight"].as<float>();
    const int refineEvery = (fixedPoints)? 2 * numIters : result["refine-every"].as<int>();
    const int warmupLength = result["warmup-length"].as<int>();
    const int resetAlphaEvery = result["reset-alpha-every"].as<int>();
    const int stopSplitAt = (fixedPoints) ? 1 : result["stop-split-at"].as<int>();
    const float densifyGradThresh = result["densify-grad-thresh"].as<float>();
    const float densifySizeThresh = result["densify-size-thresh"].as<float>();
    const int stopScreenSizeAt = result["stop-screen-size-at"].as<int>();
    const float splitScreenSize = result["split-screen-size"].as<float>();
    
    const std::string meshInput = result["mesh-file"].as<std::string>();
    const bool hasMeshInput = meshInput.size() > 0;

    torch::Device device = torch::kCPU;
    int displayStep = 1;

    if (torch::cuda::is_available() && result.count("cpu") == 0) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
        displayStep = 10;
    }else{
        std::cout << "Using CPU" << std::endl;
    }

    try{
        InputData inputData = inputDataFromX(projectRoot, meshInput);
        for(int i=0; i<inputData.cameras.size(); i++) inputData.cameras[i].idx = i;
        for (Camera &cam : inputData.cameras){
            // ! on nerfstudio/colmap, I guess we'd have to put "true" here ?
            cam.loadImage(downScaleFactor, true);
        }

        

        // Withhold a validation camera if necessary
        auto t = inputData.getCameras(validate, valImage);
        std::vector<Camera> cams = std::get<0>(t);
        Camera *valCam = std::get<1>(t);

        Model model(inputData,
                    cams.size(),
                    numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
                    refineEvery, warmupLength, resetAlphaEvery, stopSplitAt, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                    numIters, inputData.backgroundColor,
                    device);

        std::vector< size_t > camIndices( cams.size() );
        std::iota( camIndices.begin(), camIndices.end(), 0 );
        InfiniteRandomIterator<size_t> camsIter( camIndices );

        int imageSize = -1;

        std::vector<std::vector<float>> lossesByCamera(cams.size());

        for (size_t step = 1; step <= numIters; step++){

            Camera& cam = cams[ camsIter.next() ];

            if (!valRender.empty() && step % valEvery == 0){
                int i=0;
                for(auto& cam: cams) {
                    
                    torch::Tensor rgb = model.forward(cam, step);
                    cv::Mat image = tensorToImage(rgb.detach().cpu());
                    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                    torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
                    cv::Mat image_gt = tensorToImage(gt.detach().cpu());
                    cv::cvtColor(image_gt, image_gt, cv::COLOR_RGB2BGR);
                    cv::imwrite((fs::path(valRender) / (std::to_string(step) + "_" + std::to_string(i) + ".png")).string(), image);
                    cv::imwrite((fs::path(valRender) / (std::to_string(step) + "_gt_" + std::to_string(i) + ".png")).string(), image_gt);
                    i++;
                }
            }

            model.optimizersZeroGrad();

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
            gt = gt.to(device);

            if (saveEvery > 0 && step % saveEvery == 0){
                fs::path p(outputScene);
                model.savePlySplat((p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string()));
            }

            torch::Tensor mainLoss = model.mainLoss(rgb, gt, ssimWeight);
            mainLoss.backward();
            float steploss = mainLoss.item<float>();
            
            lossesByCamera[cam.idx].push_back(steploss);
            
            if (step % displayStep == 0) std::cout << "Step " << step << ": " << steploss << std::endl;

            model.optimizersStep();
            model.schedulersStep(step);
            model.afterTrain(step);
        }

        model.savePlySplat(outputScene);
        // model.saveDebugPly("debug.ply");

        // Write losses to output file
        std::ofstream losses_write;
        std::string losses_path = (fs::path(outputScene).parent_path() / "losses.txt").string();
        losses_write.open(losses_path);
        losses_write << cams.size() << std::endl;
        for(int i=0; i<cams.size(); i++) {
            for(const auto& x: lossesByCamera[i]) losses_write << x << " ";
            losses_write << std::endl;
        }
        // and now, average loss by iteration
        int iteration = 0;
        while(iteration < numIters) {
            bool shouldcontinue = false;
            float avgloss = 0.f;
            int nrcams = 0;
            for(int i=0; i<cams.size(); i++) {
                if(iteration < lossesByCamera[i].size()) {
                    shouldcontinue = true;
                    nrcams++;
                    avgloss += lossesByCamera[i][iteration];
                }
            }
            if(!shouldcontinue) break;
            else {
                losses_write << (avgloss / nrcams) << " ";
            }
            iteration++;
        }
        std::cout << "Wrote losses to " << losses_path << std::endl;
        losses_write.close();

        // Validate
        if (valCam != nullptr){
            torch::Tensor rgb = model.forward(*valCam, numIters);
            torch::Tensor gt = valCam->getImage(model.getDownscaleFactor(numIters)).to(device);
            std::cout << valCam->filePath << " validation loss: " << model.mainLoss(rgb, gt, ssimWeight).item<float>() << std::endl; 
        }
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}
