#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

namespace  relevance_map {

/**
 * @brief VCCS hyperparameters and camera intrisic parameters
 * This parameters are used as default by the image_processing library
 */
struct sv_param {
    //Supervoxels hyperparameters
    static constexpr bool use_transform = false;
    static constexpr double voxel_resolution = 0.008f;
    static constexpr double color_importance = 0.2f;
    static constexpr double spatial_importance = 0.4f;
    static constexpr double normal_importance = 0.4f;
    static constexpr double seed_resolution = 0.05f;
    //Camera intrisic parameters
    static constexpr float depth_princ_pt_x = 479.75;
    static constexpr float depth_princ_pt_y = 269.75;
    static constexpr float rgb_princ_pt_x = 479.75;
    static constexpr float rgb_princ_pt_y = 269.75;
    static constexpr float focal_length_x = 540.68603515625;
    static constexpr float focal_length_y = 540.68603515625;
    static constexpr float height = 540;
    static constexpr float width = 960;
};

}

#endif //PARAMETERS_HPP
