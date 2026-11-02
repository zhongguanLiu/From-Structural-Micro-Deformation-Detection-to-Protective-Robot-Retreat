#ifndef BASELINE_COMMON_CONFIG_HPP
#define BASELINE_COMMON_CONFIG_HPP

#include <string>

namespace baseline {

struct InputConfig {
  std::string cloud_topic = "/cloud_registered";
};

struct ReferenceConfig {
  int init_frames = 30;
  double voxel_size = 0.05;
};

struct TemporalConfig {
  int window_frames = 5;
  double voxel_size = 0.05;
};

struct ClusterConfig {
  double tolerance = 0.10;
  int min_cluster_size = 3;
  int max_cluster_size = 100000;
};

struct TemporalFusionConfig {
  int window_frames = 5;
  double voxel_size = 0.05;
};

struct C2CConfig {
  double distance_threshold = 0.05;
  ClusterConfig cluster;
};

struct M3C2Config {
  double core_voxel_size = 0.05;
  double normal_radius = 0.12;
  double cylinder_radius = 0.10;
  double cylinder_half_length = 0.20;
  int min_support = 5;
  double registration_stddev = 0.002;
  ClusterConfig cluster;
};

struct BaselineConfig {
  InputConfig input;
  ReferenceConfig reference;
  TemporalConfig temporal;
  ClusterConfig cluster;
  C2CConfig c2c;
  M3C2Config m3c2;
};

}  // namespace baseline

#endif  // BASELINE_COMMON_CONFIG_HPP
