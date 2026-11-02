#ifndef BASELINE_COMMON_TYPES_HPP
#define BASELINE_COMMON_TYPES_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace baseline {

using PointT = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct C2CPointRecord {
  Vec3 point;
  double nn_distance = 0.0;
  bool changed = false;
  int cluster_id = -1;
};

struct M3C2CoreRecord {
  Vec3 core_point;
  Vec3 normal;
  double signed_distance = 0.0;
  double lod95 = 0.0;
  bool significant = false;
  int cluster_id = -1;
  int reference_support = 0;
  int query_support = 0;
};

struct ClusterRecord {
  int id = -1;
  std::string method;
  Vec3 center;
  Vec3 bbox_min;
  Vec3 bbox_max;
  int support_count = 0;
  double score = 0.0;
};

struct CycleRuntimeRecord {
  uint64_t cycle_index = 0;
  double total_ms = 0.0;
  double fusion_ms = 0.0;
  double method_ms = 0.0;
  double clustering_ms = 0.0;
  double io_ms = 0.0;
};

struct C2CResult {
  std::vector<C2CPointRecord> point_records;
  std::vector<ClusterRecord> clusters;
};

struct M3C2Result {
  std::vector<M3C2CoreRecord> core_records;
  std::vector<ClusterRecord> clusters;
};

}  // namespace baseline

#endif  // BASELINE_COMMON_TYPES_HPP
