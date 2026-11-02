#include "baseline/core/temporal_window_fusion.hpp"

#include <pcl/filters/voxel_grid.h>

namespace baseline {

void TemporalWindowFusion::SetConfig(const TemporalFusionConfig& config) {
  config_ = config;
}

void TemporalWindowFusion::PushCloud(const PointCloudPtr& cloud, double stamp_sec) {
  frames_.push_back(std::make_pair(stamp_sec, cloud));
  const size_t max_frames = static_cast<size_t>(std::max(1, config_.window_frames));
  while (frames_.size() > max_frames) {
    frames_.pop_front();
  }
}

size_t TemporalWindowFusion::FrameCount() const {
  return frames_.size();
}

void TemporalWindowFusion::Reset() {
  frames_.clear();
}

PointCloudPtr TemporalWindowFusion::BuildFusedCloud() const {
  PointCloudPtr merged(new PointCloud());
  for (const auto& entry : frames_) {
    if (!entry.second) {
      continue;
    }
    *merged += *entry.second;
  }
  if (merged->empty()) {
    return merged;
  }

  pcl::VoxelGrid<PointT> voxel_grid;
  voxel_grid.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
  voxel_grid.setInputCloud(merged);
  PointCloudPtr fused(new PointCloud());
  voxel_grid.filter(*fused);
  return fused;
}

}  // namespace baseline
