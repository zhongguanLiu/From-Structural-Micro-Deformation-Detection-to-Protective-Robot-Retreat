#include "baseline/core/reference_manager.hpp"

#include <pcl/filters/voxel_grid.h>

namespace baseline {

void ReferenceManager::SetConfig(const ReferenceConfig& config) {
  config_ = config;
}

bool ReferenceManager::AddCloud(const PointCloudPtr& cloud) {
  if (frozen_) {
    return true;
  }
  if (!cloud) {
    return false;
  }

  *accumulated_cloud_ += *cloud;
  ++accumulated_frames_;
  if (accumulated_frames_ >= config_.init_frames) {
    Freeze();
  }
  return frozen_;
}

bool ReferenceManager::IsFrozen() const {
  return frozen_;
}

void ReferenceManager::Reset() {
  accumulated_frames_ = 0;
  frozen_ = false;
  accumulated_cloud_.reset(new PointCloud());
  frozen_cloud_.reset(new PointCloud());
}

bool ReferenceManager::ForceFreeze() {
  if (frozen_ || !accumulated_cloud_ || accumulated_cloud_->empty()) {
    return frozen_;
  }
  Freeze();
  return frozen_;
}

PointCloudPtr ReferenceManager::GetReferenceCloud() const {
  return frozen_cloud_;
}

void ReferenceManager::Freeze() {
  pcl::VoxelGrid<PointT> voxel_grid;
  voxel_grid.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
  voxel_grid.setInputCloud(accumulated_cloud_);
  voxel_grid.filter(*frozen_cloud_);
  frozen_ = true;
}

}  // namespace baseline
