#include "baseline/core/cluster_extractor.hpp"

#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <limits>

namespace baseline {

void ClusterExtractor::SetConfig(const ClusterConfig& config) {
  config_ = config;
}

std::vector<ClusterRecord> ClusterExtractor::Extract(const PointCloudPtr& cloud,
                                                     const std::string& method) const {
  std::vector<ClusterRecord> out;
  if (!cloud || cloud->empty()) {
    return out;
  }

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> extraction;
  extraction.setClusterTolerance(config_.tolerance);
  extraction.setMinClusterSize(config_.min_cluster_size);
  extraction.setMaxClusterSize(config_.max_cluster_size);
  extraction.setSearchMethod(tree);
  extraction.setInputCloud(cloud);
  extraction.extract(cluster_indices);

  int next_id = 0;
  for (const auto& indices : cluster_indices) {
    ClusterRecord record;
    record.id = next_id++;
    record.method = method;
    record.support_count = static_cast<int>(indices.indices.size());
    record.bbox_min = {std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::max()};
    record.bbox_max = {-std::numeric_limits<double>::max(),
                       -std::numeric_limits<double>::max(),
                       -std::numeric_limits<double>::max()};

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    for (const int idx : indices.indices) {
      const auto& pt = cloud->at(static_cast<size_t>(idx));
      sx += pt.x;
      sy += pt.y;
      sz += pt.z;
      record.bbox_min.x = std::min(record.bbox_min.x, static_cast<double>(pt.x));
      record.bbox_min.y = std::min(record.bbox_min.y, static_cast<double>(pt.y));
      record.bbox_min.z = std::min(record.bbox_min.z, static_cast<double>(pt.z));
      record.bbox_max.x = std::max(record.bbox_max.x, static_cast<double>(pt.x));
      record.bbox_max.y = std::max(record.bbox_max.y, static_cast<double>(pt.y));
      record.bbox_max.z = std::max(record.bbox_max.z, static_cast<double>(pt.z));
    }

    const double inv = 1.0 / static_cast<double>(record.support_count);
    record.center = {sx * inv, sy * inv, sz * inv};
    out.push_back(record);
  }

  return out;
}

}  // namespace baseline
