#include "baseline/m3c2/m3c2_baseline.hpp"

#include <Eigen/Dense>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <algorithm>
#include <cmath>

namespace baseline {

namespace {

Eigen::Vector3d ToEigen(const PointT& point) {
  return Eigen::Vector3d(point.x, point.y, point.z);
}

Eigen::Vector3d OrientNormal(Eigen::Vector3d normal) {
  const int axis = [&]() {
    const Eigen::Vector3d abs_normal = normal.cwiseAbs();
    int best = 0;
    if (abs_normal.y() > abs_normal.x()) {
      best = 1;
    }
    if (abs_normal.z() > abs_normal[best]) {
      best = 2;
    }
    return best;
  }();
  if (normal[axis] < 0.0) {
    normal = -normal;
  }
  return normal.normalized();
}

bool EstimateNormal(const PointCloudPtr& cloud,
                    pcl::KdTreeFLANN<PointT>* kdtree,
                    const PointT& point,
                    double radius,
                    int min_support,
                    Eigen::Vector3d* normal) {
  if (!cloud || !kdtree || !normal) {
    return false;
  }

  std::vector<int> indices;
  std::vector<float> distances;
  if (kdtree->radiusSearch(point, radius, indices, distances) < min_support) {
    return false;
  }

  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const int idx : indices) {
    centroid += ToEigen(cloud->at(static_cast<size_t>(idx)));
  }
  centroid /= static_cast<double>(indices.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const int idx : indices) {
    const Eigen::Vector3d delta = ToEigen(cloud->at(static_cast<size_t>(idx))) - centroid;
    covariance += delta * delta.transpose();
  }
  covariance /= static_cast<double>(indices.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  *normal = OrientNormal(solver.eigenvectors().col(0));
  return true;
}

std::vector<double> CollectProjectedSupport(const PointCloudPtr& cloud,
                                            pcl::KdTreeFLANN<PointT>* kdtree,
                                            const Eigen::Vector3d& core_point,
                                            const Eigen::Vector3d& normal,
                                            double radius,
                                            double half_length) {
  std::vector<double> projections;
  if (!cloud || !kdtree) {
    return projections;
  }

  PointT search_point;
  search_point.x = static_cast<float>(core_point.x());
  search_point.y = static_cast<float>(core_point.y());
  search_point.z = static_cast<float>(core_point.z());

  const double sphere_radius = std::sqrt(radius * radius + half_length * half_length);
  std::vector<int> indices;
  std::vector<float> distances;
  kdtree->radiusSearch(search_point, sphere_radius, indices, distances);
  projections.reserve(indices.size());

  for (const int idx : indices) {
    const Eigen::Vector3d delta = ToEigen(cloud->at(static_cast<size_t>(idx))) - core_point;
    const double along = delta.dot(normal);
    const Eigen::Vector3d radial = delta - along * normal;
    if (std::abs(along) <= half_length && radial.squaredNorm() <= radius * radius) {
      projections.push_back(along);
    }
  }

  return projections;
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const double value : values) {
    sum += value;
  }
  return sum / static_cast<double>(values.size());
}

double Variance(const std::vector<double>& values, double mean) {
  if (values.size() < 2) {
    return 0.0;
  }
  double acc = 0.0;
  for (const double value : values) {
    const double delta = value - mean;
    acc += delta * delta;
  }
  return acc / static_cast<double>(values.size() - 1);
}

}  // namespace

void M3C2Baseline::SetConfig(const M3C2Config& config) {
  config_ = config;
}

M3C2Result M3C2Baseline::Run(const PointCloudPtr& reference_cloud,
                             const PointCloudPtr& query_cloud) const {
  M3C2Result result;
  if (!reference_cloud || !query_cloud || reference_cloud->empty() || query_cloud->empty()) {
    return result;
  }

  pcl::VoxelGrid<PointT> core_sampler;
  core_sampler.setLeafSize(config_.core_voxel_size,
                           config_.core_voxel_size,
                           config_.core_voxel_size);
  core_sampler.setInputCloud(reference_cloud);
  PointCloudPtr core_cloud(new PointCloud());
  core_sampler.filter(*core_cloud);

  pcl::KdTreeFLANN<PointT> reference_kdtree;
  reference_kdtree.setInputCloud(reference_cloud);
  pcl::KdTreeFLANN<PointT> query_kdtree;
  query_kdtree.setInputCloud(query_cloud);

  PointCloudPtr significant_core_cloud(new PointCloud());
  std::vector<size_t> significant_indices;
  result.core_records.reserve(core_cloud->size());

  for (size_t i = 0; i < core_cloud->size(); ++i) {
    const auto& core_point = core_cloud->at(i);
    M3C2CoreRecord record;
    record.core_point = {core_point.x, core_point.y, core_point.z};

    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    if (!EstimateNormal(reference_cloud,
                        &reference_kdtree,
                        core_point,
                        config_.normal_radius,
                        config_.min_support,
                        &normal)) {
      result.core_records.push_back(record);
      continue;
    }
    record.normal = {normal.x(), normal.y(), normal.z()};

    const Eigen::Vector3d core = ToEigen(core_point);
    const auto reference_proj = CollectProjectedSupport(reference_cloud,
                                                        &reference_kdtree,
                                                        core,
                                                        normal,
                                                        config_.cylinder_radius,
                                                        config_.cylinder_half_length);
    const auto query_proj = CollectProjectedSupport(query_cloud,
                                                    &query_kdtree,
                                                    core,
                                                    normal,
                                                    config_.cylinder_radius,
                                                    config_.cylinder_half_length);
    record.reference_support = static_cast<int>(reference_proj.size());
    record.query_support = static_cast<int>(query_proj.size());

    if (record.reference_support < config_.min_support ||
        record.query_support < config_.min_support) {
      result.core_records.push_back(record);
      continue;
    }

    const double mean_ref = Mean(reference_proj);
    const double mean_query = Mean(query_proj);
    const double var_ref = Variance(reference_proj, mean_ref);
    const double var_query = Variance(query_proj, mean_query);

    record.signed_distance = mean_query - mean_ref;
    record.lod95 = 1.96 * (std::sqrt(var_ref / record.reference_support +
                                     var_query / record.query_support) +
                           config_.registration_stddev);
    record.significant = std::abs(record.signed_distance) > record.lod95;

    if (record.significant) {
      significant_indices.push_back(i);
      significant_core_cloud->push_back(core_point);
    }

    result.core_records.push_back(record);
  }

  ClusterExtractor extractor;
  extractor.SetConfig(config_.cluster);
  result.clusters = extractor.Extract(significant_core_cloud, "m3c2");
  for (const auto& cluster : result.clusters) {
    for (size_t j = 0; j < significant_indices.size(); ++j) {
      const auto& point = significant_core_cloud->at(j);
      if (point.x >= cluster.bbox_min.x && point.x <= cluster.bbox_max.x &&
          point.y >= cluster.bbox_min.y && point.y <= cluster.bbox_max.y &&
          point.z >= cluster.bbox_min.z && point.z <= cluster.bbox_max.z) {
        result.core_records[significant_indices[j]].cluster_id = cluster.id;
      }
    }
  }

  return result;
}

}  // namespace baseline
