#include "deform_monitor_v3/v1/reference_graph.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"
#include "deform_monitor_v3/v1/math.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

struct CellExposure {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StableId frame_id;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;
};

struct LocalReferencePoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position_R = Eigen::Vector3d::Zero();
  StableId point_id;
  StableId frame_id;
};

using ExposureMap = std::map<StableId, CellExposure>;

struct DisjointSet {
  explicit DisjointSet(size_t count) : parent(count), rank(count, 0U) {
    for (size_t i = 0; i < count; ++i) parent[i] = static_cast<uint32_t>(i);
  }
  uint32_t Find(uint32_t value) {
    if (parent[value] != value) parent[value] = Find(parent[value]);
    return parent[value];
  }
  void Unite(uint32_t a, uint32_t b) {
    a = Find(a); b = Find(b); if (a == b) return;
    if (rank[a] < rank[b]) std::swap(a, b);
    parent[b] = a; if (rank[a] == rank[b]) ++rank[a];
  }
  std::vector<uint32_t> parent;
  std::vector<uint32_t> rank;
};

Eigen::Vector4d GeometryProbability(const Eigen::Vector3d& eigenvalues,
                                    const CalibrationConfig& calibration) {
  const double scale = std::max(1.0e-15, eigenvalues[2]);
  Eigen::Vector4d feature;
  feature << (eigenvalues[1] - eigenvalues[0]) / scale,
             (eigenvalues[2] - eigenvalues[1]) / scale,
             eigenvalues[0] / scale, 1.0;
  const Eigen::Vector4d logits = calibration.geometry_softmax_weights * feature +
                                 calibration.geometry_softmax_bias;
  const std::vector<double> probability = Softmax(
      {logits[0], logits[1], logits[2], logits[3]});
  Eigen::Vector4d output;
  for (int i = 0; i < 4; ++i) output[i] = probability[static_cast<size_t>(i)];
  return output;
}

StableId EdgeId(const StableId& first, const StableId& second) {
  CanonicalBytes bytes;
  if (second < first) { bytes.AddId(second); bytes.AddId(first); }
  else { bytes.AddId(first); bytes.AddId(second); }
  return MakeStableId("reference-edge-v1", bytes);
}

std::vector<double> LinkGraphDistances(const ReferenceSnapshot& snapshot,
                                       uint32_t gdu_index,
                                       uint32_t source_surfel) {
  std::vector<std::vector<std::pair<uint32_t, double>>> adjacency(snapshot.surfels.size());
  for (const GraphEdge& edge : snapshot.edges) {
    if (edge.state != ReferenceEdgeState::LINK ||
        snapshot.surfels[edge.first].gdu_index != gdu_index ||
        snapshot.surfels[edge.second].gdu_index != gdu_index) continue;
    adjacency[edge.first].push_back(std::make_pair(edge.second, edge.distance_m));
    adjacency[edge.second].push_back(std::make_pair(edge.first, edge.distance_m));
  }
  std::vector<double> distance(snapshot.surfels.size(),
                               std::numeric_limits<double>::infinity());
  using QueueItem = std::pair<double, uint32_t>;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;
  distance[source_surfel] = 0.0;
  queue.push(std::make_pair(0.0, source_surfel));
  while (!queue.empty()) {
    const QueueItem item = queue.top(); queue.pop();
    if (item.first > distance[item.second]) continue;
    for (const auto& edge : adjacency[item.second]) {
      const double candidate = item.first + edge.second;
      if (candidate < distance[edge.first]) {
        distance[edge.first] = candidate;
        queue.push(std::make_pair(candidate, edge.first));
      }
    }
  }
  return distance;
}

std::vector<uint32_t> GeodesicSeeds(const ReferenceSnapshot& snapshot,
                                    uint32_t gdu_index, double spacing,
                                    size_t maximum_count,
                                    double tie_tolerance,
                                    bool* maximum_exceeded = nullptr) {
  if (maximum_exceeded) *maximum_exceeded = false;
  std::vector<uint32_t> seeds;
  if (gdu_index >= snapshot.gdus.size() ||
      snapshot.gdus[gdu_index].surfel_indices.empty() || maximum_count == 0U)
    return seeds;
  const GduDefinition& gdu = snapshot.gdus[gdu_index];
  seeds.push_back(gdu.surfel_indices.front());
  std::vector<double> nearest(snapshot.surfels.size(),
                              std::numeric_limits<double>::infinity());
  std::vector<double> distance = LinkGraphDistances(snapshot, gdu_index, seeds.front());
  for (uint32_t surfel : gdu.surfel_indices) nearest[surfel] = distance[surfel];
  while (true) {
    double farthest_distance = -1.0;
    uint32_t farthest = gdu.surfel_indices.front();
    for (uint32_t surfel : gdu.surfel_indices) {
      if (nearest[surfel] > farthest_distance + tie_tolerance ||
          (std::abs(nearest[surfel] - farthest_distance) <= tie_tolerance &&
           snapshot.surfels[surfel].id < snapshot.surfels[farthest].id)) {
        farthest_distance = nearest[surfel];
        farthest = surfel;
      }
    }
    if (!(farthest_distance > spacing)) break;
    if (seeds.size() >= maximum_count) {
      if (maximum_exceeded) *maximum_exceeded = true;
      return {};
    }
    seeds.push_back(farthest);
    distance = LinkGraphDistances(snapshot, gdu_index, farthest);
    for (uint32_t surfel : gdu.surfel_indices)
      nearest[surfel] = std::min(nearest[surfel], distance[surfel]);
  }
  return seeds;
}

std::vector<std::vector<uint32_t>> ConnectedBoundaryComponents(
    const ReferenceSnapshot& snapshot,
    const std::vector<uint32_t>& edge_indices) {
  std::vector<std::vector<uint32_t>> output;
  if (edge_indices.empty()) return output;
  DisjointSet sets(edge_indices.size());
  std::map<uint32_t, size_t> first_edge_at_surfel;
  for (size_t local = 0; local < edge_indices.size(); ++local) {
    const GraphEdge& edge = snapshot.edges[edge_indices[local]];
    for (uint32_t surfel : {edge.first, edge.second}) {
      const auto inserted = first_edge_at_surfel.emplace(surfel, local);
      if (!inserted.second)
        sets.Unite(static_cast<uint32_t>(local),
                   static_cast<uint32_t>(inserted.first->second));
    }
  }
  std::map<uint32_t, std::vector<uint32_t>> grouped;
  for (size_t local = 0; local < edge_indices.size(); ++local)
    grouped[sets.Find(static_cast<uint32_t>(local))].push_back(edge_indices[local]);
  for (auto& group : grouped) {
    std::sort(group.second.begin(), group.second.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                return snapshot.edges[lhs].id < snapshot.edges[rhs].id;
              });
    output.push_back(group.second);
  }
  std::sort(output.begin(), output.end(), [&](const auto& lhs, const auto& rhs) {
    return snapshot.edges[lhs.front()].id < snapshot.edges[rhs.front()].id;
  });
  return output;
}

Eigen::Matrix3d StableNearestProjector(const Eigen::Matrix3d& average,
                                       int rank, double tolerance) {
  if (rank <= 0) return Eigen::Matrix3d::Zero();
  if (rank >= 3) return Eigen::Matrix3d::Identity();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
      0.5 * (average + average.transpose()));
  if (eigen.info() != Eigen::Success) return Eigen::Matrix3d::Zero();
  const double scale = std::max(1.0, eigen.eigenvalues().cwiseAbs().maxCoeff());
  const int split = 3 - rank;
  if (eigen.eigenvalues()[split] - eigen.eigenvalues()[split - 1] <=
      tolerance * scale)
    return Eigen::Matrix3d::Zero();
  Eigen::Matrix3d projector = Eigen::Matrix3d::Zero();
  for (int mode = split; mode < 3; ++mode)
    projector += eigen.eigenvectors().col(mode) *
                 eigen.eigenvectors().col(mode).transpose();
  return projector;
}

Eigen::Matrix3d AxesFromProjector(const Eigen::Matrix3d& projector, int rank,
                                  double tolerance, bool* valid) {
  if (valid) *valid = false;
  Eigen::Matrix3d axes = Eigen::Matrix3d::Zero();
  if (rank <= 0 || rank > 3 || !projector.allFinite()) return axes;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
      0.5 * (projector + projector.transpose()));
  if (eigen.info() != Eigen::Success) return axes;
  const int split = 3 - rank;
  if (split > 0 &&
      eigen.eigenvalues()[split] - eigen.eigenvalues()[split - 1] <=
          tolerance)
    return axes;
  for (int column = 0; column < rank; ++column) {
    Eigen::Vector3d axis = eigen.eigenvectors().col(2 - column);
    // Freeze eigenvector sign for deterministic ids/serialization.
    int pivot = 0;
    axis.cwiseAbs().maxCoeff(&pivot);
    if (axis[pivot] < 0.0) axis = -axis;
    axes.col(column) = axis;
  }
  if (valid) *valid = true;
  return axes;
}

}  // namespace

std::vector<size_t> SelectReferenceGduCoverage(
    const ReferenceSnapshot& reference,
    const std::vector<GduDefinition,
                      Eigen::aligned_allocator<GduDefinition>>& candidates,
    const Eigen::Vector3d& frozen_sensor_origin_R, size_t maximum_count,
    double orientation_scale_m, double tie_tolerance) {
  struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d center_R = Eigen::Vector3d::Zero();
    Eigen::Matrix3d observable_projector_R = Eigen::Matrix3d::Zero();
  };
  std::vector<Feature, Eigen::aligned_allocator<Feature>> features;
  if (maximum_count == 0U || candidates.empty() ||
      !frozen_sensor_origin_R.allFinite() ||
      !std::isfinite(orientation_scale_m) || orientation_scale_m < 0.0 ||
      !std::isfinite(tie_tolerance) || tie_tolerance < 0.0) {
    return {};
  }
  features.reserve(candidates.size());
  for (const GduDefinition& candidate : candidates) {
    if (candidate.surfel_indices.empty()) return {};
    Feature feature;
    for (uint32_t surfel_index : candidate.surfel_indices) {
      if (surfel_index >= reference.surfels.size() ||
          !reference.surfels[surfel_index].center_R.allFinite() ||
          !reference.surfels[surfel_index]
               .observable_projector_R.allFinite()) {
        return {};
      }
      feature.center_R += reference.surfels[surfel_index].center_R;
      feature.observable_projector_R +=
          reference.surfels[surfel_index].observable_projector_R;
    }
    const double denominator =
        static_cast<double>(candidate.surfel_indices.size());
    feature.center_R /= denominator;
    feature.observable_projector_R /= denominator;
    features.push_back(feature);
  }
  const size_t retained = std::min(maximum_count, candidates.size());
  if (retained == candidates.size()) {
    std::vector<size_t> all(retained, 0U);
    for (size_t index = 0U; index < retained; ++index) all[index] = index;
    return all;
  }
  const auto better_tie = [&](size_t lhs, size_t rhs) {
    if (candidates[lhs].surfel_indices.size() !=
        candidates[rhs].surfel_indices.size()) {
      return candidates[lhs].surfel_indices.size() >
             candidates[rhs].surfel_indices.size();
    }
    return candidates[lhs].id < candidates[rhs].id;
  };

  // Keep the closest carrier, reserve half of the bounded budget for
  // the best-supported components, then spend the remainder on farthest
  // spatial/orientation coverage.  Pure farthest-point sampling can discard
  // every fragment of a densely observed object when another carrier happens
  // to have a nearby center, even though those discarded fragments contain
  // most of the local evidence.
  size_t first = 0U;
  double first_range =
      (features[first].center_R - frozen_sensor_origin_R).norm();
  for (size_t candidate = 1U; candidate < candidates.size(); ++candidate) {
    const double range =
        (features[candidate].center_R - frozen_sensor_origin_R).norm();
    if (range < first_range - tie_tolerance ||
        (std::abs(range - first_range) <= tie_tolerance &&
         better_tie(candidate, first))) {
      first = candidate;
      first_range = range;
    }
  }
  std::vector<size_t> selected{first};
  std::vector<bool> is_selected(candidates.size(), false);
  is_selected[first] = true;
  std::vector<size_t> support_order(candidates.size(), 0U);
  std::iota(support_order.begin(), support_order.end(), 0U);
  std::sort(support_order.begin(), support_order.end(),
            [&](size_t lhs, size_t rhs) { return better_tie(lhs, rhs); });
  const size_t support_reserve = retained / 2U;
  const size_t support_target =
      std::min(retained, support_reserve + 1U);  // closest + support reserve
  for (size_t candidate : support_order) {
    if (selected.size() >= support_target) break;
    if (is_selected[candidate]) continue;
    selected.push_back(candidate);
    is_selected[candidate] = true;
  }
  const auto feature_distance = [&](size_t lhs, size_t rhs) {
    const double position_squared =
        (features[lhs].center_R - features[rhs].center_R).squaredNorm();
    const double projector_squared =
        (features[lhs].observable_projector_R -
         features[rhs].observable_projector_R).squaredNorm();
    return std::sqrt(std::max(
        0.0, position_squared + orientation_scale_m * orientation_scale_m *
                                      projector_squared));
  };
  while (selected.size() < retained) {
    size_t farthest = candidates.size();
    double farthest_distance = -1.0;
    for (size_t candidate = 0U; candidate < candidates.size(); ++candidate) {
      if (is_selected[candidate]) continue;
      double nearest_distance = std::numeric_limits<double>::infinity();
      for (size_t existing : selected) {
        nearest_distance = std::min(
            nearest_distance, feature_distance(candidate, existing));
      }
      if (farthest == candidates.size() ||
          nearest_distance > farthest_distance + tie_tolerance ||
          (std::abs(nearest_distance - farthest_distance) <= tie_tolerance &&
           better_tie(candidate, farthest))) {
        farthest = candidate;
        farthest_distance = nearest_distance;
      }
    }
    if (farthest == candidates.size()) return {};
    selected.push_back(farthest);
    is_selected[farthest] = true;
  }
  return selected;
}

std::vector<bool> PoseControlSpatialExclusionMask(
    const ReferenceSnapshot& reference, uint32_t target_gdu_index,
    double radius_m) {
  std::vector<bool> excluded(reference.surfels.size(), false);
  if (target_gdu_index >= reference.gdus.size() ||
      !std::isfinite(radius_m) || radius_m < 0.0) {
    return excluded;
  }
  const double radius_squared = radius_m * radius_m;
  for (uint32_t owner = 0; owner < reference.surfels.size(); ++owner) {
    for (uint32_t target_surfel :
         reference.gdus[target_gdu_index].surfel_indices) {
      if (target_surfel >= reference.surfels.size()) continue;
      if ((reference.surfels[owner].center_R -
           reference.surfels[target_surfel].center_R).squaredNorm() <=
          radius_squared) {
        excluded[owner] = true;
        break;
      }
    }
  }
  return excluded;
}

int InterfaceSideForPoint(const ReferenceSnapshot& reference,
                          const InterfaceDefinition& interface,
                          uint32_t gdu_index,
                          const Eigen::Vector3d& position_R,
                          double tie_tolerance) {
  if (interface.cross_gdu) {
    if (gdu_index == interface.left_gdu) return -1;
    if (gdu_index == interface.right_gdu) return 1;
    return 0;
  }
  if (gdu_index != interface.left_gdu ||
      interface.left_gdu != interface.right_gdu) return 0;
  uint32_t nearest = std::numeric_limits<uint32_t>::max();
  double nearest_distance = std::numeric_limits<double>::infinity();
  int side = 0;
  const auto consider = [&](const std::vector<uint32_t>& region, int candidate_side,
                            uint32_t* nearest_surfel, double* best_distance,
                            int* best_side) {
    for (uint32_t surfel : region) {
      if (surfel >= reference.surfels.size()) continue;
      const double distance =
          (position_R - reference.surfels[surfel].center_R).norm();
      if (distance < *best_distance - tie_tolerance ||
          (std::abs(distance - *best_distance) <= tie_tolerance &&
           (*nearest_surfel == std::numeric_limits<uint32_t>::max() ||
            reference.surfels[surfel].id <
                reference.surfels[*nearest_surfel].id))) {
        *nearest_surfel = surfel;
        *best_distance = distance;
        *best_side = candidate_side;
      }
    }
  };
  consider(interface.left_region_surfels, -1, &nearest, &nearest_distance, &side);
  consider(interface.right_region_surfels, 1, &nearest, &nearest_distance, &side);
  return side;
}

ReferenceSnapshot ReferenceBuilder::Build(const std::deque<FrameBundle>& frames,
                                          uint32_t reference_epoch) const {
  ReferenceSnapshot snapshot;
  snapshot.epoch = reference_epoch;
  snapshot.world_frame = config_.resource.world_frame;
  const auto reference_resource_failure =
      [&](const char* stage, size_t value, size_t limit) {
        std::cerr << "[deform_v1] reference resource failure stage=" << stage
                  << " value=" << value << " limit=" << limit << std::endl;
        snapshot.reasons.push_back(ReasonCode::REFERENCE_RESOURCE_LIMIT);
      };
  if (frames.size() < static_cast<size_t>(config_.resource.reference_min_frames)) {
    snapshot.reasons.push_back(ReasonCode::REFERENCE_INSUFFICIENT_EXPOSURE);
    return snapshot;
  }
  snapshot.begin_stamp = frames.front().stamp;
  snapshot.end_stamp = frames.back().stamp;
  if ((snapshot.end_stamp - snapshot.begin_stamp).toSec() <
      config_.resource.reference_min_duration_s) {
    snapshot.reasons.push_back(ReasonCode::REFERENCE_INSUFFICIENT_EXPOSURE);
    return snapshot;
  }
  std::map<VoxelKey, ExposureMap> cells;
  std::vector<LocalReferencePoint,
      Eigen::aligned_allocator<LocalReferencePoint>> reference_points;
  std::map<VoxelKey, std::vector<size_t>> reference_points_by_cell;
  const int64_t neighborhood_radius_cells = static_cast<int64_t>(std::ceil(
      config_.resource.graph_neighbor_radius_m /
      config_.resource.surfel_seed_spacing_m));
  const long double neighborhood_width =
      2.0L * static_cast<long double>(neighborhood_radius_cells) + 1.0L;
  if (neighborhood_radius_cells < 0 ||
      neighborhood_width * neighborhood_width * neighborhood_width >
          static_cast<long double>(config_.resource.max_raw_buffer_points)) {
    reference_resource_failure("neighbor_lattice", static_cast<size_t>(
        neighborhood_width * neighborhood_width * neighborhood_width),
        config_.resource.max_raw_buffer_points);
    return snapshot;
  }
  Validity input_validity = Validity::VALID;
  uint64_t previous_frame_index = 0U;
  ros::Time previous_stamp;
  size_t accumulated_points = 0U;
  for (size_t frame_index = 0U; frame_index < frames.size(); ++frame_index) {
    const FrameBundle& frame = frames[frame_index];
    if (frame.validity == Validity::INVALID ||
        frame.session_id != config_.resource.session_id ||
        frame.world_frame != snapshot.world_frame || frame.stamp.isZero() ||
        frame.frame_id.empty() ||
        (frame_index > 0U &&
         (!(frame.stamp > previous_stamp) ||
          frame.accepted_frame_index <= previous_frame_index))) {
      snapshot.reasons.push_back(
          frame.world_frame != snapshot.world_frame
              ? ReasonCode::INPUT_FRAME_MISMATCH
              : ReasonCode::INPUT_SCHEMA_INVALID);
      return snapshot;
    }
    input_validity = CombineValidity(input_validity, frame.validity);
    snapshot.reasons.insert(snapshot.reasons.end(), frame.reasons.begin(),
                            frame.reasons.end());
    previous_stamp = frame.stamp;
    previous_frame_index = frame.accepted_frame_index;
    if (frame.points.size() > config_.resource.max_raw_buffer_points -
                                  std::min(config_.resource.max_raw_buffer_points,
                                           accumulated_points)) {
      reference_resource_failure("frame_points", frame.points.size(),
                                 config_.resource.max_raw_buffer_points);
      return snapshot;
    }
    accumulated_points += frame.points.size();
    if (accumulated_points > config_.resource.max_raw_buffer_points) {
      reference_resource_failure("accumulated_points", accumulated_points,
                                 config_.resource.max_raw_buffer_points);
      return snapshot;
    }
    for (const CanonicalPoint& point : frame.points) {
      if (point.point_id.empty() || !point.position_R.allFinite()) {
        snapshot.reasons.push_back(ReasonCode::INPUT_SCHEMA_INVALID);
        return snapshot;
      }
      VoxelKey key;
      if (!TryGridKey(point.position_R, config_.resource.grid_origin_m,
                      config_.resource.surfel_seed_spacing_m, &key)) {
        reference_resource_failure("point_grid_domain", 1U, 0U);
        return snapshot;
      }
      CellExposure& exposure = cells[key][frame.frame_id];
      exposure.frame_id = frame.frame_id;
      exposure.points.push_back(point.position_R);
      LocalReferencePoint stored;
      stored.position_R = point.position_R;
      stored.point_id = point.point_id;
      stored.frame_id = frame.frame_id;
      reference_points.push_back(stored);
      reference_points_by_cell[key].push_back(reference_points.size() - 1U);
    }
  }

  for (const auto& cell : cells) {
    // A surfel must be supported by the frozen number of independent frame
    // exposures; individual returns in one scan are never repetitions.
    // individual laser returns in one scan are not treated as independent
    // reference repetitions.
    if (cell.second.size() < static_cast<size_t>(
            config_.resource.reference_min_surfel_exposures)) continue;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> exposure_centers;
    size_t point_count = 0U;
    for (const auto& exposure_pair : cell.second) {
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      for (const Eigen::Vector3d& point : exposure_pair.second.points) center += point;
      center /= static_cast<double>(exposure_pair.second.points.size());
      exposure_centers.push_back(center);
      point_count += exposure_pair.second.points.size();
    }
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& value : exposure_centers) center += value;
    center /= static_cast<double>(exposure_centers.size());
    Eigen::Matrix3d centroid_scatter = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& value : exposure_centers) {
      const Eigen::Vector3d delta = value - center;
      centroid_scatter += delta * delta.transpose();
    }
    centroid_scatter /= static_cast<double>(exposure_centers.size() - 1U);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> stability_eigen(centroid_scatter);
    if (stability_eigen.info() != Eigen::Success) continue;
    // This covariance describes the uncertainty of the frozen reference
    // location.  It is intentionally distinct from the local point scatter
    // used below to infer observable geometric axes.
    // Exposure-center scatter already contains the independent return and
    // within-frame aggregation error.  Adding the nominal point variance here
    // would count it twice.  Epoch-common bias is represented separately on
    // the GDU and never shrinks with this exposure count.
    const Eigen::Matrix3d reference_covariance =
        centroid_scatter / static_cast<double>(exposure_centers.size());

    // Geometry is inferred from a deterministic bounded raw-point
    // neighbourhood, never from the across-frame centroid covariance above.
    std::vector<std::pair<double, size_t>> local_candidates;
    for (int64_t dx = -neighborhood_radius_cells;
         dx <= neighborhood_radius_cells; ++dx)
      for (int64_t dy = -neighborhood_radius_cells;
           dy <= neighborhood_radius_cells; ++dy)
        for (int64_t dz = -neighborhood_radius_cells;
             dz <= neighborhood_radius_cells; ++dz) {
          const int64_t x = static_cast<int64_t>(cell.first.x) + dx;
          const int64_t y = static_cast<int64_t>(cell.first.y) + dy;
          const int64_t z = static_cast<int64_t>(cell.first.z) + dz;
          if (x < std::numeric_limits<int32_t>::min() ||
              x > std::numeric_limits<int32_t>::max() ||
              y < std::numeric_limits<int32_t>::min() ||
              y > std::numeric_limits<int32_t>::max() ||
              z < std::numeric_limits<int32_t>::min() ||
              z > std::numeric_limits<int32_t>::max())
            continue;
          const auto nearby = reference_points_by_cell.find(
              VoxelKey{static_cast<int32_t>(x), static_cast<int32_t>(y),
                       static_cast<int32_t>(z)});
          if (nearby == reference_points_by_cell.end()) continue;
          for (size_t point_index : nearby->second) {
            const double distance =
                (reference_points[point_index].position_R - center).norm();
            if (distance <= config_.resource.graph_neighbor_radius_m)
              local_candidates.emplace_back(distance, point_index);
          }
        }
    std::sort(local_candidates.begin(), local_candidates.end(),
              [&](const auto& lhs, const auto& rhs) {
                if (std::abs(lhs.first - rhs.first) > config_.numerical.tie_tolerance)
                  return lhs.first < rhs.first;
                const LocalReferencePoint& left = reference_points[lhs.second];
                const LocalReferencePoint& right = reference_points[rhs.second];
                if (left.point_id != right.point_id) return left.point_id < right.point_id;
                return left.frame_id < right.frame_id;
              });
    // Balance the bounded neighbourhood across frame groups.  Taking the
    // globally nearest k returns would usually select many correlated rays
    // from one dense LiDAR scan and could both fabricate confidence and erase
    // cross-frame repeatability.
    std::map<StableId, std::vector<std::pair<double, size_t>>> by_frame;
    for (const auto& candidate : local_candidates)
      by_frame[reference_points[candidate.second].frame_id].push_back(candidate);
    std::vector<std::pair<double, size_t>> balanced_candidates;
    if (config_.resource.bounded_reference_search) {
      // With many exposures, nearest-per-frame can spend all 36 slots on
      // repeated samples of ONE position and turn a stable plane into OTHER.
      // Frame-round-robin farthest-point sampling preserves spatial spread
      // without treating repeated exposures as independent geometry rays.
      // Distance caches bound this to O(k*n), k <= graph_max_neighbors.
      std::map<StableId, std::vector<size_t>> indices_by_frame;
      for (size_t i = 0; i < local_candidates.size(); ++i)
        indices_by_frame[reference_points[local_candidates[i].second].frame_id].push_back(i);
      std::vector<double> nearest_squared(local_candidates.size(),
                                         std::numeric_limits<double>::infinity());
      std::vector<bool> used(local_candidates.size(), false);
      while (balanced_candidates.size() < static_cast<size_t>(config_.resource.graph_max_neighbors)) {
        bool added = false;
        for (const auto& frame : indices_by_frame) {
          size_t chosen = local_candidates.size();
          for (size_t i : frame.second) {
            if (!used[i] && (chosen == local_candidates.size() ||
                            nearest_squared[i] > nearest_squared[chosen])) chosen = i;
          }
          if (chosen == local_candidates.size()) continue;
          used[chosen] = true;
          balanced_candidates.push_back(local_candidates[chosen]);
          added = true;
          const Eigen::Vector3d& position = reference_points[local_candidates[chosen].second].position_R;
          for (size_t i = 0; i < local_candidates.size(); ++i) {
            if (!used[i]) nearest_squared[i] = std::min(nearest_squared[i],
                (reference_points[local_candidates[i].second].position_R - position).squaredNorm());
          }
          if (balanced_candidates.size() >= static_cast<size_t>(config_.resource.graph_max_neighbors)) break;
        }
        if (!added) break;
      }
    } else for (size_t depth = 0;
         balanced_candidates.size() <
             static_cast<size_t>(config_.resource.graph_max_neighbors);
         ++depth) {
      bool added = false;
      for (const auto& frame_candidates : by_frame) {
        if (depth >= frame_candidates.second.size()) continue;
        balanced_candidates.push_back(frame_candidates.second[depth]);
        added = true;
        if (balanced_candidates.size() >=
            static_cast<size_t>(config_.resource.graph_max_neighbors)) break;
      }
      if (!added) break;
    }
    local_candidates.swap(balanced_candidates);
    std::set<StableId> geometry_frames;
    Eigen::Vector3d geometry_center = Eigen::Vector3d::Zero();
    for (const auto& candidate : local_candidates) {
      const LocalReferencePoint& point = reference_points[candidate.second];
      geometry_center += point.position_R;
      geometry_frames.insert(point.frame_id);
    }
    if (local_candidates.size() < 3U || geometry_frames.size() < 3U) continue;
    geometry_center /= static_cast<double>(local_candidates.size());
    Eigen::Matrix3d geometry_scatter = Eigen::Matrix3d::Zero();
    for (const auto& candidate : local_candidates) {
      const Eigen::Vector3d delta =
          reference_points[candidate.second].position_R - geometry_center;
      geometry_scatter += delta * delta.transpose();
    }
    geometry_scatter /= static_cast<double>(local_candidates.size() - 1U);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(geometry_scatter);
    if (eigen.info() != Eigen::Success) continue;
    const double geometry_scale = std::max(
        config_.numerical.psd_eigenvalue_tolerance,
        eigen.eigenvalues().maxCoeff());
    const double minimum_spread_m2 =
        config_.calibration.independent_point_sigma_m *
        config_.calibration.independent_point_sigma_m;
    const bool sufficient_geometry_spread =
        eigen.eigenvalues().maxCoeff() > minimum_spread_m2;
    const bool stable_surface_direction = sufficient_geometry_spread &&
        (eigen.eigenvalues()[1] - eigen.eigenvalues()[0]) / geometry_scale >=
            config_.calibration.geometry_min_eigengap_ratio;
    const bool stable_curve_direction = sufficient_geometry_spread &&
        (eigen.eigenvalues()[2] - eigen.eigenvalues()[1]) / geometry_scale >=
            config_.calibration.geometry_min_eigengap_ratio;
    // Deterministic leave-one-frame-out jackknife: an entire LiDAR frame is
    // removed at once, so dense returns in one scan cannot create fictitious
    // directional certainty.
    Eigen::Vector4d resampled_probability = Eigen::Vector4d::Zero();
    Eigen::Matrix3d resampled_surface = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d resampled_curve = Eigen::Matrix3d::Zero();
    size_t valid_resamples = 0U;
    for (const StableId& held_out_frame : geometry_frames) {
      Eigen::Vector3d resample_center = Eigen::Vector3d::Zero();
      size_t resample_count = 0U;
      for (const auto& candidate : local_candidates) {
        const LocalReferencePoint& point = reference_points[candidate.second];
        if (point.frame_id == held_out_frame) continue;
        resample_center += point.position_R;
        ++resample_count;
      }
      if (resample_count < 3U) continue;
      resample_center /= static_cast<double>(resample_count);
      Eigen::Matrix3d resample_scatter = Eigen::Matrix3d::Zero();
      for (const auto& candidate : local_candidates) {
        const LocalReferencePoint& point = reference_points[candidate.second];
        if (point.frame_id == held_out_frame) continue;
        const Eigen::Vector3d delta = point.position_R - resample_center;
        resample_scatter += delta * delta.transpose();
      }
      resample_scatter /= static_cast<double>(resample_count - 1U);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> resample_eigen(
          resample_scatter);
      if (resample_eigen.info() != Eigen::Success) continue;
      resampled_probability += GeometryProbability(
          resample_eigen.eigenvalues(), config_.calibration);
      const Eigen::Vector3d normal = resample_eigen.eigenvectors().col(0);
      const Eigen::Vector3d tangent = resample_eigen.eigenvectors().col(2);
      resampled_surface += normal * normal.transpose();
      resampled_curve += Eigen::Matrix3d::Identity() -
                         tangent * tangent.transpose();
      ++valid_resamples;
    }
    if (valid_resamples < 3U) {
      // Position support can freeze an occupancy carrier while still yielding
      // fewer than three valid frame-block geometry replicates.  Do not
      // replace that missing resampling uncertainty by the full-sample
      // eigensystem: doing so would fabricate a confident plane/curve from an
      // underdetermined reference. Preserve the carrier under the explicit
      // OTHER geometry hypothesis instead.
      resampled_probability << 0.0, 0.0, 0.0, 1.0;
      resampled_surface.setZero();
      resampled_curve.setZero();
      valid_resamples = 1U;
    }
    resampled_probability /= static_cast<double>(valid_resamples);
    resampled_probability /= resampled_probability.sum();
    resampled_surface /= static_cast<double>(valid_resamples);
    resampled_curve /= static_cast<double>(valid_resamples);
    Surfel surfel;
    CanonicalBytes id_fields;
    id_fields.AddI64(cell.first.x); id_fields.AddI64(cell.first.y); id_fields.AddI64(cell.first.z);
    for (const auto& exposure_pair : cell.second) id_fields.AddId(exposure_pair.first);
    surfel.id = MakeStableId("surfel-v1", id_fields);
    surfel.center_R = center;
    surfel.covariance_R = reference_covariance;
    surfel.exposure_count = static_cast<uint32_t>(cell.second.size());
    surfel.point_count = static_cast<uint32_t>(point_count);
    for (const auto& exposure_pair : cell.second)
      for (const Eigen::Vector3d& point : exposure_pair.second.points)
        surfel.support_radius_m = std::max(
            surfel.support_radius_m, (point - center).norm());
    if (!std::isfinite(surfel.support_radius_m)) {
      snapshot.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
      snapshot.surfels.clear();
      return snapshot;
    }
    const double geometry_tolerance = std::max(
        config_.numerical.psd_eigenvalue_tolerance,
        config_.calibration.geometry_min_eigengap_ratio);
    if (stable_surface_direction)
      surfel.hypothesis_projector_R[0] = StableNearestProjector(
          resampled_surface, 1, geometry_tolerance);
    else
      surfel.hypothesis_projector_R[0].setZero();
    if (stable_curve_direction)
      surfel.hypothesis_projector_R[1] = StableNearestProjector(
          resampled_curve, 2, geometry_tolerance);
    else
      surfel.hypothesis_projector_R[1].setZero();
    const bool stable_full_rank =
        geometry_frames.size() >= static_cast<size_t>(
            config_.resource.reference_min_surfel_exposures) &&
        eigen.eigenvalues().minCoeff() / geometry_scale >=
            config_.calibration.geometry_min_eigengap_ratio;
    if (stable_full_rank)
      surfel.hypothesis_projector_R[2].setIdentity();
    else
      surfel.hypothesis_projector_R[2].setZero();
    surfel.hypothesis_projector_R[3] = Eigen::Matrix3d::Zero();
    if (surfel.hypothesis_projector_R[0].trace() < 0.5)
      resampled_probability[0] = 0.0;
    if (surfel.hypothesis_projector_R[1].trace() < 1.5)
      resampled_probability[1] = 0.0;
    if (!stable_full_rank) resampled_probability[2] = 0.0;
    if (!(resampled_probability.head<3>().sum() > 0.0))
      resampled_probability << 0.0, 0.0, 0.0, 1.0;
    else
      resampled_probability /= resampled_probability.sum();
    // Qualify every physical geometry hypothesis in its own idempotent
    // observable subspace.  The probability-weighted sum of projectors is
    // generally not a projector: using P_mix S P_mix^T would attenuate an
    // unstable direction approximately by the square of a small hypothesis
    // probability and could freeze a scan-phase/occlusion boundary as a
    // deformation coordinate.  Tangential ray sliding on a plane remains
    // admissible because the surface hypothesis is tested only along its
    // normal.
    const double stability_limit =
        config_.calibration.reference_stability_sigma *
        config_.calibration.independent_point_sigma_m;
    for (int h = 0; h < 3; ++h) {
      Eigen::Matrix3d& projector =
          surfel.hypothesis_projector_R[static_cast<size_t>(h)];
      if (!(resampled_probability[h] > 0.0) || projector.trace() < 0.5)
        continue;
      const Eigen::Matrix3d projected_scatter =
          projector * centroid_scatter * projector.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> stability_eigenvalues(
          0.5 * (projected_scatter + projected_scatter.transpose()),
          Eigen::EigenvaluesOnly);
      bool stable_hypothesis =
          stability_eigenvalues.info() == Eigen::Success &&
          stability_eigenvalues.eigenvalues().maxCoeff() <=
              stability_limit * stability_limit;
      // A bounded leave-one-exposure prediction check prevents a pair of
      // separated visibility modes from passing merely because their average
      // variance is diluted by many repeated exposures.  This is an XYZ/time
      // repeatability test only; no object identity or preselected region is
      // available here.
      for (size_t held_out = 0U;
           stable_hypothesis && held_out < exposure_centers.size();
           ++held_out) {
        const Eigen::Vector3d other_center =
            (center * static_cast<double>(exposure_centers.size()) -
             exposure_centers[held_out]) /
            static_cast<double>(exposure_centers.size() - 1U);
        const Eigen::Vector3d held_out_residual =
            projector * (exposure_centers[held_out] - other_center);
        stable_hypothesis = held_out_residual.norm() <= stability_limit;
      }
      if (!stable_hypothesis) {
        resampled_probability[h] = 0.0;
        projector.setZero();
      }
    }
    if (!(resampled_probability.head<3>().sum() > 0.0)) {
      resampled_probability << 0.0, 0.0, 0.0, 1.0;
      for (int h = 0; h < 3; ++h)
        surfel.hypothesis_projector_R[static_cast<size_t>(h)].setZero();
    } else {
      resampled_probability /= resampled_probability.sum();
    }
    surfel.geometry_probability = resampled_probability;
    surfel.observable_projector_R.setZero();
    for (int h = 0; h < 4; ++h)
      surfel.observable_projector_R += surfel.geometry_probability[h] *
                                       surfel.hypothesis_projector_R[static_cast<size_t>(h)];
    // A failed physical hypothesis does not delete geometry from the
    // reference/occupancy map.  It remains an explicit OTHER carrier with no
    // deformation coordinate.
    snapshot.surfels.push_back(surfel);
    // max_tiles is also the hard carrier budget for the local spatial LOD:
    // every accepted reference surfel produces one support sample and one
    // continuous-field sample.  Fail the reference transaction instead of
    // truncating support and silently changing the field basis.
    if (snapshot.surfels.size() > config_.resource.max_tiles) {
      reference_resource_failure("surfel_tiles", snapshot.surfels.size(),
                                 config_.resource.max_tiles);
      snapshot.surfels.clear();
      return snapshot;
    }
  }
  if (config_.resource.bounded_reference_search) {
    const size_t enough_exposures = static_cast<size_t>(std::count_if(
        cells.begin(), cells.end(), [&](const auto& cell) {
          return cell.second.size() >= static_cast<size_t>(
              config_.resource.reference_min_surfel_exposures);
        }));
    const size_t geometric_other = static_cast<size_t>(std::count_if(
        snapshot.surfels.begin(), snapshot.surfels.end(), [&](const Surfel& surfel) {
          return surfel.geometry_probability.head<3>().sum() <= config_.numerical.probability_floor;
        }));
    std::cerr << "[deform_v1] reference coverage: observed_cells=" << cells.size()
              << " insufficient_exposure_cells=" << cells.size() - enough_exposures
              << " surfels=" << snapshot.surfels.size()
              << " geometric_OTHER=" << geometric_other << std::endl;
  }
  std::sort(snapshot.surfels.begin(), snapshot.surfels.end(),
            [](const Surfel& lhs, const Surfel& rhs) { return lhs.id < rhs.id; });
  if (snapshot.surfels.empty()) {
    snapshot.reasons.push_back(ReasonCode::REFERENCE_INSUFFICIENT_EXPOSURE);
    snapshot.surfels.clear();
    return snapshot;
  }

  // Candidate edges require mutual membership in both deterministically
  // truncated neighbourhoods.  A one-sided top-k list is density-biased and
  // does not satisfy the frozen undirected graph definition.
  // Surfels are already seeded on a fixed spatial lattice.  Enumerate only
  // lattice cells intersecting the registered neighbour radius; an all-pairs
  // scan is mathematically identical after the distance test but becomes
  // quadratic as soon as valid planar carriers are retained.
  std::map<VoxelKey, std::vector<uint32_t>> surfels_by_cell;
  for (uint32_t i = 0; i < snapshot.surfels.size(); ++i) {
    VoxelKey key;
    if (!TryGridKey(snapshot.surfels[i].center_R,
                    config_.resource.grid_origin_m,
                    config_.resource.surfel_seed_spacing_m, &key)) {
      reference_resource_failure("surfel_grid_domain", 1U, 0U);
      snapshot.surfels.clear();
      return snapshot;
    }
    surfels_by_cell[key].push_back(i);
  }
  const int64_t edge_radius_cells = static_cast<int64_t>(std::ceil(
      config_.resource.graph_neighbor_radius_m /
      config_.resource.surfel_seed_spacing_m));
  std::vector<std::set<uint32_t>> truncated_neighbors(snapshot.surfels.size());
  for (uint32_t i = 0; i < snapshot.surfels.size(); ++i) {
    std::vector<std::pair<double, uint32_t>> neighbor;
    VoxelKey center_key;
    if (!TryGridKey(snapshot.surfels[i].center_R,
                    config_.resource.grid_origin_m,
                    config_.resource.surfel_seed_spacing_m, &center_key)) {
      reference_resource_failure("neighbor_grid_domain", 1U, 0U);
      snapshot.surfels.clear();
      return snapshot;
    }
    for (int64_t dx = -edge_radius_cells; dx <= edge_radius_cells; ++dx)
      for (int64_t dy = -edge_radius_cells; dy <= edge_radius_cells; ++dy)
        for (int64_t dz = -edge_radius_cells; dz <= edge_radius_cells; ++dz) {
          const int64_t x = static_cast<int64_t>(center_key.x) + dx;
          const int64_t y = static_cast<int64_t>(center_key.y) + dy;
          const int64_t z = static_cast<int64_t>(center_key.z) + dz;
          if (x < std::numeric_limits<int32_t>::min() ||
              x > std::numeric_limits<int32_t>::max() ||
              y < std::numeric_limits<int32_t>::min() ||
              y > std::numeric_limits<int32_t>::max() ||
              z < std::numeric_limits<int32_t>::min() ||
              z > std::numeric_limits<int32_t>::max())
            continue;
          const auto cell = surfels_by_cell.find(
              VoxelKey{static_cast<int32_t>(x), static_cast<int32_t>(y),
                       static_cast<int32_t>(z)});
          if (cell == surfels_by_cell.end()) continue;
          for (uint32_t j : cell->second) {
            if (i == j) continue;
            const double distance =
                (snapshot.surfels[i].center_R -
                 snapshot.surfels[j].center_R).norm();
            if (distance <= config_.resource.graph_neighbor_radius_m)
              neighbor.emplace_back(distance, j);
          }
        }
    std::sort(neighbor.begin(), neighbor.end(), [&](const std::pair<double, uint32_t>& lhs,
                                                     const std::pair<double, uint32_t>& rhs) {
      if (std::abs(lhs.first - rhs.first) > config_.numerical.tie_tolerance)
        return lhs.first < rhs.first;
      return snapshot.surfels[lhs.second].id < snapshot.surfels[rhs.second].id;
    });
    if (neighbor.size() > static_cast<size_t>(config_.resource.graph_max_neighbors))
      neighbor.resize(static_cast<size_t>(config_.resource.graph_max_neighbors));
    for (const auto& candidate : neighbor)
      truncated_neighbors[i].insert(candidate.second);
  }
  for (uint32_t i = 0; i < snapshot.surfels.size(); ++i) {
    for (uint32_t j : truncated_neighbors[i]) {
      if (j <= i || truncated_neighbors[j].count(i) == 0U) continue;
      const double distance =
          (snapshot.surfels[i].center_R - snapshot.surfels[j].center_R).norm();
      const double chordal = (snapshot.surfels[i].observable_projector_R -
                              snapshot.surfels[j].observable_projector_R).norm();
      const double normalized_distance = distance /
                                         config_.resource.graph_neighbor_radius_m;
      const double logit = config_.calibration.reference_link_logit_bias +
          config_.calibration.reference_link_distance_weight * normalized_distance +
          config_.calibration.reference_link_projector_weight * chordal;
      const bool first_physical =
          snapshot.surfels[i].geometry_probability.head<3>().sum() >
          config_.numerical.probability_floor;
      const bool second_physical =
          snapshot.surfels[j].geometry_probability.head<3>().sum() >
          config_.numerical.probability_floor;
      // OTHER is an occupancy/reference carrier, not a bridge through which
      // a physical deformation basis may propagate.  Keep OTHER--OTHER graph
      // connectivity for map support, but cut every mixed physical/OTHER
      // edge deterministically.
      const bool physical_other_boundary = first_physical != second_physical;
      const double p_link = physical_other_boundary
          ? 0.0 : 1.0 / (1.0 + std::exp(-logit));
      GraphEdge edge;
      edge.id = EdgeId(snapshot.surfels[i].id, snapshot.surfels[j].id);
      edge.first = i; edge.second = j; edge.distance_m = distance; edge.p_link = p_link;
      edge.state = p_link >= config_.calibration.link_probability_on
                       ? ReferenceEdgeState::LINK
                       : (p_link <= config_.calibration.cut_probability_on
                              ? ReferenceEdgeState::CUT : ReferenceEdgeState::AMBIGUOUS);
      snapshot.edges.push_back(edge);
    }
  }
  std::sort(snapshot.edges.begin(), snapshot.edges.end(),
            [](const GraphEdge& lhs, const GraphEdge& rhs) { return lhs.id < rhs.id; });

  // A graph component may be an entire wall and cannot be represented by the
  // frozen 6--16 local controls.  Split it on a world-fixed lattice whose
  // width equals two PoU support radii, then recompute connectivity using
  // only LINK edges that remain inside that tile.  Keeping the pre-split root
  // would admit subgraphs connected only through a path outside the tile and
  // produce infinite control-to-surfel graph distances.
  const double gdu_tile_width_m = 2.0 * config_.resource.pou_support_radius_m;
  std::vector<VoxelKey> tile_by_surfel(snapshot.surfels.size());
  for (uint32_t i = 0; i < snapshot.surfels.size(); ++i) {
    if (!TryGridKey(snapshot.surfels[i].center_R,
                    config_.resource.grid_origin_m, gdu_tile_width_m,
                    &tile_by_surfel[i])) {
      reference_resource_failure("gdu_tile_grid_domain", 1U, 0U);
      snapshot.surfels.clear();
      return snapshot;
    }
  }
  DisjointSet components(snapshot.surfels.size());
  for (const GraphEdge& edge : snapshot.edges)
    if (edge.state == ReferenceEdgeState::LINK &&
        tile_by_surfel[edge.first] == tile_by_surfel[edge.second])
      components.Unite(edge.first, edge.second);
  std::map<uint32_t, std::vector<uint32_t>> members;
  for (uint32_t i = 0; i < snapshot.surfels.size(); ++i)
    members[components.Find(i)].push_back(i);
  std::vector<GduDefinition> component_candidates;
  component_candidates.reserve(members.size());
  for (const auto& component : members) {
    GduDefinition gdu;
    gdu.surfel_indices = component.second;
    std::sort(gdu.surfel_indices.begin(), gdu.surfel_indices.end(), [&](uint32_t lhs, uint32_t rhs) {
      return snapshot.surfels[lhs].id < snapshot.surfels[rhs].id;
    });
    CanonicalBytes fields;
    for (uint32_t index : gdu.surfel_indices) fields.AddId(snapshot.surfels[index].id);
    gdu.id = MakeStableId("gdu-v1", fields);
    component_candidates.push_back(std::move(gdu));
  }
  std::sort(component_candidates.begin(), component_candidates.end(),
            [](const GduDefinition& lhs, const GduDefinition& rhs) {
              if (lhs.surfel_indices.size() != rhs.surfel_indices.size())
                return lhs.surfel_indices.size() > rhs.surfel_indices.size();
              return lhs.id < rhs.id;
            });
  const auto deformation_eligible = [&](const GduDefinition& candidate) {
    if (candidate.surfel_indices.size() <
        config_.resource.min_controls_per_gdu)
      return false;
    return std::all_of(
        candidate.surfel_indices.begin(), candidate.surfel_indices.end(),
        [&](uint32_t surfel_index) {
          return surfel_index < snapshot.surfels.size() &&
              snapshot.surfels[surfel_index]
                      .geometry_probability.head<3>().sum() >
                  config_.numerical.probability_floor;
        });
  };
  const size_t control_eligible_components = static_cast<size_t>(std::count_if(
      component_candidates.begin(), component_candidates.end(),
      deformation_eligible));
  const bool needs_other_carrier =
      control_eligible_components < component_candidates.size() ||
      control_eligible_components > config_.resource.max_gdu;
  const size_t active_capacity = config_.resource.max_gdu -
      (needs_other_carrier ? 1U : 0U);
  const size_t retained_active = std::min(control_eligible_components,
                                           active_capacity);
  Eigen::Vector3d frozen_sensor_origin_R = Eigen::Vector3d::Zero();
  for (const FrameBundle& frame : frames) {
    if (!frame.pose.origin_L_R.allFinite()) {
      snapshot.reasons.push_back(ReasonCode::INPUT_SCHEMA_INVALID);
      snapshot.gdus.clear();
      return snapshot;
    }
    frozen_sensor_origin_R += frame.pose.origin_L_R;
  }
  frozen_sensor_origin_R /= static_cast<double>(frames.size());
  std::vector<size_t> eligible_candidate_indices;
  std::vector<GduDefinition, Eigen::aligned_allocator<GduDefinition>>
      eligible_candidates;
  eligible_candidate_indices.reserve(control_eligible_components);
  eligible_candidates.reserve(control_eligible_components);
  for (size_t index = 0U; index < component_candidates.size(); ++index) {
    if (!deformation_eligible(component_candidates[index])) {
      continue;
    }
    eligible_candidate_indices.push_back(index);
    eligible_candidates.push_back(component_candidates[index]);
  }
  const std::vector<size_t> retained_eligible_indices =
      SelectReferenceGduCoverage(
          snapshot, eligible_candidates, frozen_sensor_origin_R,
          retained_active, gdu_tile_width_m,
          config_.numerical.tie_tolerance);
  if (retained_eligible_indices.size() != retained_active) {
    snapshot.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
    snapshot.gdus.clear();
    return snapshot;
  }
  std::set<size_t> retained_candidate_indices;
  for (size_t eligible_index : retained_eligible_indices) {
    if (eligible_index >= eligible_candidate_indices.size()) {
      snapshot.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
      snapshot.gdus.clear();
      return snapshot;
    }
    retained_candidate_indices.insert(
        eligible_candidate_indices[eligible_index]);
  }
  std::set<StableId> deformation_candidate_ids;
  for (size_t index : retained_candidate_indices) {
    deformation_candidate_ids.insert(component_candidates[index].id);
    snapshot.gdus.push_back(component_candidates[index]);
  }
  if (needs_other_carrier) {
    GduDefinition other;
    for (size_t i = 0U; i < component_candidates.size(); ++i) {
      if (retained_candidate_indices.count(i) != 0U) continue;
      other.surfel_indices.insert(other.surfel_indices.end(),
          component_candidates[i].surfel_indices.begin(),
          component_candidates[i].surfel_indices.end());
    }
    std::sort(other.surfel_indices.begin(), other.surfel_indices.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                return snapshot.surfels[lhs].id < snapshot.surfels[rhs].id;
              });
    CanonicalBytes fields;
    for (uint32_t index : other.surfel_indices)
      fields.AddId(snapshot.surfels[index].id);
    other.id = MakeStableId("gdu-other-carrier-v1", fields);
    // Deliberately absent from deformation_candidate_ids: disconnected small
    // fragments remain represented in occupancy but cannot share one
    // fabricated deformation coordinate system.
    snapshot.gdus.push_back(std::move(other));
  }
  std::sort(snapshot.gdus.begin(), snapshot.gdus.end(),
            [](const GduDefinition& lhs, const GduDefinition& rhs) { return lhs.id < rhs.id; });
  for (uint32_t g = 0; g < snapshot.gdus.size(); ++g)
    for (uint32_t index : snapshot.gdus[g].surfel_indices) snapshot.surfels[index].gdu_index = g;
  if (snapshot.gdus.size() > config_.resource.max_gdu) {
    std::cerr << "[deform_v1] reference reject components="
              << snapshot.gdus.size() << " limit="
              << config_.resource.max_gdu << std::endl;
    reference_resource_failure("gdu_count", snapshot.gdus.size(),
                               config_.resource.max_gdu);
    snapshot.gdus.clear(); return snapshot;
  }

  // Freeze one minimal physical coordinate frame per GDU using reference-only
  // geometry.  The current residual never chooses a more permissive rank.
  for (GduDefinition& gdu : snapshot.gdus) {
    if (deformation_candidate_ids.count(gdu.id) == 0U) continue;
    Eigen::Vector4d probability = Eigen::Vector4d::Zero();
    std::array<Eigen::Matrix3d, 3> projector{{
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
        Eigen::Matrix3d::Zero()}};
    for (uint32_t surfel_index : gdu.surfel_indices) {
      const Surfel& surfel = snapshot.surfels[surfel_index];
      probability += surfel.geometry_probability;
      for (int h = 0; h < 3; ++h)
        projector[static_cast<size_t>(h)] +=
            surfel.geometry_probability[h] *
            surfel.hypothesis_projector_R[static_cast<size_t>(h)];
    }
    if (!gdu.surfel_indices.empty())
      probability /= static_cast<double>(gdu.surfel_indices.size());
    Eigen::Index selected = 3;
    probability.maxCoeff(&selected);
    const int rank = selected == 0 ? 1 : (selected == 1 ? 2 :
        (selected == 2 ? 3 : 0));
    if (rank == 0) continue;
    const Eigen::Matrix3d frozen_projector = StableNearestProjector(
        projector[static_cast<size_t>(selected)] /
            std::max(config_.numerical.probability_floor,
                     probability[static_cast<int>(selected)] *
                         static_cast<double>(gdu.surfel_indices.size())),
        rank, config_.calibration.geometry_min_eigengap_ratio);
    bool axes_valid = false;
    gdu.coordinate_axes_R = AxesFromProjector(
        frozen_projector, rank,
        config_.calibration.geometry_min_eigengap_ratio, &axes_valid);
    if (!axes_valid) continue;
    gdu.coordinate_rank = rank;
    gdu.coordinate_geometry = static_cast<GeometryHypothesis>(selected);
    gdu.monitorable = true;
  }

  // Estimate one epoch-static common reference-bias covariance per GDU from
  // frame-level residuals.  Averaging already-averaged surfel covariances as
  // though cells were independent would erase the common SLAM/reference
  // component and make broad, coherent motion spuriously overconfident.
  for (GduDefinition& gdu : snapshot.gdus) {
    std::map<StableId, std::pair<Eigen::Vector3d, size_t>> residual_by_frame;
    Eigen::Matrix3d independent_fallback = Eigen::Matrix3d::Zero();
    for (uint32_t surfel_index : gdu.surfel_indices) {
      const Surfel& surfel = snapshot.surfels[surfel_index];
      independent_fallback += surfel.covariance_R;
      VoxelKey key;
      if (!TryGridKey(surfel.center_R, config_.resource.grid_origin_m,
                      config_.resource.surfel_seed_spacing_m, &key)) {
        reference_resource_failure("reference_bias_grid_domain", 1U, 0U);
        snapshot.gdus.clear();
        return snapshot;
      }
      const auto cell = cells.find(key);
      if (cell == cells.end()) continue;
      for (const auto& exposure : cell->second) {
        Eigen::Vector3d exposure_center = Eigen::Vector3d::Zero();
        for (const Eigen::Vector3d& point : exposure.second.points)
          exposure_center += point;
        exposure_center /= static_cast<double>(exposure.second.points.size());
        auto inserted = residual_by_frame.emplace(
            exposure.first,
            std::make_pair(Eigen::Vector3d::Zero(), size_t{0U}));
        inserted.first->second.first += exposure_center - surfel.center_R;
        ++inserted.first->second.second;
      }
    }
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        common_residuals;
    for (const auto& residual : residual_by_frame) {
      if (residual.second.second == 0U) continue;
      common_residuals.push_back(
          residual.second.first / static_cast<double>(residual.second.second));
    }
    if (common_residuals.size() >= 3U) {
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (const Eigen::Vector3d& residual : common_residuals) mean += residual;
      mean /= static_cast<double>(common_residuals.size());
      Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
      for (const Eigen::Vector3d& residual : common_residuals) {
        const Eigen::Vector3d delta = residual - mean;
        scatter += delta * delta.transpose();
      }
      scatter /= static_cast<double>(common_residuals.size() - 1U);
      // The across-frame common residual is an epoch-static nuisance, not an
      // average of independent samples.  Its covariance therefore does not
      // shrink with the number of exposures.
      gdu.reference_bias_covariance_R = scatter;
    } else {
      gdu.reference_bias_covariance_R = independent_fallback;
    }
    const double common_floor =
        config_.calibration.reference_epoch_common_bias_sigma_m *
        config_.calibration.reference_epoch_common_bias_sigma_m;
    gdu.reference_bias_covariance_R.diagonal().array() += common_floor;
    gdu.reference_bias_covariance_R = 0.5 *
        (gdu.reference_bias_covariance_R +
         gdu.reference_bias_covariance_R.transpose());
    if (!ValidateSymmetricPsd(
            gdu.reference_bias_covariance_R,
            config_.numerical.psd_eigenvalue_tolerance)) {
      snapshot.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
      snapshot.gdus.clear();
      return snapshot;
    }
  }

  // Deterministic geodesic farthest-point control selection.
  for (uint32_t g = 0; g < snapshot.gdus.size(); ++g) {
    GduDefinition& gdu = snapshot.gdus[g];
    if (!gdu.monitorable) continue;
    bool maximum_exceeded = false;
    gdu.control_surfel_indices = GeodesicSeeds(
        snapshot, g, config_.resource.control_spacing_m,
        config_.resource.max_controls_per_gdu,
        config_.numerical.tie_tolerance, &maximum_exceeded);
    if (maximum_exceeded) {
      reference_resource_failure("gdu_control_count",
                                 config_.resource.max_controls_per_gdu + 1U,
                                 config_.resource.max_controls_per_gdu);
      snapshot.gdus.clear();
      return snapshot;
    }
    std::set<uint32_t> selected(gdu.control_surfel_indices.begin(),
                                gdu.control_surfel_indices.end());
    for (uint32_t surfel : gdu.surfel_indices) {
      if (gdu.control_surfel_indices.size() >=
          config_.resource.min_controls_per_gdu) break;
      if (selected.insert(surfel).second)
        gdu.control_surfel_indices.push_back(surfel);
    }
    if (gdu.control_surfel_indices.size() <
        config_.resource.min_controls_per_gdu) {
      gdu.control_surfel_indices.clear();
      gdu.coordinate_axes_R.setZero();
      gdu.coordinate_rank = 0;
      gdu.coordinate_geometry = GeometryHypothesis::OTHER;
      gdu.monitorable = false;
    }
  }

  // Intra-GDU interface library: each registered scale creates a deterministic
  // geodesic Voronoi partition; connected boundary-edge components become
  // candidate discontinuities before any current residual is observed.
  for (uint32_t g = 0; g < snapshot.gdus.size(); ++g) {
    const GduDefinition& gdu = snapshot.gdus[g];
    if (!gdu.monitorable) continue;
    for (double scale : config_.resource.interface_scales_m) {
      bool maximum_exceeded = false;
      const std::vector<uint32_t> seeds = GeodesicSeeds(
          snapshot, g, scale, config_.resource.max_controls_per_gdu,
          config_.numerical.tie_tolerance, &maximum_exceeded);
      if (maximum_exceeded) {
        reference_resource_failure("interface_seed_count",
                                   config_.resource.max_controls_per_gdu + 1U,
                                   config_.resource.max_controls_per_gdu);
        snapshot.interfaces.clear();
        return snapshot;
      }
      if (seeds.size() < 2U) continue;
      std::vector<std::vector<double>> seed_distances;
      for (uint32_t seed : seeds)
        seed_distances.push_back(LinkGraphDistances(snapshot, g, seed));
      std::map<uint32_t, size_t> assignment;
      std::map<size_t, std::vector<uint32_t>> regions;
      for (uint32_t surfel : gdu.surfel_indices) {
        size_t best = 0U;
        for (size_t seed = 1U; seed < seeds.size(); ++seed) {
          if (seed_distances[seed][surfel] < seed_distances[best][surfel] -
                  config_.numerical.tie_tolerance ||
              (std::abs(seed_distances[seed][surfel] -
                        seed_distances[best][surfel]) <=
                   config_.numerical.tie_tolerance &&
               snapshot.surfels[seeds[seed]].id <
                   snapshot.surfels[seeds[best]].id)) best = seed;
        }
        assignment[surfel] = best;
        regions[best].push_back(surfel);
      }
      std::map<std::pair<size_t, size_t>, std::vector<uint32_t>> boundary_by_pair;
      for (uint32_t edge_index = 0; edge_index < snapshot.edges.size(); ++edge_index) {
        const GraphEdge& edge = snapshot.edges[edge_index];
        if (edge.state != ReferenceEdgeState::LINK ||
            snapshot.surfels[edge.first].gdu_index != g ||
            snapshot.surfels[edge.second].gdu_index != g) continue;
        size_t first = assignment[edge.first], second = assignment[edge.second];
        if (first == second) continue;
        if (snapshot.surfels[seeds[second]].id <
            snapshot.surfels[seeds[first]].id) std::swap(first, second);
        boundary_by_pair[std::make_pair(first, second)].push_back(edge_index);
      }
      for (const auto& pair_edges : boundary_by_pair) {
        for (const auto& component :
             ConnectedBoundaryComponents(snapshot, pair_edges.second)) {
          InterfaceDefinition interface;
          interface.cross_gdu = false;
          interface.center_R.setZero();
          interface.normal_R.setZero();
          interface.left_gdu = g;
          interface.right_gdu = g;
          const size_t left_seed = pair_edges.first.first;
          const size_t right_seed = pair_edges.first.second;
          interface.left_region_surfels = regions[left_seed];
          interface.right_region_surfels = regions[right_seed];
          std::sort(interface.left_region_surfels.begin(),
                    interface.left_region_surfels.end());
          std::sort(interface.right_region_surfels.begin(),
                    interface.right_region_surfels.end());
          Eigen::Vector3d left_center = Eigen::Vector3d::Zero();
          Eigen::Vector3d right_center = Eigen::Vector3d::Zero();
          size_t left_count = 0U, right_count = 0U;
          interface.left_surfel = seeds[left_seed];
          interface.right_surfel = seeds[right_seed];
          for (uint32_t edge_index : component) {
            const GraphEdge& edge = snapshot.edges[edge_index];
            interface.boundary_edge_ids.push_back(edge.id);
            interface.center_R += 0.5 *
                (snapshot.surfels[edge.first].center_R +
                 snapshot.surfels[edge.second].center_R);
            for (uint32_t surfel : {edge.first, edge.second}) {
              if (assignment[surfel] == left_seed) {
                left_center += snapshot.surfels[surfel].center_R;
                ++left_count;
              } else {
                right_center += snapshot.surfels[surfel].center_R;
                ++right_count;
              }
            }
          }
          interface.center_R /= static_cast<double>(component.size());
          left_center /= static_cast<double>(std::max<size_t>(1U, left_count));
          right_center /= static_cast<double>(std::max<size_t>(1U, right_count));
          interface.normal_R = right_center - left_center;
          if (interface.normal_R.norm() <= config_.numerical.tie_tolerance)
            interface.normal_R =
                snapshot.surfels[interface.right_surfel].center_R -
                snapshot.surfels[interface.left_surfel].center_R;
          if (interface.normal_R.norm() > config_.numerical.tie_tolerance)
            interface.normal_R.normalize();
          else
            interface.normal_R = Eigen::Vector3d::UnitX();
          interface.scale_m = scale;
          interface.p_link = 1.0;
          CanonicalBytes competition_fields;
          competition_fields.AddId(gdu.id);
          interface.competition_group_id = MakeStableId(
              "intra-gdu-interface-competition-v1", competition_fields);
          CanonicalBytes fields;
          fields.AddId(gdu.id);
          fields.AddId(snapshot.surfels[seeds[left_seed]].id);
          fields.AddId(snapshot.surfels[seeds[right_seed]].id);
          fields.AddDouble(scale);
          for (const StableId& edge_id : interface.boundary_edge_ids)
            fields.AddId(edge_id);
          interface.id = MakeStableId("intra-gdu-interface-v1", fields);
          snapshot.interfaces.push_back(interface);
        }
      }
    }
  }

  // V1.1 does not add cross-GDU jump coordinates.  Independent continuous
  // GDU fields already explain relative cross-boundary motion; a second jump
  // variable would be algebraically redundant without an explicit coupling
  // constraint.  Ambiguous cross-GDU boundaries remain protective OTHER
  // geometry in occupancy/risk.

  // Retain at most one deterministic intra-GDU candidate.  Prefer the
  // candidate supported by most frozen boundary edges, then the stable id.
  std::map<uint32_t, InterfaceDefinition> selected_interface;
  for (const InterfaceDefinition& candidate : snapshot.interfaces) {
    if (candidate.cross_gdu ||
        candidate.left_gdu >= snapshot.gdus.size() ||
        !snapshot.gdus[candidate.left_gdu].monitorable) continue;
    auto inserted = selected_interface.emplace(candidate.left_gdu, candidate);
    InterfaceDefinition& retained = inserted.first->second;
    if (!inserted.second &&
        (candidate.boundary_edge_ids.size() > retained.boundary_edge_ids.size() ||
         (candidate.boundary_edge_ids.size() == retained.boundary_edge_ids.size() &&
          candidate.id < retained.id)))
      retained = candidate;
  }
  snapshot.interfaces.clear();
  for (const auto& selected : selected_interface)
    snapshot.interfaces.push_back(selected.second);
  std::sort(snapshot.interfaces.begin(), snapshot.interfaces.end(),
            [](const InterfaceDefinition& lhs, const InterfaceDefinition& rhs) {
              return lhs.id < rhs.id;
            });
  if (snapshot.interfaces.size() > config_.resource.max_interfaces) {
    reference_resource_failure("interface_count", snapshot.interfaces.size(),
                               config_.resource.max_interfaces);
    snapshot.interfaces.clear(); return snapshot;
  }
  for (InterfaceDefinition& interface : snapshot.interfaces) {
    const GduDefinition& gdu = snapshot.gdus[interface.left_gdu];
    const int rows = static_cast<int>(gdu.surfel_indices.size());
    const int controls = static_cast<int>(gdu.control_surfel_indices.size());
    Eigen::MatrixXd base = Eigen::MatrixXd::Zero(rows, controls);
    Eigen::VectorXd jump = Eigen::VectorXd::Zero(rows);
    for (int row = 0; row < rows; ++row) {
      const uint32_t surfel_index = gdu.surfel_indices[static_cast<size_t>(row)];
      double denominator = 0.0;
      for (int column = 0; column < controls; ++column) {
        const std::vector<double> distances = LinkGraphDistances(
            snapshot, interface.left_gdu,
            gdu.control_surfel_indices[static_cast<size_t>(column)]);
        base(row, column) = WendlandC2(
            distances[surfel_index] / config_.resource.pou_support_radius_m);
        denominator += base(row, column);
      }
      if (denominator > config_.numerical.pou_denominator_min)
        base.row(row) /= denominator;
      const Eigen::Vector3d& position = snapshot.surfels[surfel_index].center_R;
      const int side = InterfaceSideForPoint(
          snapshot, interface, interface.left_gdu, position,
          config_.numerical.tie_tolerance);
      jump[row] = 0.5 * static_cast<double>(side) * WendlandC2(
          (position - interface.center_R).norm() / interface.scale_m);
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> base_qr(base);
    base_qr.setThreshold(config_.calibration.geometry_min_eigengap_ratio);
    Eigen::MatrixXd augmented(rows, controls + 1);
    augmented.leftCols(controls) = base;
    augmented.rightCols(1) = jump;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> augmented_qr(augmented);
    augmented_qr.setThreshold(config_.calibration.geometry_min_eigengap_ratio);
    interface.identifiable = jump.norm() > config_.numerical.tie_tolerance &&
        augmented_qr.rank() == base_qr.rank() + 1;
    if (!interface.identifiable)
      snapshot.reasons.push_back(ReasonCode::INTERFACE_UNIDENTIFIABLE);
  }
  // Give each registered scale equal group mass and split that scale's mass
  // equally over its reference-only position candidates.  Thus adding a
  // scale or refining the candidate tessellation changes resolution, not the
  // total probability of a discontinuity opportunity.
  std::map<StableId, std::map<double, size_t>> candidates_per_scale;
  for (const InterfaceDefinition& interface : snapshot.interfaces)
    ++candidates_per_scale[interface.competition_group_id][interface.scale_m];
  for (InterfaceDefinition& interface : snapshot.interfaces) {
    const auto group = candidates_per_scale.find(interface.competition_group_id);
    if (group == candidates_per_scale.end() || group->second.empty() ||
        group->second.find(interface.scale_m) == group->second.end()) {
      snapshot.reasons.push_back(ReasonCode::INVALID_NUMERICAL);
      snapshot.interfaces.clear();
      return snapshot;
    }
    interface.competition_weight =
        1.0 / (static_cast<double>(group->second.size()) *
               static_cast<double>(group->second.at(interface.scale_m)));
  }
  for (uint32_t i = 0; i < snapshot.interfaces.size(); ++i) {
    snapshot.gdus[snapshot.interfaces[i].left_gdu].interface_indices.push_back(i);
    if (snapshot.interfaces[i].right_gdu != snapshot.interfaces[i].left_gdu)
      snapshot.gdus[snapshot.interfaces[i].right_gdu].interface_indices.push_back(i);
  }
  CanonicalBytes snapshot_fields;
  snapshot_fields.AddString(config_.resource.session_id);
  snapshot_fields.AddU32(reference_epoch);
  for (const FrameBundle& frame : frames) snapshot_fields.AddId(frame.frame_id);
  snapshot.id = MakeStableId("reference-snapshot-v1", snapshot_fields);
  if (!StableIdRegistryHealthy()) {
    if (StableIdRegistryResourceExhausted())
      reference_resource_failure("stable_id_registry", StableIdRegistrySize(),
                                 config_.resource.max_stable_ids);
    else
      snapshot.reasons.push_back(ReasonCode::INVALID_ID_COLLISION);
    snapshot.validity = Validity::INVALID;
    snapshot.frozen = false;
    return snapshot;
  }
  snapshot.frozen = true;
  snapshot.validity = CombineValidity(
      config_.calibration.validated ? Validity::VALID
                                    : Validity::INCONCLUSIVE,
      input_validity);
  if (!config_.calibration.validated) snapshot.reasons.push_back(ReasonCode::CALIBRATION_UNSET);
  std::sort(snapshot.reasons.begin(), snapshot.reasons.end(),
            [](ReasonCode lhs, ReasonCode rhs) {
              return static_cast<uint16_t>(lhs) <
                     static_cast<uint16_t>(rhs);
            });
  snapshot.reasons.erase(
      std::unique(snapshot.reasons.begin(), snapshot.reasons.end()),
      snapshot.reasons.end());
  return snapshot;
}

std::vector<double> PouField::GraphDistances(uint32_t gdu_index,
                                             uint32_t source_surfel) const {
  const std::pair<uint32_t, uint32_t> key(gdu_index, source_surfel);
  const auto found = graph_distance_cache_.find(key);
  if (found != graph_distance_cache_.end()) return found->second;
  const std::vector<double> distances =
      LinkGraphDistances(*reference_, gdu_index, source_surfel);
  graph_distance_cache_[key] = distances;
  return distances;
}

int PouField::CoefficientCount(uint32_t gdu_index) const {
  if (!reference_ || gdu_index >= reference_->gdus.size()) return 0;
  const GduDefinition& gdu = reference_->gdus[gdu_index];
  if (!gdu.monitorable || gdu.coordinate_rank <= 0) return 0;
  return gdu.coordinate_rank * static_cast<int>(
      gdu.control_surfel_indices.size() + gdu.interface_indices.size());
}

Eigen::MatrixXd PouField::VectorBasis(uint32_t gdu_index,
                                      const Eigen::Vector3d& position_R,
                                      Validity* validity,
                                      std::vector<ReasonCode>* reasons,
                                      double* attachment_distance_m) const {
  if (validity) *validity = Validity::UNKNOWN;
  if (attachment_distance_m)
    *attachment_distance_m = std::numeric_limits<double>::infinity();
  if (!reference_ || !config_ || !reference_->frozen || gdu_index >= reference_->gdus.size()) {
    if (reasons) reasons->push_back(ReasonCode::REFERENCE_NOT_FROZEN);
    return Eigen::MatrixXd();
  }
  const GduDefinition& gdu = reference_->gdus[gdu_index];
  if (!gdu.monitorable || gdu.coordinate_rank <= 0 ||
      gdu.surfel_indices.empty() || gdu.control_surfel_indices.empty()) {
    if (reasons) reasons->push_back(ReasonCode::OUTSIDE_SUPPORT);
    return Eigen::MatrixXd();
  }

  // A 3-D query is first attached to the frozen LINK metric graph.  Along a
  // LINK edge (a,b), d_G(x,c)=min(t L+d_G(a,c),(1-t)L+d_G(b,c)); this agrees
  // at vertices and is continuous on the metric graph.  A nearest-vertex
  // substitution would jump at every Voronoi boundary and is therefore not a
  // valid implementation of the continuous field specified by DEC-006.
  uint32_t attached_vertex = gdu.surfel_indices.front();
  double nearest_vertex_distance = std::numeric_limits<double>::infinity();
  for (uint32_t index : gdu.surfel_indices) {
    const double distance = (position_R - reference_->surfels[index].center_R).norm();
    if (distance < nearest_vertex_distance - config_->numerical.tie_tolerance ||
        (std::abs(distance - nearest_vertex_distance) <=
             config_->numerical.tie_tolerance &&
         reference_->surfels[index].id <
             reference_->surfels[attached_vertex].id)) {
      nearest_vertex_distance = distance;
      attached_vertex = index;
    }
  }

  const GraphEdge* attached_edge = nullptr;
  double attached_fraction = 0.0;
  double attachment_distance = std::numeric_limits<double>::infinity();
  Eigen::Vector3d attached_point = Eigen::Vector3d::Zero();
  bool ambiguous_attachment = false;
  for (const GraphEdge& edge : reference_->edges) {
    if (edge.state != ReferenceEdgeState::LINK ||
        edge.first >= reference_->surfels.size() ||
        edge.second >= reference_->surfels.size() ||
        reference_->surfels[edge.first].gdu_index != gdu_index ||
        reference_->surfels[edge.second].gdu_index != gdu_index) continue;
    const Eigen::Vector3d first = reference_->surfels[edge.first].center_R;
    const Eigen::Vector3d delta =
        reference_->surfels[edge.second].center_R - first;
    const double squared_length = delta.squaredNorm();
    if (!(squared_length > 0.0)) continue;
    const double fraction = std::max(0.0, std::min(
        1.0, (position_R - first).dot(delta) / squared_length));
    const Eigen::Vector3d projection = first + fraction * delta;
    const double distance = (position_R - projection).norm();
    if (!attached_edge ||
        distance < attachment_distance - config_->numerical.tie_tolerance) {
      attached_edge = &edge;
      attached_fraction = fraction;
      attachment_distance = distance;
      attached_point = projection;
      ambiguous_attachment = false;
    } else if (std::abs(distance - attachment_distance) <=
                   config_->numerical.tie_tolerance) {
      if ((projection - attached_point).norm() >
          config_->numerical.tie_tolerance) {
        ambiguous_attachment = true;
      } else if (edge.id < attached_edge->id) {
        attached_edge = &edge;
        attached_fraction = fraction;
        attachment_distance = distance;
        attached_point = projection;
      }
    }
  }
  if (!attached_edge) attachment_distance = nearest_vertex_distance;
  if (ambiguous_attachment ||
      !(attachment_distance <= config_->resource.surfel_seed_spacing_m)) {
    if (reasons) reasons->push_back(ReasonCode::OUTSIDE_SUPPORT);
    return Eigen::MatrixXd();
  }
  if (attachment_distance_m) *attachment_distance_m = attachment_distance;

  std::vector<double> weight;
  weight.reserve(gdu.control_surfel_indices.size());
  for (uint32_t control : gdu.control_surfel_indices) {
    const std::vector<double> graph_distance =
        GraphDistances(gdu_index, control);
    double distance = graph_distance[attached_vertex];
    if (attached_edge) {
      const double length = (reference_->surfels[attached_edge->second].center_R -
                             reference_->surfels[attached_edge->first].center_R).norm();
      distance = std::min(
          attached_fraction * length + graph_distance[attached_edge->first],
          (1.0 - attached_fraction) * length +
              graph_distance[attached_edge->second]);
    }
    weight.push_back(WendlandC2(distance / config_->resource.pou_support_radius_m));
  }
  const double denominator = PairwiseSum(weight);
  if (!(denominator > config_->numerical.pou_denominator_min)) {
    if (reasons) reasons->push_back(ReasonCode::OUTSIDE_SUPPORT);
    return Eigen::MatrixXd();
  }
  const int coefficient_count = CoefficientCount(gdu_index);
  Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(3, coefficient_count);
  const int rank = gdu.coordinate_rank;
  const Eigen::MatrixXd axes = gdu.coordinate_axes_R.leftCols(rank);
  for (size_t i = 0; i < weight.size(); ++i)
    basis.block(0, static_cast<int>(i) * rank, 3, rank) =
        (weight[i] / denominator) * axes;
  for (size_t i = 0; i < gdu.interface_indices.size(); ++i) {
    const InterfaceDefinition& interface = reference_->interfaces[gdu.interface_indices[i]];
    const double taper = WendlandC2((position_R - interface.center_R).norm() / interface.scale_m);
    const int frozen_side = InterfaceSideForPoint(
        *reference_, interface, gdu_index, position_R,
        config_->numerical.tie_tolerance);
    const double side = 0.5 * static_cast<double>(frozen_side);
    const int column = rank * static_cast<int>(
        gdu.control_surfel_indices.size() + i);
    if (interface.identifiable)
      basis.block(0, column, 3, rank) = side * taper * axes;
  }
  if (validity) *validity = Validity::VALID;
  return basis;
}

Eigen::VectorXd PouField::ScalarBasis(uint32_t gdu_index,
                                      const Eigen::Vector3d& position_R,
                                      const Eigen::Vector3d& observation_axis_R,
                                      Validity* validity,
                                      std::vector<ReasonCode>* reasons,
                                      double* attachment_distance_m) const {
  if (!observation_axis_R.allFinite() || observation_axis_R.norm() <= 0.0) {
    if (validity) *validity = Validity::INVALID;
    if (reasons) reasons->push_back(ReasonCode::INPUT_SCHEMA_INVALID);
    return Eigen::VectorXd();
  }
  const Eigen::MatrixXd vector_basis = VectorBasis(
      gdu_index, position_R, validity, reasons, attachment_distance_m);
  if (vector_basis.size() == 0) return Eigen::VectorXd();
  return vector_basis.transpose() * observation_axis_R.normalized();
}

}  // namespace v1
}  // namespace deform_monitor_v3
