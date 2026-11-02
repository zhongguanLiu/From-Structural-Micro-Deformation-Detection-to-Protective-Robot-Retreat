#include "deform_monitor_v3/risk_map/hazard_projector.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
namespace deform_monitor_v3 { namespace risk_map {
void HazardProjector::Project(RiskMapVoxelVector* voxels) const {
  if (!voxels) return;
  std::unordered_map<VoxelKey, size_t, VoxelKeyHash> index;
  for (size_t i = 0; i < voxels->size(); ++i) index[(*voxels)[i].key] = i;
  const RiskMapVoxelVector evidence_voxels = *voxels;
  const int inflation = std::max(0, static_cast<int>(std::ceil(
      p_.robot_inflation_radius / std::max(1e-6, p_.voxel_size))));
  const int sweep_steps = std::max(0, static_cast<int>(std::ceil(
      p_.hazard_projection_length / std::max(1e-6, p_.voxel_size))));
  auto ensure_voxel = [&](const VoxelKey& key) -> RiskMapVoxel& {
    auto found = index.find(key);
    if (found != index.end()) return (*voxels)[found->second];
    RiskMapVoxel projected;
    projected.key = key;
    projected.center_R = p_.voxel_size * Eigen::Vector3d(
        key.x + 0.5, key.y + 0.5, key.z + 0.5);
    projected.knowledge_state = KnowledgeState::UNKNOWN;
    projected.semantics = EventSemantics::DEFORMATION;
    voxels->push_back(projected);
    index[key] = voxels->size() - 1;
    return voxels->back();
  };
  for (const auto& source : evidence_voxels) {
    const double deformation = std::isfinite(source.p_deformation)
                                   ? source.p_deformation : 0.0;
    const double novel = std::isfinite(source.p_novel_occupancy)
                             ? source.p_novel_occupancy : 0.0;
    if (deformation <= 0.0 && novel <= 0.0) continue;
    Eigen::Vector3d direction(0.0, 0.0, -1.0);
    if (source.principal_direction.allFinite() && source.direction_confidence >= 0.25F &&
        source.principal_direction.norm() > 1e-6F) {
      direction = source.principal_direction.cast<double>().normalized();
    }
    const int steps = deformation > 0.0 ? sweep_steps : 0;
    for (int step = 0; step <= steps; ++step) {
      const Eigen::Vector3d swept_center = source.center_R +
          direction * (step * p_.voxel_size);
      const VoxelKey swept{static_cast<int32_t>(std::floor(swept_center.x()/p_.voxel_size)),
                           static_cast<int32_t>(std::floor(swept_center.y()/p_.voxel_size)),
                           static_cast<int32_t>(std::floor(swept_center.z()/p_.voxel_size))};
      for (int dx=-inflation; dx<=inflation; ++dx) for (int dy=-inflation; dy<=inflation; ++dy)
        for (int dz=-inflation; dz<=inflation; ++dz) {
          if (dx*dx+dy*dy+dz*dz > inflation*inflation) continue;
          RiskMapVoxel& target = ensure_voxel(VoxelKey{swept.x+dx,swept.y+dy,swept.z+dz});
          const double kernel = 1.0 - 0.35 * static_cast<double>(step) /
                                         std::max(1, steps);
          const double projected_cost = kernel * (p_.decision_weight_deformation*deformation +
                                                   p_.decision_weight_novel*novel);
          target.decision_cost = std::max<double>(target.decision_cost,
                                                   std::min(1.0, projected_cost));
        }
    }
  }
  for (auto& cell : *voxels) {
    double unknown = 0.0;
    if (cell.knowledge_state == KnowledgeState::UNKNOWN) {
      unknown = p_.unknown_policy == UnknownPolicy::LETHAL_UNKNOWN
                    ? 1.0
                    : (p_.unknown_policy == UnknownPolicy::EXPLORABLE_UNKNOWN
                           ? p_.unknown_explorable_cost : p_.unknown_fixed_cost);
    }
    const double deformation = std::isfinite(cell.p_deformation) ? cell.p_deformation : 0.0;
    const double novel = std::isfinite(cell.p_novel_occupancy) ? cell.p_novel_occupancy : 0.0;
    const double local = p_.decision_weight_deformation*deformation +
                         p_.decision_weight_novel*novel + p_.decision_weight_unknown*unknown;
    cell.decision_cost = std::max<double>(cell.decision_cost,
        std::max(0.0, std::min(1.0, local)));
  }
}
} }
