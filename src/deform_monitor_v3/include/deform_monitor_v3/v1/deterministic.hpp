#ifndef DEFORM_MONITOR_V3_V1_DETERMINISTIC_HPP
#define DEFORM_MONITOR_V3_V1_DETERMINISTIC_HPP

#include "deform_monitor_v3/v1/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace deform_monitor_v3 {
namespace v1 {

using Sha256Digest = std::array<uint8_t, 32>;

Sha256Digest Sha256(const std::vector<uint8_t>& bytes);
std::string HexDigest(const Sha256Digest& digest);
std::string HexId(const StableId& id);
bool ParseHexId(const std::string& text, StableId* id);
StableId TruncatedId(const Sha256Digest& digest);
void ResetStableIdRegistry();
bool ConfigureStableIdRegistryCapacity(size_t maximum_entries);
bool RegisterStableIdDigest(const StableId& id, const Sha256Digest& digest);
bool StableIdRegistryHealthy();
bool StableIdRegistryResourceExhausted();
uint64_t StableIdCollisionCount();
size_t StableIdRegistrySize();
// Conservative transaction preflight.  Existing duplicate digests may make
// the true cost smaller, but a false result guarantees that a request needing
// `maximum_new_entries` cannot be admitted atomically.
bool StableIdRegistryCanRegister(size_t maximum_new_entries);

// Bounded full-digest collision audit for one atomic construction operation.
// Commit validates the operation and releases its audit entries; Rollback also
// releases them.  Stable IDs remain deterministic values carried by committed
// state, while the table cannot grow with the duration of a monitoring session.
// A true truncated-ID collision remains a fatal process condition and is never
// hidden by either Commit or Rollback.
class StableIdRegistryTransaction {
 public:
  StableIdRegistryTransaction();
  ~StableIdRegistryTransaction();

  StableIdRegistryTransaction(const StableIdRegistryTransaction&) = delete;
  StableIdRegistryTransaction& operator=(
      const StableIdRegistryTransaction&) = delete;

  bool active() const { return active_; }
  bool Commit();
  void Rollback();

 private:
  bool active_ = false;
};

class CanonicalBytes {
 public:
  void AddU8(uint8_t value);
  void AddU32(uint32_t value);
  void AddU64(uint64_t value);
  void AddI64(int64_t value);
  void AddDouble(double value);
  void AddFloat(float value);
  void AddString(const std::string& value);
  void AddId(const StableId& value);
  const std::vector<uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
};

StableId MakeStableId(const std::string& domain, const CanonicalBytes& fields);
PointRole RoleForPoint(const std::string& session_id, uint32_t reference_epoch,
                       uint64_t accepted_frame_index, uint64_t canonical_rank);

// Canonical V1 geometry/identity is XYZ+session+epoch+stamp only.  The strict
// XYZI transport intensity is validated but normalized to zero because the
// frozen algorithm contract disables intensity likelihoods.
bool CanonicalizePointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                            const std::string& session_id,
                            uint32_t reference_epoch,
                            const ros::Time& measurement_stamp,
                            uint64_t accepted_frame_index,
                            CanonicalPointVector* points,
                            StableId* frame_id,
                            std::string* error);

// Validate every coordinate against every int32-keyed spatial product before
// CanonicalizePointCloud constructs frame/point identities.  A rejected input
// transaction must not consume the atomic stable-id audit budget.
bool PreflightPointCloudSpatialDomain(
    const pcl::PointCloud<pcl::PointXYZI>& cloud,
    const Eigen::Vector3d& sensor_origin_R,
    const Eigen::Vector3d& grid_origin_R,
    double occupancy_voxel_m,
    double surfel_seed_spacing_m,
    ReasonCode* reason,
    std::string* error);

// Checked mathematical-floor conversion.  All external coordinates must pass
// through this function before entering an int32-keyed spatial product.
bool TryGridKey(const Eigen::Vector3d& point_R,
                const Eigen::Vector3d& origin_R, double voxel_size_m,
                VoxelKey* key);

// Fixed-order pairwise reduction. It is independent of thread scheduling and
// avoids the container-order dependence of a naive accumulation.
double PairwiseSum(const std::vector<double>& values);

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_DETERMINISTIC_HPP
