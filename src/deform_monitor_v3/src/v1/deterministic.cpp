#include "deform_monitor_v3/v1/deterministic.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>

namespace deform_monitor_v3 {
namespace v1 {
namespace {

inline uint32_t RotateRight(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32U - n));
}

const uint32_t kSha256Round[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

uint64_t FirstU64(const Sha256Digest& digest) {
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value = (value << 8U) | digest[static_cast<size_t>(i)];
  return value;
}

uint32_t FloatBits(float value) {
  if (value == 0.0F) value = 0.0F;
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint64_t DoubleBits(double value) {
  if (value == 0.0) value = 0.0;
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double PairwiseSumRange(const std::vector<double>& values, size_t begin, size_t end) {
  if (begin >= end) return 0.0;
  if (end - begin == 1U) return values[begin];
  const size_t middle = begin + (end - begin) / 2U;
  return PairwiseSumRange(values, begin, middle) + PairwiseSumRange(values, middle, end);
}

std::mutex& IdRegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<StableId, Sha256Digest>& IdRegistry() {
  static std::map<StableId, Sha256Digest> registry;
  return registry;
}

uint64_t& IdCollisionCounter() {
  static uint64_t collisions = 0U;
  return collisions;
}

size_t& IdRegistryCapacity() {
  static size_t capacity = 2000000U;
  return capacity;
}

bool& IdRegistryExhausted() {
  static bool exhausted = false;
  return exhausted;
}

bool& IdTransactionActive() {
  static bool active = false;
  return active;
}

bool& IdTransactionExhaustedAtBegin() {
  static bool exhausted = false;
  return exhausted;
}

std::vector<StableId>& IdTransactionAdded() {
  static std::vector<StableId> added;
  return added;
}

}  // namespace

Sha256Digest Sha256(const std::vector<uint8_t>& bytes) {
  std::vector<uint8_t> padded(bytes);
  const uint64_t bit_count = static_cast<uint64_t>(bytes.size()) * 8ULL;
  padded.push_back(0x80U);
  while ((padded.size() % 64U) != 56U) padded.push_back(0U);
  for (int i = 7; i >= 0; --i) {
    padded.push_back(static_cast<uint8_t>((bit_count >> (8U * static_cast<uint32_t>(i))) & 0xffU));
  }

  uint32_t h[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                   0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  for (size_t offset = 0; offset < padded.size(); offset += 64U) {
    uint32_t w[64] = {};
    for (int i = 0; i < 16; ++i) {
      const size_t p = offset + static_cast<size_t>(4 * i);
      w[i] = (static_cast<uint32_t>(padded[p]) << 24U) |
             (static_cast<uint32_t>(padded[p + 1U]) << 16U) |
             (static_cast<uint32_t>(padded[p + 2U]) << 8U) |
             static_cast<uint32_t>(padded[p + 3U]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = RotateRight(w[i - 15], 7U) ^ RotateRight(w[i - 15], 18U) ^
                          (w[i - 15] >> 3U);
      const uint32_t s1 = RotateRight(w[i - 2], 17U) ^ RotateRight(w[i - 2], 19U) ^
                          (w[i - 2] >> 10U);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = RotateRight(e, 6U) ^ RotateRight(e, 11U) ^ RotateRight(e, 25U);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t t1 = hh + s1 + choose + kSha256Round[i] + w[i];
      const uint32_t s0 = RotateRight(a, 2U) ^ RotateRight(a, 13U) ^ RotateRight(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + majority;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  Sha256Digest digest{};
  for (int i = 0; i < 8; ++i) {
    for (int b = 0; b < 4; ++b) {
      digest[static_cast<size_t>(4 * i + b)] =
          static_cast<uint8_t>((h[i] >> (24U - 8U * static_cast<uint32_t>(b))) & 0xffU);
    }
  }
  return digest;
}

std::string HexDigest(const Sha256Digest& digest) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (uint8_t byte : digest) stream << std::setw(2) << static_cast<unsigned int>(byte);
  return stream.str();
}

std::string HexId(const StableId& id) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (uint8_t byte : id.bytes) stream << std::setw(2) << static_cast<unsigned int>(byte);
  return stream.str();
}

bool ParseHexId(const std::string& text, StableId* id) {
  if (!id || (text.size() != 32U && !text.empty())) return false;
  *id = StableId();
  if (text.empty()) return true;
  auto nibble = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; i < 16U; ++i) {
    const int high = nibble(text[2U * i]); const int low = nibble(text[2U * i + 1U]);
    if (high < 0 || low < 0) return false;
    id->bytes[i] = static_cast<uint8_t>((high << 4) | low);
  }
  return true;
}

StableId TruncatedId(const Sha256Digest& digest) {
  StableId id;
  std::copy(digest.begin(), digest.begin() + 16, id.bytes.begin());
  return id;
}

void ResetStableIdRegistry() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  IdRegistry().clear();
  IdCollisionCounter() = 0U;
  IdRegistryExhausted() = false;
  IdTransactionActive() = false;
  IdTransactionExhaustedAtBegin() = false;
  IdTransactionAdded().clear();
}

bool ConfigureStableIdRegistryCapacity(size_t maximum_entries) {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  if (maximum_entries == 0U || IdTransactionActive()) return false;
  IdRegistryCapacity() = maximum_entries;
  IdRegistryExhausted() = IdRegistry().size() > maximum_entries;
  return !IdRegistryExhausted();
}

bool RegisterStableIdDigest(const StableId& id, const Sha256Digest& digest) {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  // StableId128 reserves all zero as INVALID on every wire/API.  A full
  // digest whose truncation lands there is an identity failure; accepting it
  // would make a generated object indistinguishable from an absent ID.
  if (id.empty()) {
    ++IdCollisionCounter();
    return false;
  }
  const auto existing = IdRegistry().find(id);
  if (existing != IdRegistry().end()) {
    if (existing->second == digest) return true;
    ++IdCollisionCounter();
    return false;
  }
  if (IdRegistry().size() >= IdRegistryCapacity()) {
    IdRegistryExhausted() = true;
    return false;
  }
  const auto inserted = IdRegistry().emplace(id, digest);
  if (inserted.second) {
    if (IdTransactionActive()) IdTransactionAdded().push_back(id);
    return true;
  }
  ++IdCollisionCounter();
  return false;
}

bool StableIdRegistryHealthy() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  return IdCollisionCounter() == 0U && !IdRegistryExhausted();
}

bool StableIdRegistryResourceExhausted() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  return IdRegistryExhausted();
}

uint64_t StableIdCollisionCount() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  return IdCollisionCounter();
}

size_t StableIdRegistrySize() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  return IdRegistry().size();
}

bool StableIdRegistryCanRegister(size_t maximum_new_entries) {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  if (IdCollisionCounter() != 0U || IdRegistryExhausted() ||
      IdRegistry().size() > IdRegistryCapacity())
    return false;
  return maximum_new_entries <= IdRegistryCapacity() - IdRegistry().size();
}

StableIdRegistryTransaction::StableIdRegistryTransaction() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  if (IdTransactionActive() || IdCollisionCounter() != 0U ||
      IdRegistryExhausted()) {
    return;
  }
  IdTransactionActive() = true;
  IdTransactionExhaustedAtBegin() = IdRegistryExhausted();
  IdTransactionAdded().clear();
  active_ = true;
}

StableIdRegistryTransaction::~StableIdRegistryTransaction() {
  Rollback();
}

bool StableIdRegistryTransaction::Commit() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  if (!active_ || !IdTransactionActive() ||
      IdCollisionCounter() != 0U || IdRegistryExhausted()) {
    return false;
  }
  // The registry is an atomic-operation collision audit, not an unbounded
  // session ledger.  The committed objects retain their deterministic 128-bit
  // IDs; only the full-digest audit entries are released here.
  for (const StableId& id : IdTransactionAdded()) IdRegistry().erase(id);
  IdTransactionAdded().clear();
  IdTransactionActive() = false;
  IdTransactionExhaustedAtBegin() = false;
  active_ = false;
  return true;
}

void StableIdRegistryTransaction::Rollback() {
  std::lock_guard<std::mutex> lock(IdRegistryMutex());
  if (!active_) return;
  if (IdTransactionActive()) {
    for (const StableId& id : IdTransactionAdded()) IdRegistry().erase(id);
    IdRegistryExhausted() = IdTransactionExhaustedAtBegin();
    IdTransactionAdded().clear();
    IdTransactionActive() = false;
    IdTransactionExhaustedAtBegin() = false;
  }
  active_ = false;
}

void CanonicalBytes::AddU8(uint8_t value) { bytes_.push_back(value); }
void CanonicalBytes::AddU32(uint32_t value) {
  for (int i = 3; i >= 0; --i) AddU8(static_cast<uint8_t>((value >> (8U * i)) & 0xffU));
}
void CanonicalBytes::AddU64(uint64_t value) {
  for (int i = 7; i >= 0; --i) AddU8(static_cast<uint8_t>((value >> (8U * i)) & 0xffU));
}
void CanonicalBytes::AddI64(int64_t value) { AddU64(static_cast<uint64_t>(value)); }
void CanonicalBytes::AddDouble(double value) { AddU64(DoubleBits(value)); }
void CanonicalBytes::AddFloat(float value) { AddU32(FloatBits(value)); }
void CanonicalBytes::AddString(const std::string& value) {
  AddU64(static_cast<uint64_t>(value.size()));
  bytes_.insert(bytes_.end(), value.begin(), value.end());
}
void CanonicalBytes::AddId(const StableId& value) {
  bytes_.insert(bytes_.end(), value.bytes.begin(), value.bytes.end());
}

StableId MakeStableId(const std::string& domain, const CanonicalBytes& fields) {
  CanonicalBytes bytes;
  bytes.AddString(domain);
  for (uint8_t value : fields.bytes()) bytes.AddU8(value);
  const Sha256Digest digest = Sha256(bytes.bytes());
  const StableId id = TruncatedId(digest);
  RegisterStableIdDigest(id, digest);
  return id;
}

PointRole RoleForPoint(const std::string& session_id, uint32_t reference_epoch,
                       uint64_t accepted_frame_index, uint64_t canonical_rank) {
  CanonicalBytes key;
  key.AddString(session_id);
  key.AddU32(reference_epoch);
  key.AddU64(accepted_frame_index);
  key.AddU64(canonical_rank);
  key.AddString("role-v1");
  const uint64_t bucket = FirstU64(Sha256(key.bytes())) % 100ULL;
  if (bucket < 15ULL) return PointRole::POSE_CONTROL;
  if (bucket < 45ULL) return PointRole::ASSOCIATION;
  if (bucket < 90ULL) return PointRole::MEASUREMENT;
  return PointRole::OCCUPANCY;
}

bool CanonicalizePointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                            const std::string& session_id,
                            uint32_t reference_epoch,
                            const ros::Time& measurement_stamp,
                            uint64_t accepted_frame_index,
                            CanonicalPointVector* points,
                            StableId* frame_id,
                            std::string* error) {
  if (!points || !frame_id || session_id.empty() || measurement_stamp.isZero()) {
    if (error) *error = "null output, empty session_id, or zero measurement stamp";
    return false;
  }
  points->clear();
  points->reserve(cloud.size());
  for (size_t i = 0; i < cloud.size(); ++i) {
    const pcl::PointXYZI& raw = cloud[i];
    if (!std::isfinite(raw.x) || !std::isfinite(raw.y) ||
        !std::isfinite(raw.z) || !std::isfinite(raw.intensity)) {
      if (error) *error = "non-finite PointXYZI";
      points->clear();
      return false;
    }
    CanonicalPoint point;
    point.position_R = Eigen::Vector3d(raw.x, raw.y, raw.z);
    // V1 freezes intensity_likelihood_usage=DISABLED.  Keep validating the
    // producer's strict XYZI wire schema above, but erase intensity at the
    // canonical algorithm boundary so Gazebo laser_retro/reflectivity cannot
    // change geometric identity, evidence assignment, support roles, or any
    // downstream detection input.  XYZ is preserved exactly.
    point.intensity = 0.0F;
    point.original_index = static_cast<uint64_t>(i);
    points->push_back(point);
  }
  std::stable_sort(points->begin(), points->end(), [](const CanonicalPoint& lhs,
                                                       const CanonicalPoint& rhs) {
    for (int axis = 0; axis < 3; ++axis) {
      if (lhs.position_R[axis] != rhs.position_R[axis])
        return lhs.position_R[axis] < rhs.position_R[axis];
    }
    // Exact XYZ duplicates are geometrically indistinguishable in V1;
    // intensity is forbidden as an ordering tie-breaker.
    return lhs.original_index < rhs.original_index;
  });

  CanonicalBytes frame_fields;
  frame_fields.AddString(session_id);
  frame_fields.AddU32(reference_epoch);
  // A retransmitted sensor message must retain its identity even after the
  // local accepted-frame counter has advanced, while two genuinely distinct
  // observations of an unchanged scene must not collapse to one frame.  The
  // source timestamp therefore belongs to FrameId; accepted_frame_index is
  // reserved for the deterministic 15/30/45/10 support split.
  frame_fields.AddU32(measurement_stamp.sec);
  frame_fields.AddU32(measurement_stamp.nsec);
  frame_fields.AddU64(static_cast<uint64_t>(points->size()));
  for (size_t i = 0; i < points->size(); ++i) {
    CanonicalPoint& point = (*points)[i];
    point.canonical_rank = static_cast<uint64_t>(i);
    frame_fields.AddDouble(point.position_R.x());
    frame_fields.AddDouble(point.position_R.y());
    frame_fields.AddDouble(point.position_R.z());
  }
  *frame_id = MakeStableId("frame-v1", frame_fields);
  for (CanonicalPoint& point : *points) {
    CanonicalBytes point_fields;
    point_fields.AddId(*frame_id);
    point_fields.AddU64(point.canonical_rank);
    point.point_id = MakeStableId("point-v1", point_fields);
    point.evidence_id = MakeStableId("evidence-v1", point_fields);
    point.correlation_group_id = *frame_id;
    point.role = RoleForPoint(session_id, reference_epoch, accepted_frame_index,
                             point.canonical_rank);
  }
  if (!StableIdRegistryHealthy()) {
    if (error) *error = "stable id truncation collision";
    points->clear();
    *frame_id = StableId();
    return false;
  }
  return true;
}

bool PreflightPointCloudSpatialDomain(
    const pcl::PointCloud<pcl::PointXYZI>& cloud,
    const Eigen::Vector3d& sensor_origin_R,
    const Eigen::Vector3d& grid_origin_R,
    double occupancy_voxel_m,
    double surfel_seed_spacing_m,
    ReasonCode* reason,
    std::string* error) {
  if (reason) *reason = ReasonCode::INPUT_SCHEMA_INVALID;
  if (!sensor_origin_R.allFinite() || !grid_origin_R.allFinite() ||
      !(occupancy_voxel_m > 0.0) ||
      !std::isfinite(occupancy_voxel_m) ||
      !(surfel_seed_spacing_m > 0.0) ||
      !std::isfinite(surfel_seed_spacing_m)) {
    if (error) *error = "invalid sensor origin or spatial-grid contract";
    return false;
  }
  VoxelKey checked_key;
  if (!TryGridKey(sensor_origin_R, grid_origin_R, occupancy_voxel_m,
                  &checked_key)) {
    if (reason) *reason = ReasonCode::RESOURCE_LIMIT;
    if (error) *error = "sensor origin is outside the occupancy grid domain";
    return false;
  }
  for (const pcl::PointXYZI& point : cloud) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z) || !std::isfinite(point.intensity)) {
      if (error) *error = "non-finite PointXYZI";
      return false;
    }
    const Eigen::Vector3d position_R(point.x, point.y, point.z);
    if (!TryGridKey(position_R, grid_origin_R, occupancy_voxel_m,
                    &checked_key) ||
        !TryGridKey(position_R, grid_origin_R, surfel_seed_spacing_m,
                    &checked_key)) {
      if (reason) *reason = ReasonCode::RESOURCE_LIMIT;
      if (error) *error =
          "point is outside an int32-keyed spatial product domain";
      return false;
    }
  }
  return true;
}

bool TryGridKey(const Eigen::Vector3d& point_R,
                const Eigen::Vector3d& origin_R, double voxel_size_m,
                VoxelKey* key) {
  if (!key || !point_R.allFinite() || !origin_R.allFinite() ||
      !(voxel_size_m > 0.0) || !std::isfinite(voxel_size_m)) {
    return false;
  }
  const Eigen::Array3d scaled = (point_R - origin_R).array() / voxel_size_m;
  if (!scaled.isFinite().all()) return false;
  const double minimum = static_cast<double>(std::numeric_limits<int32_t>::min());
  // floor(x) must remain <= INT32_MAX.  Testing the floored value also admits
  // coordinates within the final half-open voxel without an unsafe cast.
  const double maximum = static_cast<double>(std::numeric_limits<int32_t>::max());
  Eigen::Array3d floored;
  for (int axis = 0; axis < 3; ++axis) {
    floored[axis] = std::floor(scaled[axis]);
    if (!std::isfinite(floored[axis]) || floored[axis] < minimum ||
        floored[axis] > maximum) {
      return false;
    }
  }
  *key = VoxelKey{static_cast<int32_t>(floored.x()),
                  static_cast<int32_t>(floored.y()),
                  static_cast<int32_t>(floored.z())};
  return true;
}

double PairwiseSum(const std::vector<double>& values) {
  return PairwiseSumRange(values, 0U, values.size());
}

}  // namespace v1
}  // namespace deform_monitor_v3
