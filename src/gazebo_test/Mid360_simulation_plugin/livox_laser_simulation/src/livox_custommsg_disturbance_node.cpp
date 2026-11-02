#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <livox_ros_driver/CustomMsg.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

namespace {

constexpr double kPi = 3.14159265358979323846;

double Clamp(const double value, const double lower, const double upper) {
  return std::max(lower, std::min(upper, value));
}

double WrapAngle(double angle) {
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream stream;
  for (const char character : value) {
    switch (character) {
      case '\\': stream << "\\\\"; break;
      case '"': stream << "\\\""; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default: stream << character; break;
    }
  }
  return stream.str();
}

double RadToDeg(const double radians) { return radians * 180.0 / kPi; }
double DegToRad(const double degrees) { return degrees * kPi / 180.0; }

// Parses a comma-separated list of doubles (whitespace around each token is
// trimmed, empty tokens are skipped). Used for the occlusion_multi sector
// parameters, which are passed as rosparam strings rather than a YAML array
// so the existing single-value rosrun command-line plumbing in
// run_alert_synchronized_10obj_formal_batch_v2_disturbution.sh keeps working
// unchanged for every other profile.
std::vector<double> SplitDoubleList(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const std::size_t start = token.find_first_not_of(" \t");
    if (start == std::string::npos) {
      continue;
    }
    const std::size_t end = token.find_last_not_of(" \t");
    values.push_back(std::stod(token.substr(start, end - start + 1)));
  }
  return values;
}

// splitmix64 finalizer, used only to turn (seed, frame_index, grid cell) into
// a deterministic pseudo-random activation decision for fog_patchy -- see
// CellActivationValue below. Not a cryptographic hash, just decorrelation.
std::uint64_t MixHash(std::uint64_t value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return value;
}

// Deterministic pseudo-random value in [0, 1) for one (frame, angular grid
// cell) pair. Recomputed from scratch on every call -- nothing about a cell's
// fog-patch activation is stored between frames or between points, so two
// points falling in the same cell within the same frame always get the same
// activation decision (this is what makes a "patch" spatially coherent
// within a frame) while consecutive frames are independent draws (this is
// what gives the "quickly appears/disappears" flicker the reviewer asked
// for) without any persistent smoke-blob state.
double CellActivationValue(const std::uint64_t seed, const std::uint64_t frame_index,
                           const std::int64_t cell_azimuth, const std::int64_t cell_elevation) {
  std::uint64_t hash = MixHash(seed ^ (frame_index * 0x9E3779B97F4A7C15ULL));
  hash = MixHash(hash ^ (static_cast<std::uint64_t>(cell_azimuth) * 0xBF58476D1CE4E5B9ULL));
  hash = MixHash(hash ^ (static_cast<std::uint64_t>(cell_elevation) * 0x94D049BB133111EBULL));
  return static_cast<double>(hash) /
         static_cast<double>(std::numeric_limits<std::uint64_t>::max());
}

// One occlusion cone for the occlusion_multi mode: same azimuth-sector /
// shared-elevation-band / near-range-return model as the existing single-
// sector occlusion mode, just parameterized per sector so several
// non-overlapping cones (different bearings, different stand-off ranges) can
// be active at once.
struct OcclusionSector {
  double center_azimuth_rad = 0.0;
  double half_width_rad = 0.0;
  double range_m = 1.8;
};

struct FrameStats {
  std::uint64_t frame_index = 0;
  std::uint64_t input_points = 0;
  std::uint64_t output_points = 0;
  std::uint64_t fog_dropped_points = 0;
  std::uint64_t fog_noisy_points = 0;
  std::uint64_t fog_backscatter_points = 0;
  std::uint64_t occluded_points = 0;
};

class LivoxCustomMsgDisturbanceNode {
 public:
  LivoxCustomMsgDisturbanceNode()
      : private_node_("~"), random_engine_(1), uniform_(0.0, 1.0) {
    private_node_.param<std::string>("input_topic", input_topic_, "/livox/lidar");
    private_node_.param<std::string>("output_topic", output_topic_,
                                    "/livox/lidar_disturbed");
    private_node_.param<std::string>("phase_topic", phase_topic_,
                                    "/alert_experiment/phase");
    private_node_.param<std::string>("profile", profile_, "clean_passthrough");
    private_node_.param<std::string>("mode", mode_, "clean");
    private_node_.param<std::string>("activation_phase", activation_phase_, "MOTION");
    private_node_.param<std::string>("deactivation_phase", deactivation_phase_,
                                    "POST_MOTION_HOLD");
    private_node_.param<std::string>("audit_csv_path", audit_csv_path_, "");
    private_node_.param<std::string>("manifest_path", manifest_path_, "");

    int seed = 1;
    private_node_.param("seed", seed, 1);
    seed_ = static_cast<std::uint64_t>(std::max(0, seed));
    random_engine_.seed(seed_);

    private_node_.param("fog_beta_per_m", fog_beta_per_m_, 0.0);
    private_node_.param("fog_sigma_base_m", fog_sigma_base_m_, 0.0);
    private_node_.param("fog_sigma_per_m", fog_sigma_per_m_, 0.0);
    private_node_.param("fog_backscatter_probability", fog_backscatter_probability_, 0.0);
    private_node_.param("fog_backscatter_min_range_m", fog_backscatter_min_range_m_, 0.3);
    private_node_.param("fog_backscatter_max_range_m", fog_backscatter_max_range_m_, 3.0);

    double occlusion_center_deg = 0.0;
    double occlusion_width_deg = 180.0;
    double occlusion_min_elevation_deg = -90.0;
    double occlusion_max_elevation_deg = 90.0;
    private_node_.param("occlusion_center_azimuth_deg", occlusion_center_deg, 0.0);
    private_node_.param("occlusion_width_deg", occlusion_width_deg, 180.0);
    private_node_.param("occlusion_min_elevation_deg", occlusion_min_elevation_deg, -90.0);
    private_node_.param("occlusion_max_elevation_deg", occlusion_max_elevation_deg, 90.0);
    private_node_.param("occlusion_return_range_m", occlusion_return_range_m_, 1.8);
    private_node_.param("occlusion_return_sigma_m", occlusion_return_sigma_m_, 0.005);
    occlusion_center_azimuth_rad_ = occlusion_center_deg * kPi / 180.0;
    occlusion_half_width_rad_ = Clamp(occlusion_width_deg, 0.0, 360.0) * kPi / 360.0;
    occlusion_min_elevation_rad_ = occlusion_min_elevation_deg * kPi / 180.0;
    occlusion_max_elevation_rad_ = occlusion_max_elevation_deg * kPi / 180.0;

    // occlusion_multi: N independent sectors, one per (azimuth, width, range)
    // triple, all sharing the elevation band above. Every list must be the
    // same length; that length is the sector count. Empty (mode != occlusion
    // or profile not occlusion_multi) means this mode is inert.
    private_node_.param<std::string>("occlusion_multi_center_azimuth_deg",
                                    occlusion_multi_center_azimuth_deg_raw_, "");
    private_node_.param<std::string>("occlusion_multi_width_deg",
                                    occlusion_multi_width_deg_raw_, "");
    private_node_.param<std::string>("occlusion_multi_range_m",
                                    occlusion_multi_range_m_raw_, "");
    BuildOcclusionSectors();

    // fog_patchy: same base range-dependent dropout + (enlarged) radial noise
    // as the existing fog mode for points that don't fall in an active patch
    // this frame. A point's azimuth/elevation is binned into an angular grid
    // (fog_patch_cell_deg wide); each grid cell is independently re-rolled
    // active/inactive every frame via CellActivationValue (no state kept
    // between frames -- see that function's comment). Points in an active
    // cell have their real return dropped entirely and are replaced with a
    // synthetic near-range return whose direction is jittered away from the
    // original ray by up to fog_patch_angular_jitter_deg, so the several
    // points that fall in the same active cell in the same frame cluster
    // into a small, non-planar, off-axis blob rather than lying exactly on
    // the true surface -- and because each cell's activation is redrawn every
    // frame, the blob's cell membership (and therefore its footprint) moves
    // frame to frame, giving the "floats / flickers / drifts" look requested
    // without simulating individual smoke particles.
    private_node_.param("fog_patch_probability", fog_patch_probability_, 0.0);
    private_node_.param("fog_patch_cell_deg", fog_patch_cell_deg_, 15.0);
    private_node_.param("fog_patch_min_range_m", fog_patch_min_range_m_, 0.3);
    private_node_.param("fog_patch_max_range_m", fog_patch_max_range_m_, 2.5);
    private_node_.param("fog_patch_angular_jitter_deg", fog_patch_angular_jitter_deg_, 3.0);

    ValidateParameters();
    OpenAuditCsv();
    WriteManifest();

    point_publisher_ = node_.advertise<livox_ros_driver::CustomMsg>(output_topic_, 5);
    stats_publisher_ = private_node_.advertise<std_msgs::String>("stats", 10);
    phase_subscriber_ = node_.subscribe(phase_topic_, 5,
                                       &LivoxCustomMsgDisturbanceNode::HandlePhase, this);
    point_subscriber_ = node_.subscribe(input_topic_, 5,
                                       &LivoxCustomMsgDisturbanceNode::HandleCloud, this);

    ROS_INFO_STREAM("Livox disturbance relay ready: profile=" << profile_
                    << " mode=" << mode_ << " input=" << input_topic_
                    << " output=" << output_topic_
                    << " activation_phase=" << activation_phase_
                    << " deactivation_phase=" << deactivation_phase_);
  }

 private:
  void BuildOcclusionSectors() {
    occlusion_sectors_.clear();
    if (mode_ != "occlusion_multi") {
      return;
    }
    const std::vector<double> centers = SplitDoubleList(occlusion_multi_center_azimuth_deg_raw_);
    const std::vector<double> widths = SplitDoubleList(occlusion_multi_width_deg_raw_);
    const std::vector<double> ranges = SplitDoubleList(occlusion_multi_range_m_raw_);
    if (centers.empty() || centers.size() != widths.size() ||
        centers.size() != ranges.size()) {
      throw std::invalid_argument(
          "occlusion_multi_center_azimuth_deg/width_deg/range_m must be "
          "non-empty comma-separated lists of equal length");
    }
    for (std::size_t i = 0; i < centers.size(); ++i) {
      OcclusionSector sector;
      sector.center_azimuth_rad = DegToRad(centers[i]);
      sector.half_width_rad = DegToRad(Clamp(widths[i], 0.0, 360.0)) / 2.0;
      sector.range_m = ranges[i];
      occlusion_sectors_.push_back(sector);
    }
  }

  void ValidateParameters() const {
    if (input_topic_.empty() || output_topic_.empty() || input_topic_ == output_topic_) {
      throw std::invalid_argument("input_topic and output_topic must be non-empty and distinct");
    }
    if (mode_ != "clean" && mode_ != "fog" && mode_ != "occlusion" &&
        mode_ != "occlusion_multi" && mode_ != "fog_patchy") {
      throw std::invalid_argument(
          "mode must be clean, fog, occlusion, occlusion_multi, or fog_patchy");
    }
    const double values[] = {
        fog_beta_per_m_, fog_sigma_base_m_, fog_sigma_per_m_,
        fog_backscatter_probability_, fog_backscatter_min_range_m_,
        fog_backscatter_max_range_m_, occlusion_return_range_m_,
        occlusion_return_sigma_m_, fog_patch_probability_, fog_patch_cell_deg_,
        fog_patch_min_range_m_, fog_patch_max_range_m_, fog_patch_angular_jitter_deg_};
    for (const double value : values) {
      if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument("disturbance numeric parameters must be finite and non-negative");
      }
    }
    if (fog_backscatter_probability_ > 1.0 ||
        fog_backscatter_max_range_m_ < fog_backscatter_min_range_m_) {
      throw std::invalid_argument("fog backscatter parameters are invalid");
    }
    if (occlusion_min_elevation_rad_ > occlusion_max_elevation_rad_ ||
        occlusion_return_range_m_ <= 0.0) {
      throw std::invalid_argument("occlusion parameters are invalid");
    }
    if (mode_ == "occlusion_multi" && occlusion_sectors_.empty()) {
      throw std::invalid_argument("occlusion_multi mode requires at least one sector");
    }
    if (mode_ == "fog_patchy") {
      if (fog_patch_probability_ > 1.0) {
        throw std::invalid_argument("fog_patch_probability must be in [0, 1]");
      }
      if (fog_patch_cell_deg_ <= 0.0 || fog_patch_cell_deg_ > 90.0) {
        throw std::invalid_argument("fog_patch_cell_deg must be in (0, 90]");
      }
      if (fog_patch_max_range_m_ < fog_patch_min_range_m_) {
        throw std::invalid_argument("fog_patch_max_range_m must be >= fog_patch_min_range_m");
      }
    }
  }

  void OpenAuditCsv() {
    if (audit_csv_path_.empty()) {
      return;
    }
    const std::filesystem::path path(audit_csv_path_);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    audit_csv_.open(path, std::ios::out | std::ios::trunc);
    if (!audit_csv_) {
      throw std::runtime_error("cannot open disturbance audit CSV: " + audit_csv_path_);
    }
    audit_csv_ << "stamp_sec,frame_index,phase,active,profile,mode,input_points,"
                  "output_points,fog_dropped_points,fog_noisy_points,"
                  "fog_backscatter_points,occluded_points,retention_ratio\n";
    audit_csv_.flush();
  }

  void WriteManifest() const {
    if (manifest_path_.empty()) {
      return;
    }
    const std::filesystem::path path(manifest_path_);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot write disturbance manifest: " + manifest_path_);
    }
    output << std::setprecision(12)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"profile\": \"" << JsonEscape(profile_) << "\",\n"
           << "  \"mode\": \"" << JsonEscape(mode_) << "\",\n"
           << "  \"input_topic\": \"" << JsonEscape(input_topic_) << "\",\n"
           << "  \"output_topic\": \"" << JsonEscape(output_topic_) << "\",\n"
           << "  \"phase_topic\": \"" << JsonEscape(phase_topic_) << "\",\n"
           << "  \"activation_phase\": \"" << JsonEscape(activation_phase_) << "\",\n"
           << "  \"deactivation_phase\": \"" << JsonEscape(deactivation_phase_) << "\",\n"
           << "  \"seed\": " << seed_ << ",\n"
           << "  \"fog\": {\n"
           << "    \"beta_per_m\": " << fog_beta_per_m_ << ",\n"
           << "    \"sigma_base_m\": " << fog_sigma_base_m_ << ",\n"
           << "    \"sigma_per_m\": " << fog_sigma_per_m_ << ",\n"
           << "    \"backscatter_probability\": " << fog_backscatter_probability_ << ",\n"
           << "    \"backscatter_min_range_m\": " << fog_backscatter_min_range_m_ << ",\n"
           << "    \"backscatter_max_range_m\": " << fog_backscatter_max_range_m_ << "\n"
           << "  },\n"
           << "  \"occlusion\": {\n"
           << "    \"center_azimuth_rad\": " << occlusion_center_azimuth_rad_ << ",\n"
           << "    \"half_width_rad\": " << occlusion_half_width_rad_ << ",\n"
           << "    \"min_elevation_rad\": " << occlusion_min_elevation_rad_ << ",\n"
           << "    \"max_elevation_rad\": " << occlusion_max_elevation_rad_ << ",\n"
           << "    \"return_range_m\": " << occlusion_return_range_m_ << ",\n"
           << "    \"return_sigma_m\": " << occlusion_return_sigma_m_ << "\n"
           << "  },\n"
           << "  \"occlusion_multi\": {\n"
           << "    \"sector_count\": " << occlusion_sectors_.size() << ",\n"
           << "    \"sectors\": [";
    for (std::size_t i = 0; i < occlusion_sectors_.size(); ++i) {
      const auto& sector = occlusion_sectors_[i];
      output << (i == 0 ? "\n" : ",\n")
             << "      {\"center_azimuth_rad\": " << sector.center_azimuth_rad
             << ", \"half_width_rad\": " << sector.half_width_rad
             << ", \"range_m\": " << sector.range_m << "}";
    }
    output << (occlusion_sectors_.empty() ? "]\n" : "\n    ]\n")
           << "  },\n"
           << "  \"fog_patchy\": {\n"
           << "    \"patch_probability\": " << fog_patch_probability_ << ",\n"
           << "    \"cell_deg\": " << fog_patch_cell_deg_ << ",\n"
           << "    \"min_range_m\": " << fog_patch_min_range_m_ << ",\n"
           << "    \"max_range_m\": " << fog_patch_max_range_m_ << ",\n"
           << "    \"angular_jitter_deg\": " << fog_patch_angular_jitter_deg_ << "\n"
           << "  },\n"
           << "  \"reflectivity_policy\": \"preserve_original_ids_and_use_zero_for_synthetic_returns\",\n"
           << "  \"timing_policy\": \"clean_reference_then_disturb_during_motion\"\n"
           << "}\n";
  }

  void HandlePhase(const std_msgs::String::ConstPtr& message) {
    static const std::regex phase_pattern("\\\"phase\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(message->data, match, phase_pattern) || match.size() < 2) {
      ROS_WARN_THROTTLE(5.0, "Disturbance relay received an invalid experiment phase payload");
      return;
    }
    current_phase_ = match[1].str();
    if (current_phase_ == activation_phase_) {
      active_ = mode_ != "clean";
    } else if (current_phase_ == deactivation_phase_ || current_phase_ == "COMPLETED") {
      active_ = false;
    }
    ROS_INFO_STREAM("Livox disturbance phase=" << current_phase_
                    << " active=" << (active_ ? "true" : "false"));
  }

  void HandleCloud(const livox_ros_driver::CustomMsg::ConstPtr& input) {
    livox_ros_driver::CustomMsg output = *input;
    output.points.clear();
    output.points.reserve(input->points.size());

    FrameStats stats;
    stats.frame_index = ++frame_index_;
    stats.input_points = input->points.size();

    for (const auto& original : input->points) {
      if (!active_) {
        output.points.push_back(original);
        continue;
      }
      if (mode_ == "fog") {
        ProcessFogPoint(original, &output, &stats);
      } else if (mode_ == "occlusion") {
        ProcessOcclusionPoint(original, &output, &stats);
      } else if (mode_ == "occlusion_multi") {
        ProcessOcclusionMultiPoint(original, &output, &stats);
      } else if (mode_ == "fog_patchy") {
        ProcessFogPatchyPoint(original, stats.frame_index, &output, &stats);
      } else {
        output.points.push_back(original);
      }
    }

    output.point_num = static_cast<std::uint32_t>(output.points.size());
    stats.output_points = output.points.size();
    point_publisher_.publish(output);
    PublishStats(input->header.stamp.toSec(), stats);
  }

  void ProcessFogPoint(const livox_ros_driver::CustomPoint& original,
                       livox_ros_driver::CustomMsg* output, FrameStats* stats) {
    const double x = original.x;
    const double y = original.y;
    const double z = original.z;
    const double range = std::sqrt(x * x + y * y + z * z);
    if (!std::isfinite(range) || range <= 1.0e-6) {
      output->points.push_back(original);
      return;
    }

    const double keep_probability = Clamp(std::exp(-fog_beta_per_m_ * range), 0.0, 1.0);
    if (uniform_(random_engine_) > keep_probability) {
      ++stats->fog_dropped_points;
      return;
    }

    auto point = original;
    double disturbed_range = range;
    if (uniform_(random_engine_) < fog_backscatter_probability_ &&
        range > fog_backscatter_min_range_m_) {
      const double upper = std::min(range, fog_backscatter_max_range_m_);
      if (upper > fog_backscatter_min_range_m_) {
        disturbed_range = fog_backscatter_min_range_m_ +
            uniform_(random_engine_) * (upper - fog_backscatter_min_range_m_);
        point.reflectivity = 0;
        ++stats->fog_backscatter_points;
      }
    } else {
      const double sigma = fog_sigma_base_m_ + fog_sigma_per_m_ * range;
      if (sigma > 0.0) {
        std::normal_distribution<double> noise(0.0, sigma);
        disturbed_range = std::max(0.21, range + noise(random_engine_));
        ++stats->fog_noisy_points;
      }
    }
    const double scale = disturbed_range / range;
    point.x = static_cast<float>(x * scale);
    point.y = static_cast<float>(y * scale);
    point.z = static_cast<float>(z * scale);
    output->points.push_back(point);
  }

  void ProcessOcclusionPoint(const livox_ros_driver::CustomPoint& original,
                             livox_ros_driver::CustomMsg* output, FrameStats* stats) {
    const double x = original.x;
    const double y = original.y;
    const double z = original.z;
    const double horizontal_range = std::hypot(x, y);
    const double range = std::hypot(horizontal_range, z);
    if (!std::isfinite(range) || range <= 1.0e-6) {
      output->points.push_back(original);
      return;
    }
    const double azimuth = std::atan2(y, x);
    const double elevation = std::atan2(z, horizontal_range);
    const bool in_sector =
        std::abs(WrapAngle(azimuth - occlusion_center_azimuth_rad_)) <=
            occlusion_half_width_rad_ &&
        elevation >= occlusion_min_elevation_rad_ &&
        elevation <= occlusion_max_elevation_rad_;
    if (!in_sector || range <= occlusion_return_range_m_) {
      output->points.push_back(original);
      return;
    }

    std::normal_distribution<double> noise(0.0, occlusion_return_sigma_m_);
    const double disturbed_range = std::max(
        0.21, occlusion_return_range_m_ +
                  (occlusion_return_sigma_m_ > 0.0 ? noise(random_engine_) : 0.0));
    const double scale = disturbed_range / range;
    auto point = original;
    point.x = static_cast<float>(x * scale);
    point.y = static_cast<float>(y * scale);
    point.z = static_cast<float>(z * scale);
    point.reflectivity = 0;
    output->points.push_back(point);
    ++stats->occluded_points;
  }

  void ProcessOcclusionMultiPoint(const livox_ros_driver::CustomPoint& original,
                                  livox_ros_driver::CustomMsg* output, FrameStats* stats) {
    const double x = original.x;
    const double y = original.y;
    const double z = original.z;
    const double horizontal_range = std::hypot(x, y);
    const double range = std::hypot(horizontal_range, z);
    if (!std::isfinite(range) || range <= 1.0e-6) {
      output->points.push_back(original);
      return;
    }
    const double azimuth = std::atan2(y, x);
    const double elevation = std::atan2(z, horizontal_range);
    if (elevation < occlusion_min_elevation_rad_ || elevation > occlusion_max_elevation_rad_) {
      output->points.push_back(original);
      return;
    }

    // First sector (by list order) whose azimuth band contains this point AND
    // whose stand-off range is nearer than the point wins -- each sector is a
    // self-contained "something is standing at range_m in this direction"
    // obstruction, independent of the others; sectors are expected not to
    // overlap in azimuth (the reviewer batch keeps them at least one sector
    // width apart), so in practice at most one sector ever matches a given
    // point.
    for (const auto& sector : occlusion_sectors_) {
      const bool in_sector =
          std::abs(WrapAngle(azimuth - sector.center_azimuth_rad)) <= sector.half_width_rad;
      if (!in_sector || range <= sector.range_m) {
        continue;
      }
      std::normal_distribution<double> noise(0.0, occlusion_return_sigma_m_);
      const double disturbed_range = std::max(
          0.21, sector.range_m +
                    (occlusion_return_sigma_m_ > 0.0 ? noise(random_engine_) : 0.0));
      const double scale = disturbed_range / range;
      auto point = original;
      point.x = static_cast<float>(x * scale);
      point.y = static_cast<float>(y * scale);
      point.z = static_cast<float>(z * scale);
      point.reflectivity = 0;
      output->points.push_back(point);
      ++stats->occluded_points;
      return;
    }
    output->points.push_back(original);
  }

  void ProcessFogPatchyPoint(const livox_ros_driver::CustomPoint& original,
                             const std::uint64_t frame_index,
                             livox_ros_driver::CustomMsg* output, FrameStats* stats) {
    const double x = original.x;
    const double y = original.y;
    const double z = original.z;
    const double horizontal_range = std::hypot(x, y);
    const double range = std::hypot(horizontal_range, z);
    if (!std::isfinite(range) || range <= 1.0e-6) {
      output->points.push_back(original);
      return;
    }

    // Base range-dependent extinction: same shape as the existing fog mode,
    // still governed by fog_beta_per_m so the two "clean" degradation knobs
    // (dropout, noise) stay comparable in kind across fog profiles.
    const double keep_probability = Clamp(std::exp(-fog_beta_per_m_ * range), 0.0, 1.0);
    if (uniform_(random_engine_) > keep_probability) {
      ++stats->fog_dropped_points;
      return;
    }

    const double azimuth_deg = RadToDeg(std::atan2(y, x));
    const double elevation_deg = RadToDeg(std::atan2(z, horizontal_range));
    const auto cell_azimuth =
        static_cast<std::int64_t>(std::floor(azimuth_deg / fog_patch_cell_deg_));
    const auto cell_elevation =
        static_cast<std::int64_t>(std::floor(elevation_deg / fog_patch_cell_deg_));
    const bool cell_active =
        fog_patch_probability_ > 0.0 &&
        CellActivationValue(seed_, frame_index, cell_azimuth, cell_elevation) <
            fog_patch_probability_;

    auto point = original;
    if (cell_active) {
      // The real return behind this patch is fully replaced (not merely
      // perturbed) -- this is the "backscatter wins the first/strongest
      // return, the true surface point disappears" mechanism the reviewer
      // described, and repeated over many frames it is what produces a
      // standing "shadow" behind an active patch. The synthetic return's
      // direction is jittered off the original ray (unlike the existing
      // fog_backscatter effect, which stays exactly on-ray) so that the
      // handful of points landing in the same active cell this frame do not
      // all lie on one radial line -- they scatter into a small volume with
      // no shared surface normal, i.e. not a plane or edge.
      std::normal_distribution<double> jitter(0.0, fog_patch_angular_jitter_deg_);
      const double jittered_azimuth =
          DegToRad(azimuth_deg + jitter(random_engine_));
      const double jittered_elevation = Clamp(
          DegToRad(elevation_deg + jitter(random_engine_)), -kPi / 2.0 + 1.0e-3,
          kPi / 2.0 - 1.0e-3);
      const double patch_range =
          fog_patch_min_range_m_ +
          uniform_(random_engine_) * std::max(0.0, fog_patch_max_range_m_ - fog_patch_min_range_m_);
      const double horiz = patch_range * std::cos(jittered_elevation);
      point.x = static_cast<float>(horiz * std::cos(jittered_azimuth));
      point.y = static_cast<float>(horiz * std::sin(jittered_azimuth));
      point.z = static_cast<float>(patch_range * std::sin(jittered_elevation));
      point.reflectivity = 0;
      ++stats->fog_backscatter_points;
    } else {
      const double sigma = fog_sigma_base_m_ + fog_sigma_per_m_ * range;
      double disturbed_range = range;
      if (sigma > 0.0) {
        std::normal_distribution<double> noise(0.0, sigma);
        disturbed_range = std::max(0.21, range + noise(random_engine_));
        ++stats->fog_noisy_points;
      }
      const double scale = disturbed_range / range;
      point.x = static_cast<float>(x * scale);
      point.y = static_cast<float>(y * scale);
      point.z = static_cast<float>(z * scale);
    }
    output->points.push_back(point);
  }

  void PublishStats(const double stamp_sec, const FrameStats& stats) {
    const double retention = stats.input_points > 0
                                 ? static_cast<double>(stats.output_points) /
                                       static_cast<double>(stats.input_points)
                                 : 0.0;
    std::ostringstream payload;
    payload << std::setprecision(12)
            << "{\"stamp_sec\":" << stamp_sec
            << ",\"frame_index\":" << stats.frame_index
            << ",\"phase\":\"" << JsonEscape(current_phase_) << "\""
            << ",\"active\":" << (active_ ? "true" : "false")
            << ",\"profile\":\"" << JsonEscape(profile_) << "\""
            << ",\"mode\":\"" << JsonEscape(mode_) << "\""
            << ",\"input_points\":" << stats.input_points
            << ",\"output_points\":" << stats.output_points
            << ",\"fog_dropped_points\":" << stats.fog_dropped_points
            << ",\"fog_noisy_points\":" << stats.fog_noisy_points
            << ",\"fog_backscatter_points\":" << stats.fog_backscatter_points
            << ",\"occluded_points\":" << stats.occluded_points
            << ",\"retention_ratio\":" << retention << "}";
    std_msgs::String message;
    message.data = payload.str();
    stats_publisher_.publish(message);

    if (audit_csv_) {
      audit_csv_ << std::setprecision(12) << stamp_sec << ',' << stats.frame_index << ','
                 << current_phase_ << ',' << (active_ ? 1 : 0) << ',' << profile_ << ','
                 << mode_ << ',' << stats.input_points << ',' << stats.output_points << ','
                 << stats.fog_dropped_points << ',' << stats.fog_noisy_points << ','
                 << stats.fog_backscatter_points << ',' << stats.occluded_points << ','
                 << retention << '\n';
      // This audit is small compared with the point stream; flush every frame
      // so an interrupted formal run still has a precise disturbance history.
      audit_csv_.flush();
    }
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Subscriber point_subscriber_;
  ros::Subscriber phase_subscriber_;
  ros::Publisher point_publisher_;
  ros::Publisher stats_publisher_;

  std::string input_topic_;
  std::string output_topic_;
  std::string phase_topic_;
  std::string profile_;
  std::string mode_;
  std::string activation_phase_;
  std::string deactivation_phase_;
  std::string audit_csv_path_;
  std::string manifest_path_;
  std::string current_phase_ = "PRE_COORDINATOR";
  bool active_ = false;
  std::uint64_t frame_index_ = 0;
  std::uint64_t seed_ = 1;

  double fog_beta_per_m_ = 0.0;
  double fog_sigma_base_m_ = 0.0;
  double fog_sigma_per_m_ = 0.0;
  double fog_backscatter_probability_ = 0.0;
  double fog_backscatter_min_range_m_ = 0.3;
  double fog_backscatter_max_range_m_ = 3.0;
  double occlusion_center_azimuth_rad_ = 0.0;
  double occlusion_half_width_rad_ = kPi / 2.0;
  double occlusion_min_elevation_rad_ = -kPi / 2.0;
  double occlusion_max_elevation_rad_ = kPi / 2.0;
  double occlusion_return_range_m_ = 1.8;
  double occlusion_return_sigma_m_ = 0.005;

  std::string occlusion_multi_center_azimuth_deg_raw_;
  std::string occlusion_multi_width_deg_raw_;
  std::string occlusion_multi_range_m_raw_;
  std::vector<OcclusionSector> occlusion_sectors_;

  double fog_patch_probability_ = 0.0;
  double fog_patch_cell_deg_ = 15.0;
  double fog_patch_min_range_m_ = 0.3;
  double fog_patch_max_range_m_ = 2.5;
  double fog_patch_angular_jitter_deg_ = 3.0;

  std::mt19937_64 random_engine_;
  std::uniform_real_distribution<double> uniform_;
  std::ofstream audit_csv_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "livox_disturbance");
  try {
    LivoxCustomMsgDisturbanceNode node;
    ros::spin();
  } catch (const std::exception& exception) {
    ROS_FATAL_STREAM("Livox disturbance relay failed: " << exception.what());
    return 1;
  }
  return 0;
}
