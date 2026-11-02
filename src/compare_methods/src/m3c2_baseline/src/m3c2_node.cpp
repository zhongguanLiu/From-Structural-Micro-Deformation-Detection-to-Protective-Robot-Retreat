// Adapter only: the imported reference, fusion and M3C2 implementations are unchanged.
#include "baseline/core/reference_manager.hpp"
#include "baseline/core/temporal_window_fusion.hpp"
#include "baseline/m3c2/m3c2_baseline.hpp"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <boost/filesystem.hpp>
#include <zlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
// Exclusive creation prevents accidental replacement of a completed experiment.
class GzipCsv {
 public:
  void Open(const std::string& path, const std::string& header) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) throw std::runtime_error("Cannot exclusively create " + path);
    file_ = gzdopen(fd, "wb1");
    if (!file_) { ::close(fd); throw std::runtime_error("Cannot compress " + path); }
    Write(header);
  }
  void Write(const std::string& text) {
    if (gzwrite(file_, text.data(), static_cast<unsigned>(text.size())) !=
        static_cast<int>(text.size())) throw std::runtime_error("M3C2 recording failed");
  }
  ~GzipCsv() { if (file_) gzclose(file_); }
 private:
  gzFile file_ = nullptr;
};

class Node {
 public:
  Node() : private_("~") {
    std::string topic, output;
    private_.param<std::string>("input_topic", topic, "/cloud_registered");
    if (!private_.getParam("output_directory", output) || output.empty())
      throw std::runtime_error("output_directory is required");
    baseline::ReferenceConfig reference;
    private_.param("reference/init_frames", reference.init_frames, 30);
    private_.param("reference/voxel_size", reference.voxel_size, 0.05);
    private_.param("temporal/window_frames", temporal_.window_frames, 5);
    private_.param("temporal/voxel_size", temporal_.voxel_size, 0.05);
    private_.param("m3c2/core_voxel_size", config_.core_voxel_size, 0.05);
    private_.param("m3c2/normal_radius", config_.normal_radius, 0.12);
    private_.param("m3c2/cylinder_radius", config_.cylinder_radius, 0.10);
    private_.param("m3c2/cylinder_half_length", config_.cylinder_half_length, 0.20);
    private_.param("m3c2/min_support", config_.min_support, 5);
    private_.param("m3c2/registration_stddev", config_.registration_stddev, 0.002);
    private_.param("m3c2/cluster/tolerance", config_.cluster.tolerance, 0.12);
    private_.param("m3c2/cluster/min_cluster_size", config_.cluster.min_cluster_size, 3);
    private_.param("m3c2/cluster/max_cluster_size", config_.cluster.max_cluster_size, 100000);
    for (double value : {reference.voxel_size, temporal_.voxel_size,
         config_.core_voxel_size, config_.normal_radius, config_.cylinder_radius,
         config_.cylinder_half_length, config_.cluster.tolerance})
      if (!std::isfinite(value) || value <= 0) throw std::runtime_error("Invalid spatial parameter");
    if (reference.init_frames < 2 || temporal_.window_frames < 1 || config_.min_support < 3 ||
        !std::isfinite(config_.registration_stddev) || config_.registration_stddev < 0 ||
        config_.cluster.min_cluster_size < 1 ||
        config_.cluster.max_cluster_size < config_.cluster.min_cluster_size)
      throw std::runtime_error("Invalid M3C2 configuration");
    reference_.SetConfig(reference); fusion_.SetConfig(temporal_); method_.SetConfig(config_);
    boost::filesystem::create_directories(output);
    cores_.Open(output + "/reference_cores.csv.gz",
      "core_id,x,y,z,nx,ny,nz,reference_support,reference_first_source_s,reference_last_source_s\n");
    measurements_.Open(output + "/measurements.csv.gz",
      "frame_index,core_id,signed_distance_m,lod95_m,query_support,valid,significant,cluster_id\n");
    frames_.Open(output + "/frames.csv.gz",
      "frame_index,source_stamp_s,window_first_source_s,window_last_source_s,window_frames,"
      "input_points,fused_points,core_count,valid_count,significant_count,processing_time_ms,"
      "fusion_time_ms,method_time_ms,recording_time_ms\n");
    dynamic_ = nh_.advertise<sensor_msgs::PointCloud2>("/compare/m3c2/dynamic_points", 5);
    runtime_ = nh_.advertise<std_msgs::Float64MultiArray>("/compare/m3c2/runtime", 20);
    ready_ = nh_.advertise<std_msgs::Bool>("/compare/m3c2/reference_ready", 1, true);
    subscriber_ = nh_.subscribe(topic, 2, &Node::Receive, this);
  }
 private:
  void Receive(const sensor_msgs::PointCloud2ConstPtr& message) {
    try { Process(message); }
    catch (const std::exception& error) { ROS_FATAL_STREAM(error.what()); ros::shutdown(); }
  }
  void Process(const sensor_msgs::PointCloud2ConstPtr& message) {
    const auto start = Clock::now();
    const double stamp = message->header.stamp.toSec();
    if (message->header.frame_id != "camera_init" || !std::isfinite(stamp) || stamp <= last_stamp_)
      throw std::runtime_error("Invalid M3C2 input frame or nonmonotonic source timestamp");
    last_stamp_ = stamp;
    baseline::PointCloudPtr cloud(new baseline::PointCloud);
    pcl::fromROSMsg(*message, *cloud);
    cloud->is_dense = false;
    std::vector<int> finite_indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, finite_indices);
    if (cloud->empty()) return;
    if (!reference_.IsFrozen()) {
      if (reference_first_ < 0) reference_first_ = stamp;
      reference_last_ = stamp;
      reference_.AddCloud(cloud);
      return;
    }
    fusion_.PushCloud(cloud, stamp); stamps_.push_back(stamp);
    while (stamps_.size() > static_cast<size_t>(temporal_.window_frames)) stamps_.pop_front();
    if (stamps_.size() < static_cast<size_t>(temporal_.window_frames)) return;
    const auto fusion_start = Clock::now();
    const auto fused = fusion_.BuildFusedCloud();
    const double fusion_ms = Milliseconds(fusion_start);
    const auto method_start = Clock::now();
    const auto result = method_.Run(reference_.GetReferenceCloud(), fused);
    const double method_ms = Milliseconds(method_start);
    baseline::PointCloud significant;
    size_t valid_count = 0;
    for (size_t i = 0; i < result.core_records.size(); ++i) {
      const auto& r = result.core_records[i];
      const bool valid = r.reference_support >= config_.min_support &&
        r.query_support >= config_.min_support && std::isfinite(r.signed_distance) &&
        std::isfinite(r.lod95) && r.lod95 >= 0;
      valid_count += valid;
      if (valid && r.significant) {
        baseline::PointT p;
        // Endpoint estimate for the common dynamic-point evaluation adapter.
        p.x = r.core_point.x + r.normal.x * r.signed_distance;
        p.y = r.core_point.y + r.normal.y * r.signed_distance;
        p.z = r.core_point.z + r.normal.z * r.signed_distance;
        p.intensity = r.lod95 > 0 ? std::abs(r.signed_distance) / r.lod95 : 0;
        significant.push_back(p);
      }
    }
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(significant, output); output.header = message->header;
    dynamic_.publish(output);
    const double processing_ms = Milliseconds(start);
    const auto io_start = Clock::now();
    std::ostringstream samples, cores;
    samples << std::setprecision(17); cores << std::setprecision(17);
    for (size_t i = 0; i < result.core_records.size(); ++i) {
      const auto& r = result.core_records[i];
      const bool valid = r.reference_support >= config_.min_support &&
        r.query_support >= config_.min_support && std::isfinite(r.signed_distance) &&
        std::isfinite(r.lod95) && r.lod95 >= 0;
      if (cycle_ == 0)
        cores << i << ',' << r.core_point.x << ',' << r.core_point.y << ',' << r.core_point.z
          << ',' << r.normal.x << ',' << r.normal.y << ',' << r.normal.z << ','
          << r.reference_support << ',' << reference_first_ << ',' << reference_last_ << '\n';
      samples << cycle_ << ',' << i << ',';
      if (valid) samples << r.signed_distance;
      samples << ','; if (valid) samples << r.lod95;
      samples << ',' << r.query_support << ',' << valid << ',' << (valid && r.significant)
        << ',' << r.cluster_id << '\n';
    }
    if (cycle_ == 0) cores_.Write(cores.str());
    measurements_.Write(samples.str());
    const double recording_ms = Milliseconds(io_start);
    std::ostringstream frame;
    frame << std::setprecision(17) << cycle_ << ',' << stamp << ',' << stamps_.front() << ','
      << stamps_.back() << ',' << stamps_.size() << ',' << cloud->size() << ',' << fused->size()
      << ',' << result.core_records.size() << ',' << valid_count << ',' << significant.size()
      << ',' << processing_ms << ',' << fusion_ms << ',' << method_ms << ',' << recording_ms << '\n';
    frames_.Write(frame.str());
    std_msgs::Float64MultiArray timing;
    timing.data = {stamp, static_cast<double>(cycle_), static_cast<double>(cloud->size()),
      static_cast<double>(significant.size()), processing_ms};
    runtime_.publish(timing);
    if (cycle_++ == 0) {
      std_msgs::Bool ready; ready.data = true; ready_.publish(ready);
      ROS_INFO("M3C2_REFERENCE_READY cores=%zu reference=[%.3f, %.3f]", result.core_records.size(),
        reference_first_, reference_last_);
    }
  }
  ros::NodeHandle nh_, private_;
  ros::Subscriber subscriber_;
  ros::Publisher dynamic_, runtime_, ready_;
  baseline::ReferenceManager reference_;
  baseline::TemporalWindowFusion fusion_;
  baseline::TemporalFusionConfig temporal_;
  baseline::M3C2Baseline method_;
  baseline::M3C2Config config_;
  GzipCsv cores_, measurements_, frames_;
  std::deque<double> stamps_;
  size_t cycle_ = 0;
  double reference_first_ = -1, reference_last_ = -1, last_stamp_ = -1;
};
}
int main(int argc, char** argv) {
  ros::init(argc, argv, "m3c2_baseline");
  try { Node node; ros::spin(); }
  catch (const std::exception& error) { ROS_FATAL_STREAM(error.what()); return 1; }
  return 0;
}
