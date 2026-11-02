//
// Created by lfc on 2021/2/28.
//

#ifndef SRC_GAZEBO_LIVOX_POINTS_PLUGIN_H
#define SRC_GAZEBO_LIVOX_POINTS_PLUGIN_H
#include <atomic>
#include <mutex>
#include <ros/node_handle.h>
#include <tf/transform_broadcaster.h>
#include <gazebo/plugins/RayPlugin.hh>
#include "livox_laser_simulation/dual_lidar_trigger.hpp"
#include "livox_ode_multiray_shape.h"
#include "livox_laser_simulation/point_time_model.hpp"

namespace gazebo {
struct AviaRotateInfo {
    double time;
    double azimuth;
    double zenith;
    uint8_t line;
};


class LivoxPointsPlugin : public RayPlugin {
 public:
    LivoxPointsPlugin();

    virtual ~LivoxPointsPlugin();

    void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

 private:
    ignition::math::Angle AngleMin() const;

    ignition::math::Angle AngleMax() const;

    double GetAngleResolution() const GAZEBO_DEPRECATED(7.0);

    double AngleResolution() const;

    double GetRangeMin() const GAZEBO_DEPRECATED(7.0);

    double RangeMin() const;

    double GetRangeMax() const GAZEBO_DEPRECATED(7.0);

    double RangeMax() const;

    double GetRangeResolution() const GAZEBO_DEPRECATED(7.0);

    double RangeResolution() const;

    int GetRayCount() const GAZEBO_DEPRECATED(7.0);

    int RayCount() const;

    int GetRangeCount() const GAZEBO_DEPRECATED(7.0);

    int RangeCount() const;

    int GetVerticalRayCount() const GAZEBO_DEPRECATED(7.0);

    int VerticalRayCount() const;

    int GetVerticalRangeCount() const GAZEBO_DEPRECATED(7.0);

    int VerticalRangeCount() const;

    ignition::math::Angle VerticalAngleMin() const;

    ignition::math::Angle VerticalAngleMax() const;

    double GetVerticalAngleResolution() const GAZEBO_DEPRECATED(7.0);

    double VerticalAngleResolution() const;

 protected:
    virtual void OnNewLaserScans();

 private:
    enum PointCloudType {
        SENSOR_MSG_POINT_CLOUD = 0,
        SENSOR_MSG_POINT_CLOUD2_POINTXYZ = 1,
        SENSOR_MSG_POINT_CLOUD2_LIVOXPOINTXYZRTLT = 2,
        livox_laser_simulation_CUSTOM_MSG = 3,
    };

    void InitializeRays(std::vector<std::pair<int, AviaRotateInfo>>& points_pair,
                        boost::shared_ptr<physics::LivoxOdeMultiRayShape>& ray_shape);

    // Evaluate and publish this sensor's custom ray table at the supplied
    // simulation acquisition epoch.  In synchronized dual-LiDAR mode the
    // primary callback calls this once for itself and once for the disabled
    // secondary RaySensor, so both scans are real ray casts at one epoch.
    bool GenerateScanAt(const common::Time& acquisition_time);
    bool GenerateScanAtNanoseconds(uint64_t acquisition_time_ns);

    void InitializeScan(msgs::LaserScan*& scan);

    void SendRosTf(const ignition::math::Pose3d& pose, const std::string& father_frame, const std::string& child_frame);

    void PublishPointCloud(std::vector<std::pair<int, AviaRotateInfo>>& points_pair);
    void PublishPointCloud2XYZ(std::vector<std::pair<int, AviaRotateInfo>>& points_pair);
    void PublishLivoxROSDriverCustomMsg(std::vector<std::pair<int, AviaRotateInfo>>& points_pair);
    void PublishPointCloud2XYZRTLT(std::vector<std::pair<int, AviaRotateInfo>>& points_pair);

    boost::shared_ptr<physics::LivoxOdeMultiRayShape> rayShape;
    gazebo::physics::CollisionPtr laserCollision;
    physics::EntityPtr parentEntity;
    transport::PublisherPtr scanPub;
    sdf::ElementPtr sdfPtr;
    msgs::LaserScanStamped laserMsg;
    // Timestamp of the sensor measurement that triggered the current scan.
    // This is deliberately not the wall/simulation time at which the costly
    // ray-cast finishes: two synchronised LiDARs may have different compute
    // latency while still observing the same measurement epoch.
    ros::Time scanStamp;
    transport::NodePtr node;
    gazebo::sensors::SensorPtr raySensor;
    std::vector<AviaRotateInfo> aviaInfos;

    std::shared_ptr<ros::NodeHandle> rosNode;
    ros::Publisher rosPointPub;
    ros::Publisher readyPub;
    // Latched, monotonic telemetry for scheduler callbacks suppressed before
    // ray generation because a saved-world catch-up event arrived earlier
    // than one configured sensor period after the last physical pair.
    ros::Publisher schedulerCatchUpSuppressedPub;
    std::shared_ptr<tf::TransformBroadcaster> tfBroadcaster;

    int64_t samplesStep = 0;
    int64_t currStartIndex = 0;
    int64_t maxPointSize = 1000;
    int64_t downSample = 1;
    uint16_t publishPointCloudType;
    livox_laser_simulation::PointTimeMode pointTimeMode =
        livox_laser_simulation::PointTimeMode::LEGACY_ROLLING;
    bool visualize = false;
    std::string frameName = "livox";

    std::string synchronizationGroup;
    livox_laser_simulation::DualLidarTriggerRole synchronizationRole =
        livox_laser_simulation::DualLidarTriggerRole::PRIMARY;
    uint64_t synchronizationToken = 0U;
    bool synchronizedAcquisition = false;
    uint64_t minimumSynchronizedAcquisitionIntervalNs = 0U;

    double maxDist = 400.0;
    double minDist = 0.1;

    bool useInf = true;
    std::atomic<bool> shuttingDown{false};
    std::mutex scanMutex;
};

}  // namespace gazebo

#endif  // SRC_GAZEBO_LIVOX_POINTS_PLUGIN_H
