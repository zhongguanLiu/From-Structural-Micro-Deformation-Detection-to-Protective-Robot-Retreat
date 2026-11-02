//
// Created by lfc on 2021/2/28.
//

#include "livox_laser_simulation/livox_points_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <std_msgs/UInt64.h>
#include <ignition/math/Vector3.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/MultiRayShape.hh>
#include <gazebo/physics/PhysicsEngine.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/sensors/RaySensor.hh>
#include <gazebo/transport/Node.hh>
#include <ignition/math/Vector3.hh>
#include <livox_ros_driver/CustomMsg.h>
#include <limits>
#include "livox_laser_simulation/custommsg_point_writer.hpp"
#include "livox_laser_simulation/csv_reader.hpp"
#include "livox_laser_simulation/livox_ode_multiray_shape.h"
#include "livox_laser_simulation/livox_point_xyzrtl.h"

namespace gazebo {

GZ_REGISTER_SENSOR_PLUGIN(LivoxPointsPlugin)

LivoxPointsPlugin::LivoxPointsPlugin() {}

LivoxPointsPlugin::~LivoxPointsPlugin() {
    // RayPlugin disconnects its scan callback in the base destructor, which
    // runs after this class's members are destroyed. Stop the sensor first so
    // no callback can race those member destructors during gzserver shutdown.
    shuttingDown.store(true);
    if (raySensor) {
        raySensor->SetActive(false);
    }
    // Unregister before destroying any callback-owned state.  TriggerPair
    // executes under the registry mutex, so this also waits for an in-flight
    // shared acquisition without leaving a dangling plugin pointer.
    if (synchronizationToken != 0U) {
        livox_laser_simulation::GlobalDualLidarTriggerRegistry().Unregister(
            synchronizationToken);
        synchronizationToken = 0U;
    }
    std::lock_guard<std::mutex> scanLock(scanMutex);
    rosPointPub.shutdown();
    readyPub.shutdown();
    schedulerCatchUpSuppressedPub.shutdown();
    tfBroadcaster.reset();
    rosNode.reset();
    scanPub.reset();
    node.reset();
    rayShape.reset();
    laserCollision.reset();
    parentEntity.reset();
    raySensor.reset();
}

void convertDataToRotateInfo(const std::vector<std::vector<double>> &datas, std::vector<AviaRotateInfo> &avia_infos) {
    avia_infos.reserve(datas.size());
    double deg_2_rad = M_PI / 180.0;
    for (int i = 0; i < datas.size(); ++i) {
        auto &data = datas[i];
        if (data.size() == 3) {
            avia_infos.emplace_back();
            avia_infos.back().time = data[0];
            auto azimuth = data[1] * deg_2_rad;
            azimuth = std::remainder(azimuth, 2.0 * M_PI);
            if (azimuth <= -M_PI) {
                azimuth += 2.0 * M_PI;
            } else if (azimuth > M_PI) {
                azimuth -= 2.0 * M_PI;
            }
            avia_infos.back().azimuth = azimuth;
            avia_infos.back().zenith = data[2] * deg_2_rad - M_PI_2;
            avia_infos.back().line = i % 4;
        } else {
            ROS_INFO_STREAM("data size is not 3!");
        }
    }
}

void LivoxPointsPlugin::Load(gazebo::sensors::SensorPtr _parent, sdf::ElementPtr sdf) {
    std::vector<std::vector<double>> datas;
    std::string file_name = sdf->Get<std::string>("csv_file_name");

    std::string filePath(__FILE__);
    size_t found = filePath.find_last_of("/\\");
    file_name = std::string(filePath.substr(0, found)) + "/../scan_mode/" + file_name;

    ROS_INFO_STREAM("load csv file name:" << file_name);
    if (!CsvReader::ReadCsvFile(file_name, datas)) {
        ROS_INFO_STREAM("cannot get csv file!" << file_name << "will return !");
        return;
    }
    
    sdfPtr = sdf;
    auto rayElem = sdfPtr->GetElement("ray");
    auto scanElem = rayElem->GetElement("scan");
    auto rangeElem = rayElem->GetElement("range");

    int argc = 0;
    char **argv = nullptr;
    auto curr_scan_topic = sdf->Get<std::string>("ros_topic");
    frameName = sdf->Get<std::string>("frameName");
    ROS_INFO_STREAM("ros topic name:" << curr_scan_topic);
    ROS_INFO_STREAM("ros frame id: "<<frameName);

    raySensor = _parent;
    auto sensor_pose = raySensor->Pose();
    SendRosTf(sensor_pose, raySensor->ParentName(), raySensor->Name());

    node = transport::NodePtr(new transport::Node());
    node->Init(raySensor->WorldName());
    scanPub = node->Advertise<msgs::LaserScanStamped>(_parent->Topic(), 50);
    aviaInfos.clear();
    convertDataToRotateInfo(datas, aviaInfos);
    ROS_INFO_STREAM("scan info size:" << aviaInfos.size());
    maxPointSize = aviaInfos.size();

    RayPlugin::Load(_parent, sdfPtr);
    laserMsg.mutable_scan()->set_frame(_parent->ParentName());
    parentEntity = world->EntityByName(_parent->ParentName());
    auto physics = world->Physics();
    laserCollision = physics->CreateCollision("multiray", _parent->ParentName());
    laserCollision->SetName("ray_sensor_collision");
    laserCollision->SetRelativePose(_parent->Pose());
    laserCollision->SetInitialRelativePose(_parent->Pose());
    rayShape.reset(new gazebo::physics::LivoxOdeMultiRayShape(laserCollision));
    laserCollision->SetShape(rayShape);
    samplesStep = sdfPtr->Get<int>("samples");
    downSample = sdfPtr->Get<int>("downsample");
    if (downSample < 1) {
        downSample = 1;
    }
    ROS_INFO_STREAM("sample:" << samplesStep);
    ROS_INFO_STREAM("downsample:" << downSample);

    // Backward compatibility is deliberate: worlds without this element keep
    // the historical synthetic rolling offsets.  ICRA2027 explicitly selects
    // snapshot because all rays in one Gazebo update are evaluated at one
    // simulation pose.
    std::string point_time_mode = "legacy_rolling";
    if (sdfPtr->HasElement("point_time_mode")) {
        point_time_mode = sdfPtr->Get<std::string>("point_time_mode");
    }
    if (!livox_laser_simulation::ParsePointTimeMode(point_time_mode,
                                                     &pointTimeMode)) {
        ROS_ERROR_STREAM("invalid point_time_mode '" << point_time_mode
                         << "'; expected snapshot or legacy_rolling");
        return;
    }
    ROS_INFO_STREAM("point_time_mode: "
                    << livox_laser_simulation::PointTimeModeName(pointTimeMode));

    const bool has_synchronization_group =
        sdfPtr->HasElement("synchronization_group");
    const bool has_synchronization_role =
        sdfPtr->HasElement("synchronization_role");
    if (has_synchronization_group != has_synchronization_role) {
        ROS_ERROR_STREAM("synchronization_group and synchronization_role must "
                         "be configured together");
        raySensor->SetActive(false);
        return;
    }
    if (has_synchronization_group) {
        synchronizationGroup =
            sdfPtr->Get<std::string>("synchronization_group");
        const std::string role =
            sdfPtr->Get<std::string>("synchronization_role");
        if (synchronizationGroup.empty() ||
            !livox_laser_simulation::ParseDualLidarTriggerRole(
                role, &synchronizationRole)) {
            ROS_ERROR_STREAM("invalid synchronized dual-LiDAR configuration: "
                             "group must be non-empty and role must be primary "
                             "or secondary");
            raySensor->SetActive(false);
            return;
        }
        synchronizedAcquisition = true;
    }

    publishPointCloudType = sdfPtr->Get<int>("publish_pointcloud_type");
    ROS_INFO_STREAM("publish_pointcloud_type: " << publishPointCloudType);
    ros::init(argc, argv, curr_scan_topic);
    rosNode.reset(new ros::NodeHandle);
    switch (publishPointCloudType) {
        case SENSOR_MSG_POINT_CLOUD:
            rosPointPub = rosNode->advertise<sensor_msgs::PointCloud>(curr_scan_topic, 5);
            break;
        case SENSOR_MSG_POINT_CLOUD2_POINTXYZ:
        case SENSOR_MSG_POINT_CLOUD2_LIVOXPOINTXYZRTLT:
            rosPointPub = rosNode->advertise<sensor_msgs::PointCloud2>(curr_scan_topic, 5);
            break;
        case livox_laser_simulation_CUSTOM_MSG:
            rosPointPub = rosNode->advertise<livox_ros_driver::CustomMsg>(curr_scan_topic, 5);
            break;
        default:
            break;
    }

    visualize = sdfPtr->Get<bool>("visualize");

    rayShape->RayShapes().reserve(samplesStep / downSample);
    rayShape->Load(sdfPtr);
    rayShape->Init();
    minDist = rangeElem->Get<double>("min");
    maxDist = rangeElem->Get<double>("max");
    auto offset = laserCollision->RelativePose();
    ignition::math::Vector3d start_point, end_point;
    for (int j = 0; j < samplesStep; j += downSample) {
        int index = j % maxPointSize;
        auto &rotate_info = aviaInfos[index];
        ignition::math::Quaterniond ray;
        ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
        auto axis = offset.Rot() * ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
        start_point = minDist * axis + offset.Pos() - minDist * axis;
        end_point = maxDist * axis + offset.Pos();
        rayShape->AddRay(start_point, end_point);
    }
    if (synchronizedAcquisition) {
        // Complete all role-specific scheduling state before registration.
        // Registration may make the group ready immediately when the peer was
        // loaded first, so publishing the token earlier would expose a race in
        // which the first primary callback still used the default zero gate.
        if (synchronizationRole ==
            livox_laser_simulation::DualLidarTriggerRole::SECONDARY) {
            // Only one Gazebo RaySensor owns the schedule.  This secondary's
            // real custom ray table is driven explicitly through the shared
            // registry by each accepted primary acquisition.
            raySensor->SetActive(false);
        } else {
            // Gazebo 11 exposes ResetLastUpdateTime(), but it resets to zero;
            // it cannot anchor a sensor to a saved world's current SimTime.
            // Derive the guard directly from the one authoritative RaySensor
            // rate and let the registry suppress only scheduler catch-up
            // callbacks before either ray table is evaluated or published.
            const double update_rate_hz = raySensor->UpdateRate();
            const long double interval_ns =
                1.0e9L / static_cast<long double>(update_rate_hz);
            if (!std::isfinite(update_rate_hz) || update_rate_hz <= 0.0 ||
                !std::isfinite(interval_ns) || interval_ns < 1.0L ||
                interval_ns > static_cast<long double>(
                    std::numeric_limits<uint64_t>::max())) {
                ROS_ERROR_STREAM(
                    "synchronized primary Livox requires a finite positive "
                    "RaySensor update rate with a representable period");
                raySensor->SetActive(false);
                return;
            }
            minimumSynchronizedAcquisitionIntervalNs =
                static_cast<uint64_t>(std::ceil(interval_ns));
            schedulerCatchUpSuppressedPub =
                rosNode->advertise<std_msgs::UInt64>(
                    curr_scan_topic +
                        "/scheduler_catch_up_suppressed_count",
                    1, true);
            std_msgs::UInt64 initial_suppression_count;
            initial_suppression_count.data = 0U;
            schedulerCatchUpSuppressedPub.publish(initial_suppression_count);
            ROS_INFO_STREAM(
                "synchronized primary Livox cadence gate="
                << minimumSynchronizedAcquisitionIntervalNs
                << " ns from RaySensor update_rate=" << update_rate_hz
                << " Hz");
        }
        std::string registration_error;
        const bool registered =
            livox_laser_simulation::GlobalDualLidarTriggerRegistry().Register(
                synchronizationGroup, synchronizationRole,
                [this](uint64_t acquisition_time_ns) {
                    return GenerateScanAtNanoseconds(acquisition_time_ns);
                },
                &synchronizationToken, &registration_error);
        if (!registered) {
            ROS_ERROR_STREAM("cannot register synchronized dual-LiDAR sensor: "
                             << registration_error);
            raySensor->SetActive(false);
            return;
        }
    }
    // Advertise readiness only after the complete ray table and collision
    // shape and synchronization registration exist.  The dual-LiDAR launch
    // consumes both latched readiness messages while physics is paused.
    readyPub = rosNode->advertise<std_msgs::Bool>(
        curr_scan_topic + "/ready", 1, true);
    std_msgs::Bool ready;
    ready.data = true;
    readyPub.publish(ready);
}

void LivoxPointsPlugin::OnNewLaserScans() {
    if (shuttingDown.load()) {
        return;
    }
    if (synchronizedAcquisition &&
        synchronizationRole ==
            livox_laser_simulation::DualLidarTriggerRole::SECONDARY) {
        ROS_ERROR_THROTTLE(
            1.0, "secondary synchronized Livox RaySensor updated independently");
        return;
    }
    // A sensor-manager initialization callback can occur while Gazebo is
    // still paused and before the rear plugin has finished registering.  It
    // is not a valid dual-LiDAR acquisition: no rear ray table exists at that
    // epoch.  Gate it before either ray-cast or ROS publication rather than
    // creating an honest-but-unpairable primary message.  Once both members
    // are registered every primary event is evaluated and counted normally.
    if (synchronizedAcquisition &&
        !livox_laser_simulation::GlobalDualLidarTriggerRegistry().GroupReady(
            synchronizationGroup)) {
        ROS_INFO_THROTTLE(
            1.0, "waiting for complete synchronized dual-Livox group");
        return;
    }
    common::Time measurementTime = raySensor->LastMeasurementTime();
    if (measurementTime.Double() <= 0.0) {
        measurementTime = world->SimTime();
    }
    if (synchronizedAcquisition) {
        if (measurementTime.sec < 0 || measurementTime.nsec < 0 ||
            measurementTime.nsec >= 1000000000) {
            ROS_ERROR_THROTTLE(
                1.0, "primary synchronized Livox produced an invalid epoch");
            return;
        }
        const uint64_t seconds = static_cast<uint64_t>(measurementTime.sec);
        if (seconds >
            (std::numeric_limits<uint64_t>::max() -
             static_cast<uint64_t>(measurementTime.nsec)) /
                1000000000ULL) {
            ROS_ERROR_THROTTLE(
                1.0, "primary synchronized Livox epoch overflowed nanoseconds");
            return;
        }
        const uint64_t acquisition_time_ns =
            seconds * 1000000000ULL +
            static_cast<uint64_t>(measurementTime.nsec);
        std::string trigger_error;
        uint64_t catch_up_suppressed_count = 0U;
        const auto result =
            livox_laser_simulation::GlobalDualLidarTriggerRegistry()
                .TriggerPairWithMinimumInterval(
                    synchronizationToken, acquisition_time_ns,
                    minimumSynchronizedAcquisitionIntervalNs,
                    &catch_up_suppressed_count, &trigger_error);
        if (result == livox_laser_simulation::DualLidarTriggerResult::
                          SCHEDULER_CATCH_UP_SUPPRESSED) {
            std_msgs::UInt64 message;
            message.data = catch_up_suppressed_count;
            schedulerCatchUpSuppressedPub.publish(message);
            ROS_WARN_THROTTLE(
                1.0,
                "suppressed Gazebo dual-Livox scheduler catch-up event "
                "before ray generation; cumulative count=%llu",
                static_cast<unsigned long long>(catch_up_suppressed_count));
        } else if (result ==
                   livox_laser_simulation::DualLidarTriggerResult::ERROR) {
            ROS_ERROR_THROTTLE(
                1.0, "synchronized dual Livox acquisition failed: %s",
                trigger_error.c_str());
        }
        return;
    }
    GenerateScanAt(measurementTime);
}

bool LivoxPointsPlugin::GenerateScanAtNanoseconds(
    uint64_t acquisition_time_ns) {
    const uint64_t seconds = acquisition_time_ns / 1000000000ULL;
    const uint64_t nanoseconds = acquisition_time_ns % 1000000000ULL;
    if (seconds > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    return GenerateScanAt(common::Time(
        static_cast<int32_t>(seconds), static_cast<int32_t>(nanoseconds)));
}

bool LivoxPointsPlugin::GenerateScanAt(
    const common::Time& acquisition_time) {
    std::lock_guard<std::mutex> scanLock(scanMutex);
    if (shuttingDown.load() || !rayShape || acquisition_time.sec < 0 ||
        acquisition_time.nsec < 0 || acquisition_time.nsec >= 1000000000) {
        return false;
    }
    if (static_cast<uint64_t>(acquisition_time.sec) >
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    scanStamp = ros::Time(static_cast<uint32_t>(acquisition_time.sec),
                          static_cast<uint32_t>(acquisition_time.nsec));
    std::vector<std::pair<int, AviaRotateInfo>> points_pair;
    InitializeRays(points_pair, rayShape);
    rayShape->Update();
    msgs::Set(laserMsg.mutable_time(), acquisition_time);

    switch (publishPointCloudType) {
        case SENSOR_MSG_POINT_CLOUD:
            PublishPointCloud(points_pair);
            break;
        case SENSOR_MSG_POINT_CLOUD2_POINTXYZ:
            PublishPointCloud2XYZ(points_pair);
            break;
        case SENSOR_MSG_POINT_CLOUD2_LIVOXPOINTXYZRTLT:
            PublishPointCloud2XYZRTLT(points_pair);
            break;
        case livox_laser_simulation_CUSTOM_MSG:
            PublishLivoxROSDriverCustomMsg(points_pair);
            break;
        default:
            return false;
    }
    return true;
}

void LivoxPointsPlugin::InitializeRays(std::vector<std::pair<int, AviaRotateInfo>> &points_pair,
                                       boost::shared_ptr<physics::LivoxOdeMultiRayShape> &ray_shape) {
    auto &rays = ray_shape->RayShapes();
    ignition::math::Vector3d start_point, end_point;
    ignition::math::Quaterniond ray;
    auto offset = laserCollision->RelativePose();
    int64_t end_index = currStartIndex + samplesStep;
    int ray_index = 0;
    auto ray_size = rays.size();
    points_pair.clear();
    points_pair.reserve(rays.size());
    for (int k = currStartIndex; k < end_index; k += downSample) {
        auto index = k % maxPointSize;
        auto &rotate_info = aviaInfos[index];
        ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
        auto axis = offset.Rot() * ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
        start_point = minDist * axis + offset.Pos() - minDist * axis;
        end_point = maxDist * axis + offset.Pos();
        if (ray_index < ray_size) {
            rays[ray_index]->SetPoints(start_point, end_point);
            points_pair.emplace_back(ray_index, rotate_info);
        }
        ray_index++;
    }
    currStartIndex += samplesStep;
    if (currStartIndex > maxPointSize) {
        currStartIndex -= maxPointSize;
    }
}

void LivoxPointsPlugin::InitializeScan(msgs::LaserScan *&scan) {
    // Store the latest laser scans into laserMsg
    msgs::Set(scan->mutable_world_pose(), raySensor->Pose() + parentEntity->WorldPose());
    scan->set_angle_min(AngleMin().Radian());
    scan->set_angle_max(AngleMax().Radian());
    scan->set_angle_step(AngleResolution());
    scan->set_count(RangeCount());

    scan->set_vertical_angle_min(VerticalAngleMin().Radian());
    scan->set_vertical_angle_max(VerticalAngleMax().Radian());
    scan->set_vertical_angle_step(VerticalAngleResolution());
    scan->set_vertical_count(VerticalRangeCount());

    scan->set_range_min(RangeMin());
    scan->set_range_max(RangeMax());

    scan->clear_ranges();
    scan->clear_intensities();

    unsigned int rangeCount = RangeCount();
    unsigned int verticalRangeCount = VerticalRangeCount();

    for (unsigned int j = 0; j < verticalRangeCount; ++j) {
        for (unsigned int i = 0; i < rangeCount; ++i) {
            scan->add_ranges(0);
            scan->add_intensities(0);
        }
    }
}

ignition::math::Angle LivoxPointsPlugin::AngleMin() const {
    if (rayShape)
        return rayShape->MinAngle();
    else
        return -1;
}

ignition::math::Angle LivoxPointsPlugin::AngleMax() const {
    if (rayShape) {
        return ignition::math::Angle(rayShape->MaxAngle().Radian());
    } else
        return -1;
}

double LivoxPointsPlugin::GetRangeMin() const { return RangeMin(); }

double LivoxPointsPlugin::RangeMin() const {
    if (rayShape)
        return rayShape->GetMinRange();
    else
        return -1;
}

double LivoxPointsPlugin::GetRangeMax() const { return RangeMax(); }

double LivoxPointsPlugin::RangeMax() const {
    if (rayShape)
        return rayShape->GetMaxRange();
    else
        return -1;
}

double LivoxPointsPlugin::GetAngleResolution() const { return AngleResolution(); }

double LivoxPointsPlugin::AngleResolution() const { return (AngleMax() - AngleMin()).Radian() / (RangeCount() - 1); }

double LivoxPointsPlugin::GetRangeResolution() const { return RangeResolution(); }

double LivoxPointsPlugin::RangeResolution() const {
    if (rayShape)
        return rayShape->GetResRange();
    else
        return -1;
}

int LivoxPointsPlugin::GetRayCount() const { return RayCount(); }

int LivoxPointsPlugin::RayCount() const {
    if (rayShape)
        return rayShape->GetSampleCount();
    else
        return -1;
}

int LivoxPointsPlugin::GetRangeCount() const { return RangeCount(); }

int LivoxPointsPlugin::RangeCount() const {
    if (rayShape)
        return rayShape->GetSampleCount() * rayShape->GetScanResolution();
    else
        return -1;
}

int LivoxPointsPlugin::GetVerticalRayCount() const { return VerticalRayCount(); }

int LivoxPointsPlugin::VerticalRayCount() const {
    if (rayShape)
        return rayShape->GetVerticalSampleCount();
    else
        return -1;
}

int LivoxPointsPlugin::GetVerticalRangeCount() const { return VerticalRangeCount(); }

int LivoxPointsPlugin::VerticalRangeCount() const {
    if (rayShape)
        return rayShape->GetVerticalSampleCount() * rayShape->GetVerticalScanResolution();
    else
        return -1;
}

ignition::math::Angle LivoxPointsPlugin::VerticalAngleMin() const {
    if (rayShape) {
        return ignition::math::Angle(rayShape->VerticalMinAngle().Radian());
    } else
        return -1;
}

ignition::math::Angle LivoxPointsPlugin::VerticalAngleMax() const {
    if (rayShape) {
        return ignition::math::Angle(rayShape->VerticalMaxAngle().Radian());
    } else
        return -1;
}

double LivoxPointsPlugin::GetVerticalAngleResolution() const { return VerticalAngleResolution(); }

double LivoxPointsPlugin::VerticalAngleResolution() const {
    return (VerticalAngleMax() - VerticalAngleMin()).Radian() / (VerticalRangeCount() - 1);
}
void LivoxPointsPlugin::SendRosTf(const ignition::math::Pose3d &pose, const std::string &father_frame,
                                  const std::string &child_frame) {
    if (!tfBroadcaster) {
        tfBroadcaster.reset(new tf::TransformBroadcaster);
    }
    tf::Transform tf;
    auto rot = pose.Rot();
    auto pos = pose.Pos();
    tf.setRotation(tf::Quaternion(rot.X(), rot.Y(), rot.Z(), rot.W()));
    tf.setOrigin(tf::Vector3(pos.X(), pos.Y(), pos.Z()));
    tfBroadcaster->sendTransform(
        tf::StampedTransform(tf, ros::Time::now(), raySensor->ParentName(), raySensor->Name()));
}

void LivoxPointsPlugin::PublishPointCloud(std::vector<std::pair<int, AviaRotateInfo>> &points_pair) {
    auto rayCount = RayCount();
    auto verticalRayCount = VerticalRayCount();
    auto angle_min = AngleMin().Radian();
    auto angle_incre = AngleResolution();
    auto verticle_min = VerticalAngleMin().Radian();
    auto verticle_incre = VerticalAngleResolution();

    msgs::LaserScan *scan = laserMsg.mutable_scan();
    InitializeScan(scan);
    // SendRosTf(parentEntity->WorldPose(), world->Name(), raySensor->ParentName());

    sensor_msgs::PointCloud scan_point;
    scan_point.header.stamp = scanStamp;
    scan_point.header.frame_id = frameName;
    auto &scan_points = scan_point.points;
    for (auto &pair : points_pair) {
        int verticle_index = roundf((pair.second.zenith - verticle_min) / verticle_incre);
        int horizon_index = roundf((pair.second.azimuth - angle_min) / angle_incre);
        if (verticle_index < 0 || horizon_index < 0) {
            continue;
        }
        if (verticle_index < verticalRayCount && horizon_index < rayCount) {
            auto index = (verticalRayCount - verticle_index - 1) * rayCount + horizon_index;
            auto range = rayShape->GetRange(pair.first);
            auto intensity = rayShape->GetRetro(pair.first);
            if (range >= maxDist || range <= minDist || range <= 1e-5) {
                range = 0.0;
            }
            scan->set_ranges(index, range);
            scan->set_intensities(index, intensity);

            auto rotate_info = pair.second;
            ignition::math::Quaterniond ray;
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));

            auto axis = ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            auto point = range * axis;
            scan_points.emplace_back();
            scan_points.back().x = point.X();
            scan_points.back().y = point.Y();
            scan_points.back().z = point.Z();
        }
    }
    rosPointPub.publish(scan_point);
    ros::spinOnce();
    if (scanPub && scanPub->HasConnections() && visualize) {
        scanPub->Publish(laserMsg);
    }
}

void LivoxPointsPlugin::PublishPointCloud2XYZ(std::vector<std::pair<int, AviaRotateInfo>> &points_pair) {
    auto rayCount = RayCount();
    auto verticalRayCount = VerticalRayCount();
    auto angle_min = AngleMin().Radian();
    auto angle_incre = AngleResolution();
    auto verticle_min = VerticalAngleMin().Radian();
    auto verticle_incre = VerticalAngleResolution();

    msgs::LaserScan *scan = laserMsg.mutable_scan();
    InitializeScan(scan);

    sensor_msgs::PointCloud2 scan_point;

    pcl::PointCloud<pcl::PointXYZ> pc;
    pc.points.resize(points_pair.size());
    const ros::Time timestamp = scanStamp;
    int pt_count = 0;
#pragma omp parallel for
    for (int i = 0; i < points_pair.size(); ++i) {
        std::pair<int, gazebo::AviaRotateInfo> &pair = points_pair[i];
        int verticle_index = roundf((pair.second.zenith - verticle_min) / verticle_incre);
        int horizon_index = roundf((pair.second.azimuth - angle_min) / angle_incre);
        if (verticle_index < 0 || horizon_index < 0) {
            continue;
        }
        if (verticle_index < verticalRayCount && horizon_index < rayCount) {
            auto index = (verticalRayCount - verticle_index - 1) * rayCount + horizon_index;
            auto range = rayShape->GetRange(pair.first);
            auto intensity = rayShape->GetRetro(pair.first);
            if (range >= maxDist || range <= minDist || range <= 1e-5){
                range = 0.0;
            }

            scan->set_ranges(index, range);
            scan->set_intensities(index, intensity);

            auto rotate_info = pair.second;
            ignition::math::Quaterniond ray;
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
            //                auto axis = rotate * ray * math::Vector3(1.0, 0.0, 0.0);
            //                auto point = range * axis + world_pose.Pos();
            auto axis = ray * ignition::math::Vector3d(1.0, 0.0, 0.0);

            // if (range < 0.3) {
            //     ROS_WARN_STREAM("Small pt: range: " << range << ", axis: " << axis);
            // }
            auto point = range * axis;
            pcl::PointXYZ pt;
            pt.x = point.X();
            pt.y = point.Y();
            pt.z = point.Z();
            // if (pt_count < pc.size() && pt_count > 0)
#pragma omp critical
            {
                pc[pt_count] = pt;
                ++pt_count;
            }
        }
    }
    pc.resize(pt_count);
    pcl::toROSMsg(pc, scan_point);
    scan_point.header.stamp = timestamp;
    scan_point.header.frame_id = frameName;
    rosPointPub.publish(scan_point);
    // SendRosTf(parentEntity->WorldPose(), world->Name(), raySensor->ParentName());
    ros::spinOnce();
    if (scanPub && scanPub->HasConnections() && visualize) {
        scanPub->Publish(laserMsg);
    }
}

void LivoxPointsPlugin::PublishPointCloud2XYZRTLT(std::vector<std::pair<int, AviaRotateInfo>> &points_pair) {
    auto rayCount = RayCount();
    auto verticalRayCount = VerticalRayCount();
    auto angle_min = AngleMin().Radian();
    auto angle_incre = AngleResolution();
    auto verticle_min = VerticalAngleMin().Radian();
    auto verticle_incre = VerticalAngleResolution();

    msgs::LaserScan *scan = laserMsg.mutable_scan();
    InitializeScan(scan);
    // SendRosTf(parentEntity->WorldPose(), world->Name(), raySensor->ParentName());

    sensor_msgs::PointCloud2 scan_point;

    pcl::PointCloud<pcl::LivoxPointXyzrtlt> pc;
    pc.points.reserve(points_pair.size());
    const ros::Time header_timestamp = scanStamp;
    auto header_timestamp_sec_nsec = header_timestamp.toNSec();

    
    // auto start = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    for (int i = 0; i < points_pair.size(); ++i) {
        std::pair<int, AviaRotateInfo> &pair = points_pair[i];
        int verticle_index = roundf((pair.second.zenith - verticle_min) / verticle_incre);
        int horizon_index = roundf((pair.second.azimuth - angle_min) / angle_incre);
        if (verticle_index < 0 || horizon_index < 0) {
            continue;
        }
        if (verticle_index < verticalRayCount && horizon_index < rayCount) {
            auto index = (verticalRayCount - verticle_index - 1) * rayCount + horizon_index;
            auto range = rayShape->GetRange(pair.first);
            auto intensity = rayShape->GetRetro(pair.first);
            if (range >= maxDist || range <= minDist|| abs(range) <= 1e-5) {
                range = 0.0;
            }
            scan->set_ranges(index, range);
            scan->set_intensities(index, intensity);

            auto rotate_info = pair.second;
            ignition::math::Quaterniond ray;
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
            //                auto axis = rotate * ray * math::Vector3(1.0, 0.0, 0.0);
            //                auto point = range * axis + world_pose.Pos();

            auto axis = ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            auto point = range * axis;
            pcl::LivoxPointXyzrtlt pt;
            
            pt.x = point.X();
            pt.y = point.Y();
            pt.z = point.Z();
            pt.intensity = static_cast<float>(intensity);
            pt.tag = 0;
            pt.line = pair.second.line;
            pt.timestamp = static_cast<double>(
                livox_laser_simulation::SimulatedPointOffsetNs(pointTimeMode, i)) +
                header_timestamp_sec_nsec;

            pc.push_back(std::move(pt));
        }
    }
    pcl::toROSMsg(pc, scan_point);
    scan_point.header.stamp = header_timestamp;
    scan_point.header.frame_id = frameName;
    rosPointPub.publish(scan_point);
    ros::spinOnce();
    if (scanPub && scanPub->HasConnections() && visualize) {
        scanPub->Publish(laserMsg);
    }
}

void LivoxPointsPlugin::PublishLivoxROSDriverCustomMsg(std::vector<std::pair<int, AviaRotateInfo>> &points_pair) {
    auto rayCount = RayCount();
    auto verticalRayCount = VerticalRayCount();
    auto angle_min = AngleMin().Radian();
    auto angle_incre = AngleResolution();
    auto verticle_min = VerticalAngleMin().Radian();
    auto verticle_incre = VerticalAngleResolution();

    // The protobuf LaserScan is consumed only by Gazebo ray visualization.
    // In formal runs visualize=false, so avoid allocating and clearing its
    // 5,000 range/intensity cells and avoid writing them for every Livox ray.
    // Angle-domain validation below remains unconditional and the scientific
    // CustomMsg construction is shared by both branches.
    const bool maintain_visualization_scan =
        visualize && scanPub && scanPub->HasConnections();
    msgs::LaserScan *scan = nullptr;
    if (maintain_visualization_scan) {
        scan = laserMsg.mutable_scan();
        InitializeScan(scan);
    }
    // SendRosTf(parentEntity->WorldPose(), world->Name(), raySensor->ParentName());

    livox_ros_driver::CustomMsg msg;
    msg.header.frame_id = frameName;
    msg.header.stamp = scanStamp;
    msg.timebase = msg.header.stamp.toNSec();
    msg.lidar_id = 0;
    msg.points.reserve(points_pair.size());
    for (int i = 0; i < points_pair.size(); ++i) {
        std::pair<int, AviaRotateInfo> &pair = points_pair[i];
        int verticle_index = roundf((pair.second.zenith - verticle_min) / verticle_incre);
        int horizon_index = roundf((pair.second.azimuth - angle_min) / angle_incre);
        if (verticle_index < 0 || horizon_index < 0) {
            continue;
        }
        if (verticle_index < verticalRayCount && horizon_index < rayCount) {
            auto index = (verticalRayCount - verticle_index - 1) * rayCount + horizon_index;
            auto range = rayShape->GetRange(pair.first);
            auto intensity = rayShape->GetRetro(pair.first);
            if (range >= maxDist || range <= minDist || abs(range) <= 1e-5) {
                range = 0.0;
            }
            auto rotate_info = pair.second;
            ignition::math::Quaterniond ray;
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
            //                auto axis = rotate * ray * math::Vector3(1.0, 0.0, 0.0);
            //                auto point = range * axis + world_pose.Pos(); Convert to world coordinate system

            auto axis = ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            auto point = range * axis;
            livox_laser_simulation::AppendCustomPointAndOptionalVisualization(
                point, range, intensity, pair.second.line,
                livox_laser_simulation::SimulatedPointOffsetNs(pointTimeMode, i),
                index, scan, &msg);
        }
    }
    // clock_gettime(CLOCK_REALTIME, &tn);
    // uint64_t interval = tn.tv_nsec - msg.timebase;
    // for (int i = 0; i < msg.points.size(); ++i) {
    //     msg.points[i].offset_time = (float)interval / msg.points.size() * i * 10;
    // }
    msg.point_num = msg.points.size();
    rosPointPub.publish(msg);
    ros::spinOnce();
    if (maintain_visualization_scan) {
        scanPub->Publish(laserMsg);
    }
}

}  // namespace gazebo
