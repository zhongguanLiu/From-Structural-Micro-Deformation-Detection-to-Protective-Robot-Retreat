#ifndef DEFORM_MONITOR_V3_V1_ROS_BRIDGE_HPP
#define DEFORM_MONITOR_V3_V1_ROS_BRIDGE_HPP

#include "deform_monitor_v3/V1DeformationField.h"
#include "deform_monitor_v3/V1DetectionTile.h"
#include "deform_monitor_v3/V1DetectionCommon.h"
#include "deform_monitor_v3/V1DetectionStatus.h"
#include "deform_monitor_v3/V1DeformationSpatialTile.h"
#include "deform_monitor_v3/V1InterfaceMotionFrame.h"
#include "deform_monitor_v3/V1OccupancyKnowledgeFrame.h"
#include "deform_monitor_v3/V1RiskAssessment.h"
#include "deform_monitor_v3/V1StructuralEventFrame.h"
#include "deform_monitor_v3/V1StableId128.h"
#include "deform_monitor_v3/V1SystemStatus.h"
#include "deform_monitor_v3/v1/pipeline.hpp"

namespace deform_monitor_v3 {
namespace v1 {

deform_monitor_v3::V1StableId128 BuildStableIdMessage(const StableId& id);
bool ParseStableIdMessage(const deform_monitor_v3::V1StableId128& message,
                          StableId* id);

deform_monitor_v3::V1Common BuildCommonMessage(const CommonHeader& common);
deform_monitor_v3::V1DetectionCommon BuildDetectionCommonMessage(
    const CommonHeader& common);
deform_monitor_v3::V1DeformationField BuildDeformationMessage(const V1Snapshot& snapshot);
deform_monitor_v3::V1InterfaceMotionFrame BuildInterfaceMessage(const V1Snapshot& snapshot);
deform_monitor_v3::V1StructuralEventFrame BuildStructuralEventMessage(const V1Snapshot& snapshot);
deform_monitor_v3::V1OccupancyKnowledgeFrame BuildOccupancyMessage(
    const V1Snapshot& snapshot, const V1Config& config);
std::vector<deform_monitor_v3::V1DeformationSpatialTile> BuildSpatialTileMessages(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const OccupancyEventGraph& occupancy, const V1Config& config);
std::vector<deform_monitor_v3::V1DetectionTile> BuildDetectionTileMessages(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const V1Config& config);
// Exact single-worker reference path retained solely for deterministic
// regression tests.  Production uses the bounded per-GDU parallel entry point
// above; both paths execute the same local builder and canonical merge.
std::vector<deform_monitor_v3::V1DetectionTile>
BuildDetectionTileMessagesSerialForTesting(
    const V1Snapshot& snapshot, const UacGaussianSumFilter& filter,
    const V1Config& config);
deform_monitor_v3::V1RiskAssessment BuildRiskAssessmentMessage(
    const RiskAssessment& assessment);
deform_monitor_v3::V1SystemStatus BuildStatusMessage(
    const V1Snapshot& snapshot, MonitoringPhase phase,
    uint64_t deadline_failure_count, uint64_t resource_failure_count);
deform_monitor_v3::V1DetectionStatus BuildDetectionStatusMessage(
    const V1Snapshot& snapshot, MonitoringPhase phase,
    uint64_t deadline_failure_count, uint64_t resource_failure_count);

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_ROS_BRIDGE_HPP
