cmake_minimum_required(VERSION 3.0.2)
project(deform_monitor_v3)

# The algorithm identity used by experiment metadata and analysis tools.
add_compile_definitions(ALERT_PLUSPLUS_NAME=\"alert_plusplus\")

add_compile_options(-std=c++14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(catkin REQUIRED COMPONENTS
  fast_lio
  livox_ros_driver
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  message_generation
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

add_message_files(
  FILES
  AnchorState.msg
  AnchorObjectSummary.msg
  AnchorStates.msg
  MotionCluster.msg
  MotionClusters.msg
  ObjectHitStat.msg
  ObjectObservationStats.msg
  RiskEvidence.msg
  RiskEvidenceArray.msg
  RiskVoxel.msg
  RiskVoxelField.msg
  RiskRegion.msg
  RiskRegions.msg
  PersistentRiskRegion.msg
  PersistentRiskRegions.msg
  ProbabilisticRiskVoxel.msg
  ProbabilisticRiskMap.msg
  ProbabilisticRiskRegion.msg
  ProbabilisticRiskRegions.msg
  StructureMotion.msg
  StructureMotions.msg
  V1StableId128.msg
  V1DenseMatrix64.msg
  V1SparseCsrMatrix64.msg
  V1TimedRobotPose.msg
  V1PathInvalidation.msg
  V1ObservationExplanation.msg
  V1Common.msg
  V1DetectionCommon.msg
  V1GaussianComponent.msg
  V1GduPosterior.msg
  V1DeformationField.msg
  V1InterfacePosterior.msg
  V1InterfaceMotionFrame.msg
  V1HazardGeometry.msg
  V1StructuralEventFrame.msg
  V1OccupancyCell.msg
  V1OccupancyEventLineage.msg
  V1OccupancyKnowledgeFrame.msg
  V1RiskAssessment.msg
  V1RiskConfigurationSample.msg
  V1SystemStatus.msg
  V1DetectionStatus.msg
  V1DeformationSample.msg
  V1DeformationComponentQuery.msg
  V1ReferenceSupportSample.msg
  V1SpatialDetectionEvidence.msg
  V1DetectionTile.msg
  V1DeformationSpatialTile.msg
  V1RiskCostFrame.msg
)

add_service_files(
  FILES
  SetV1MonitoringPhase.srv
  QueryV1StateRisk.srv
  AssessV1ExternalPath.srv
  QueryV1MotionRisk.srv
  RegisterV1ExternalPath.srv
  QueryV1Deformation.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    fast_lio
    livox_ros_driver
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    pcl_ros
    pcl_conversions
    message_runtime
  DEPENDS
    PCL
    EIGEN3
    Boost
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_install_python(PROGRAMS
  scripts/algorithm_frame_store.py
  scripts/analyze_anchor_continuity.py
  scripts/analyze_sim_run.py
  scripts/check_sim_object_associations.py
  scripts/compare_algorithm_storage.py
  scripts/real_experiment_recorder.py
  scripts/sim_experiment_recorder.py
  scripts/sim_truth_rviz_bridge.py
  scripts/v1_monitoring_cli.py
  scripts/v1_real_recorder.py
  scripts/validate_recorded_run.py
  scripts/validate_sim_experiment_setup.py
  tools/risk_map/record_risk_map_frames.py
  tools/risk_map/risk_map_io.py
  tools/risk_map/plot_risk_map_3d.py
  tools/risk_map/plot_risk_map_slices.py
  tools/risk_map/plot_risk_map_timeline.py
  tools/risk_map/render_risk_map_sequence.py
  tools/risk_map/validate_risk_map_run.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(${PROJECT_NAME}
  src/data_types.cpp
  src/core/object_id_association.cpp
  src/core/object_observation_stats.cpp
  src/core/covariance_extractor.cpp
  src/core/observable_subspace.cpp
  src/core/edge_feature.cpp
  src/core/anchor_builder.cpp
  src/core/current_observation_extractor.cpp
  src/core/scalar_measurement_builder.cpp
  src/core/imm_information_filter.cpp
  src/core/weak_plane_motion_detector.cpp
  src/core/motion_clusterer.cpp
  src/core/region_hypothesis_builder.cpp
  src/core/region_correspondence_solver.cpp
  src/core/risk_evidence_adapter.cpp
  src/core/risk_field_builder.cpp
  src/core/persistent_risk_region_tracker.cpp
  src/core/reference_manager.cpp
  src/core/structure_unit_builder.cpp
  src/core/structure_unit_tracker.cpp
  src/risk_map/channel_adapter.cpp
  src/risk_map/window_buffer.cpp
  src/risk_map/pose_covariance_model.cpp
  src/risk_map/deformation_evidence.cpp
  src/risk_map/reference_free_space.cpp
  src/risk_map/current_occupancy.cpp
  src/risk_map/visibility_model.cpp
  src/risk_map/match_evidence.cpp
  src/risk_map/emergence_evidence.cpp
  src/risk_map/exclusive_association.cpp
  src/risk_map/belief_map.cpp
  src/risk_map/hazard_projector.cpp
  src/risk_map/risk_map_pipeline.cpp
  src/risk_map/risk_map_publisher.cpp
  src/risk_map/risk_map_exporter.cpp
  src/v1/deterministic.cpp
  src/v1/math.cpp
  src/v1/config.cpp
  src/v1/input_contract.cpp
  src/v1/pose_adapter.cpp
  src/v1/reference_graph.cpp
  src/v1/association.cpp
  src/v1/uac_filter.cpp
  src/v1/occupancy.cpp
  src/v1/risk_query.cpp
  src/v1/pipeline.cpp
  src/v1/ros_bridge.cpp
  src/v1/ros_config.cpp
  src/v1/rviz_adapter.cpp
  src/risk_visualization_publisher.cpp
  src/structure_visualization_publisher.cpp
  src/visualization_publisher.cpp
)

add_dependencies(${PROJECT_NAME}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  yaml-cpp
)

add_executable(deform_monitor_v3_node
  src/deform_monitor_v3_node.cpp
  src/deform_monitor_v3_main.cpp
)

target_compile_definitions(deform_monitor_v3_node PRIVATE
  DEFORM_MONITOR_V3_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_dependencies(deform_monitor_v3_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(deform_monitor_v3_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  Threads::Threads
)

add_executable(deform_monitor_v1_node
  src/v1/node.cpp
  src/v1/main.cpp
)
add_dependencies(deform_monitor_v1_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(deform_monitor_v1_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  Threads::Threads
)

add_executable(deform_monitor_v1_rviz_node
  src/v1/rviz_main.cpp
)
add_dependencies(deform_monitor_v1_rviz_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(deform_monitor_v1_rviz_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_v1_config_yaml.py)
  catkin_add_nosetests(test/test_v1_monitoring_cli.py)
  catkin_add_nosetests(test/test_v1_real_recorder.py)

  catkin_add_gtest(test_v1_foundation test/test_v1_foundation.cpp)
  if(TARGET test_v1_foundation)
    target_link_libraries(test_v1_foundation ${PROJECT_NAME} ${catkin_LIBRARIES}
      ${PCL_LIBRARIES} ${Boost_LIBRARIES} Threads::Threads)
  endif()
  catkin_add_gtest(test_v1_rviz_adapter test/test_v1_rviz_adapter.cpp)
  if(TARGET test_v1_rviz_adapter)
    target_link_libraries(test_v1_rviz_adapter ${PROJECT_NAME}
      ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_risk_map_core test/test_risk_map_core.cpp)
  if(TARGET test_risk_map_core)
    target_link_libraries(test_risk_map_core ${PROJECT_NAME} ${catkin_LIBRARIES}
      ${PCL_LIBRARIES} ${Boost_LIBRARIES} Threads::Threads)
  endif()
  catkin_add_gtest(test_anchor_object_association
    test/test_anchor_object_association.cpp
  )

  if(TARGET test_anchor_object_association)
    target_link_libraries(test_anchor_object_association
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_edge_measurement_model
    test/test_edge_measurement_model.cpp
  )

  if(TARGET test_edge_measurement_model)
  target_link_libraries(test_edge_measurement_model
    ${PROJECT_NAME}
    gtest_main
    ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_anchor_reference_continuity
    test/test_anchor_reference_continuity.cpp
  )

  if(TARGET test_anchor_reference_continuity)
    target_link_libraries(test_anchor_reference_continuity
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_ablation_switches
    test/test_ablation_switches.cpp
    src/deform_monitor_v3_node.cpp
  )

  if(TARGET test_ablation_switches)
    target_link_libraries(test_ablation_switches
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_weak_plane_motion_detector
    test/test_weak_plane_motion_detector.cpp
  )

  if(TARGET test_weak_plane_motion_detector)
    target_link_libraries(test_weak_plane_motion_detector
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()

  catkin_add_gtest(test_persistent_risk_region_tracker
    test/test_persistent_risk_region_tracker.cpp
  )

  if(TARGET test_persistent_risk_region_tracker)
    target_link_libraries(test_persistent_risk_region_tracker
      ${PROJECT_NAME}
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${Boost_LIBRARIES}
      Threads::Threads
    )
  endif()
endif()
