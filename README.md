# From Structural Micro-Deformation Detection to Protective Robot Retreat

**Estimate small surface displacements from repeated LiDAR observations and use confirmed changes to inform robot retreat.**

This ROS 1 workspace connects structural monitoring with robot task execution. A robot builds a local reference from its own observations, estimates displacement relative to that reference, retains confirmed deformation evidence, and passes that evidence to a navigation system. It includes the estimator, Gazebo simulation, visualization, stationary monitoring and mobile mission runners, comparison methods, and evaluation tools.

## Motivation and purpose

Robots operating in confined spaces with damaged structures need information about how their surroundings are changing. A passage can remain geometrically traversable while a nearby surface has begun to move. Measuring that change can provide a basis for ending an observation task and choosing a protective return route before the passage becomes obstructed.

Small-displacement measurement is difficult because pose error, changing LiDAR samples, uncertain correspondences, and actual surface motion all produce differences between scans. A locally consistent SLAM map alone does not separate these effects. The problem is especially relevant in unknown environments, where a design model, surveyed benchmark, or predefined set of structural components may be unavailable.

The workspace addresses this problem by establishing an on-site reference and estimating displacement in the directions supported by the observed geometry. Its downstream mission demonstrates how persistent deformation evidence can affect both the decision to return and the cost map used to select the return route.

## Estimation method

The detector consumes registered LiDAR clouds, sensor poses, and pose uncertainty from the modified FAST-LIO frontend. Its main stages are:

1. **Establish a fixed reference.** Initial stable observations form local surface elements, or surfels. Connected, geometrically compatible surfels are grouped into geometric deformation units (GDUs). These units come from observed geometry and do not require object labels or component models.
2. **Estimate an observable displacement field.** A continuous field represents displacement over each unit, allowing samples taken at different positions and times to constrain a common state. Geometry determines the admissible directions: a locally planar surface primarily constrains normal displacement, while richer geometry can support additional directions.
3. **Account for shared measurement errors.** Reference bias, extrinsic uncertainty, and residual pose error can affect many observations together. The temporal estimator carries these effects alongside displacement, preserving their correlation instead of treating repeated observations as independent evidence. Pose-control and displacement-measurement samples have separate roles, and local pose refinement excludes the monitored unit's neighborhood.
4. **Retain correspondence uncertainty.** Several plausible matches are evaluated together with geometric and temporal hypotheses. This reduces dependence on a single nearest-neighbor choice when sampling or visibility changes.
5. **Distinguish motion from a retained offset.** Temporal hypotheses represent no identifiable change, ongoing displacement, and a persistent offset. A surface that moves and then stops can therefore retain evidence of its changed position.
6. **Confirm and retain supported changes.** The downstream evolution layer applies observation and persistence requirements, retaining supported sources for task use. Current measurement support and retained deformation information remain distinguishable in the outputs.

The resulting products include displacement estimates, uncertainty and observability information, monitoring state, and retained deformation sources. RViz displays these products alongside the registered scene, robot trajectory, and navigation maps.

## From deformation evidence to retreat

```text
LiDAR + IMU
    |
FAST-LIO: registered clouds, pose, and covariance
    |
Fixed reference + observable displacement estimation
    |
Temporal confirmation and retained deformation sources
    |
Ground-plane deformation costs + SLAM occupancy map
    |
Mission trigger + Navfn planning / DWA control
    |
Protective return to the starting location
```

The mobile mission first maps the environment while approaching an observation position. It stops to establish a reference and monitor the scene. When retained deformation satisfies the configured persistence condition and is represented in the decision map, the mission requests a return. The existing ROS navigation stack plans and executes that return using occupancy and finite deformation costs.

This connection lets measured structural change influence robot behavior while free space remains available. The deformation layer can make a route through an affected region more expensive, allowing the planner to trade travel distance against exposure to retained deformation evidence. Finite costs do not guarantee complete avoidance.

Gazebo object identities and ground-truth poses support recording and offline evaluation. They are not used as target labels for online displacement estimation or as the source of the retreat trigger.

## Included functionality

| Component | Role |
|---|---|
| [`src/deform_monitor_v3`](src/deform_monitor_v3) | Fixed-reference deformation estimation, uncertainty-aware temporal inference, ROS messages/services, visualization, and recording. The package name is historical; the main pipeline runs `deform_monitor_v1_node`. |
| [`src/deform_evolution_map`](src/deform_evolution_map) | Observation-supported confirmation, retained deformation sources, and short-horizon forecasts. |
| [`src/deform_avoidance_map`](src/deform_avoidance_map) | Current and predicted deformation cost overlays for navigation. |
| [`src/deform_monitor_bringup`](src/deform_monitor_bringup) | Integrated launch files, stationary and mobile workflows, mission state management, map integration, and experiment recording. |
| [`src/FAST_LIO`](src/FAST_LIO) | LiDAR-inertial registration, odometry, and pose-covariance output. |
| [`src/livox_ros_driver`](src/livox_ros_driver) | Livox ROS interfaces and bundled Livox SDK source. |
| [`src/gazebo_test/Mid360_simulation_plugin`](src/gazebo_test/Mid360_simulation_plugin) | A2 robot model, Mid360 simulation, dual-sensor merging, scene assets, scan patterns, and controlled target motion. |
| [`src/compare_methods`](src/compare_methods) | M-detector, spatiotemporal normals, Dynablox/Voxblox, and M3C2 with benchmark runners and supporting source dependencies. |
| [`src/batch_script`](src/batch_script) | Repeated trials and deformation-speed sweeps. |
| [`src/analyze_script`](src/analyze_script), [`src/figure_tools`](src/figure_tools) | Detection, displacement, runtime, trajectory, and map analysis; plotting tools. Additional experiment-specific analysis source is retained under `src/`. |

The main operating modes are stationary observation, mobile approach followed by stationary monitoring and retreat, and processing of compatible real LiDAR-inertial streams or recordings. The mobile workflow performs deformation monitoring while stopped.

## Environment and build

The reference environment is **Ubuntu 20.04, ROS Noetic, Gazebo Classic 11, Python 3.8, and a C++14-capable compiler**. Gazebo GUI and RViz require a graphical desktop with OpenGL. Headless execution is also supported.

After installing ROS Noetic, install the dependencies:

```bash
sudo apt-get update
sudo apt-get install \
  build-essential cmake python3-rosdep python3-catkin-pkg python3-rospkg \
  python3-numpy python3-scipy python3-pandas python3-matplotlib python3-yaml \
  python3-nose python3-pytest \
  libeigen3-dev libpcl-dev libboost-all-dev libyaml-cpp-dev libapr1-dev \
  libgoogle-glog-dev libgflags-dev libprotobuf-dev protobuf-compiler zlib1g-dev \
  ros-noetic-gazebo-ros-pkgs ros-noetic-rviz ros-noetic-xacro \
  ros-noetic-robot-state-publisher ros-noetic-joint-state-publisher \
  ros-noetic-navigation ros-noetic-octomap-server ros-noetic-pcl-ros \
  ros-noetic-cv-bridge ros-noetic-eigen-conversions ros-noetic-interactive-markers
```

This repository is a catkin workspace. Build from its root:

```bash
git clone https://github.com/zhongguanLiu/From-Structural-Micro-Deformation-Detection-to-Protective-Robot-Retreat.git micro_deformation_ws
cd micro_deformation_ws
source /opt/ros/noetic/setup.bash
# Requires rosdep initialization on this machine.
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release -j2
source devel/setup.bash
```

The Livox SDK is built from bundled source. Included Gazebo worlds resolve their scene models through the bundled model directory. Required meshes, textures, and LiDAR scan tables are included; recorded experiment data are not.

Use this checkout's `devel/setup.bash` in a fresh terminal to avoid selecting packages from another workspace. The root `CATKIN_IGNORE` prevents a parent workspace from discovering a nested checkout; it does not prevent building this repository's own `src/`. The comparison workspace has a separate `CATKIN_IGNORE` and is built as an overlay.

## Run the simulation workflows

Run the following commands from the repository root in Bash. Each filename pattern below selects one bundled runner.

For stationary observation:

```bash
DEFORM_MONITOR_ROS_MASTER_PORT=11321 \
  ./src/deform_monitor_bringup/scripts/run_*_stationary_full_process.sh \
  gui:=true rviz:=true wait_for_operator:=true
```

The robot stays at the monitoring pose. After stabilization and reference construction, the default experiment contains a 10 s stationary baseline, 40 s of translation at 1 mm/s, and a 10 s hold. These intervals use simulation time; initialization happens before experiment time zero.

For approach, monitoring, and protective retreat, after the stationary run has exited:

```bash
DEFORM_MONITOR_ROS_MASTER_PORT=11322 \
  ./src/deform_monitor_bringup/scripts/run_*_closed_loop.sh \
  gui:=true rviz:=true wait_for_operator:=true
```

The mobile runner starts the navigation RViz profile. Its default target-motion sequence is a 3 s baseline, 15 s of translation at 1 mm/s, and a 10 s hold. A retained source must satisfy the configured 10 s source-time persistence condition and appear in the decision map before it can trigger return.

Both interactive runners pause after initialization and print **`READY_TO_RECORD`**. Arrange the Gazebo and RViz windows, start an external video recorder if desired, then return to the launch terminal and press **Enter**, or type **`1`** and press Enter. This releases the workflow; it does not record the screen or start `rosbag record`.

Wait for **`RUN_COMPLETE`** before interrupting finalization. The stationary workflow exits after recording; the mobile workflow keeps the environment open until `Ctrl+C`. Run the workflows sequentially, since their separate ROS masters do not also isolate Gazebo's master port. For unattended execution, use `gui:=false rviz:=false wait_for_operator:=false`.

For component-level integration, the following launches are also available:

```bash
# Simulator, frontend, detector, evolution, avoidance, and visualization:
roslaunch deform_monitor_bringup simulation_full.launch

# Evolution and avoidance with an already running detector:
roslaunch deform_monitor_bringup analysis_only.launch

# Local comparison of occupancy-only and deformation-aware plans:
roslaunch deform_monitor_bringup planning_evaluation.launch
```

The component launches do not perform the complete mission orchestration of the two runners.

## Visualization and ROS outputs

Use **`camera_init`** as the RViz fixed frame. The bundled mission profile displays registered clouds, deformation evidence, navigation maps, and paths.

| Topic | Content |
|---|---|
| `/cloud_registered` | Registered LiDAR scene. |
| `/Odometry`, `/lio_odom_cov` | Frontend pose and covariance. |
| `/deform/v1/status` | Detector phase, reference state, and diagnostics. |
| `/deform/v1/detection_tiles` | Structured estimates and deformation evidence. |
| `/deform/v1/rviz/deformation_markers` | Reference support, probability coloring, and displacement vectors. |
| `/deform/evolution/frame`, `/deform/evolution/markers` | Current and retained deformation information. |
| `/deform/avoidance/current_overlay` | Current deformation cost grid. |
| `/mission/navigation_map` | Occupancy and deformation costs used by navigation. |
| `/move_base/NavfnROS/plan` | Global navigation plan. |
| `/move_base/DWAPlannerROS/local_plan` | Local controller trajectory. |
| `/deform/mission/status` | Mission state and completion information. |

Displacement arrows are enlarged **50 times for display**, with a 1 m display-length cap. Estimates and recorded measurements retain their physical units. Deformation layers appear when supporting evidence is available; navigation plans appear when a goal is active.

## Real sensor data

Supply a compatible LiDAR/IMU stream or bag using the topics and message types expected by the frontend. In separate terminals, source the workspace and start the required components:

```bash
roslaunch fast_lio mapping_mid360_real.launch rviz:=false
roslaunch deform_monitor_v3 real.launch record:=true source_bag:=/absolute/path/to/your.bag
rosbag play --clock --pause /absolute/path/to/your.bag
```

Advance the recording through a stationary segment and initialize monitoring with:

```bash
rosrun deform_monitor_v3 v1_monitoring_cli.py start
rosrun deform_monitor_v3 v1_monitoring_cli.py status
```

Keep the sensor and scene stationary during reference construction. Proceed to the moving segment only after the reference is active, and configure the CLI's region-of-interest bounds for the scene. For a live stream, use `use_sim_time:=false` on `real.launch`. This launch does not start a hardware driver or navigation; device compatibility and calibration must be established separately.

## Comparison and evaluation

Build the optional comparison overlay after the main workspace:

```bash
source devel/setup.bash
cd src/compare_methods
catkin_make -DCMAKE_BUILD_TYPE=Release -j2 \
  -DUSE_SYSTEM_EIGEN=ON -DUSE_SYSTEM_GLOG=ON -DUSE_SYSTEM_PROTOBUF=ON
source devel/setup.bash
cd ../..

./src/compare_methods/run_single_comparison.sh \
  m_detector 1 "$PWD/experiment_output/comparison_example"
```

Supported method names are `m_detector`, `spatiotemporal_normals`, `dynablox`, and `m3c2`; benchmark speeds are 1, 3, 5, and 10 mm/s. The first three methods output dynamic-point indications. M3C2 measures signed normal distances. Their outputs should be evaluated according to those meanings rather than treated as interchangeable displacement estimates.

The comparison benchmark uses the front LiDAR; the main mission defaults to dual LiDAR. Both follow controlled target-motion schedules, but their sensor inputs differ. Batch scripts support repeated trials and validate recording completeness. Use a dedicated output directory because incomplete trial directories can be removed during retries.

The evaluation tools support detection and confirmation timing, displacement error against simulated truth, estimate availability, processing time, and analysis of robot trajectories and navigation costs. Real-data processing provides estimates under real sensing conditions; absolute accuracy requires independent displacement ground truth.

## Recorded outputs and source checks

The default results root is `experiment_output/`, with timestamped runs under `stationary_full_process/` and `closed_loop/`. Set `DEFORM_MONITOR_OUTPUT_ROOT` to use another location.

Recorders export detector products, sensor/runtime diagnostics, evaluation truth, mission events, trajectories, map products, and analysis results as applicable. Inspect `run_manifest.json` for finalization and required-artifact checks, and `mission_summary.json` for the mission outcome. The recorders finish writing before the runners shut down ROS.

The repository contains source and runtime assets. Raw recordings, aggregate experiment results, generated figures, and videos are excluded. Historical analysis scripts may require named datasets and intermediate results supplied separately; the general tools in `src/analyze_script` accept new runs.

Both catkin workspaces compiled successfully during source packaging, and 168 targeted Python tests passed. Main mission launches and all four comparison-method launches resolved. Example checks after building and sourcing the main workspace are:

```bash
python3 -B -m unittest discover -s src/deform_monitor_bringup/test -p 'test_closed_loop_core.py'
python3 -B -m unittest discover -s src/deform_monitor_bringup/test -p 'test_stationary_monitor_cloud_adapter.py'
python3 -B src/deform_monitor_bringup/test/test_operator_start.py
python3 -B -m unittest discover -s src/deform_monitor_v3/test -p 'test_sim_experiment_recorder.py'
python3 -B -m unittest discover -s src/deform_monitor_v3/test -p 'test_launch_output_root.py'
```

These checks cover software integration and selected logic; a full Gazebo mission was not rerun during packaging.

## Scope and practical limits

The estimator targets small displacements relative to a fixed local reference. Measurement availability depends on visibility, geometric observability, correspondence support, and the configured reference neighborhood. Large changes can move a surface beyond that support.

The supplied research calibration is marked unvalidated. Physical deployment requires sensor/extrinsic calibration and application-specific assessment of noise, task scales, and confirmation criteria. Actual processing and sensor delivery rates depend on hardware and simulation load; use recorded timestamps and diagnostics when evaluating a run.

The mobile demonstration uses planar robot kinematics and stationary deformation monitoring. It does not validate quadruped gait, continuous deformation measurement while walking, or structural-collapse prediction. Deformation costs provide task-relevant evidence; they are not a structural failure model or a guarantee of safe passage.

## Licenses

This is a mixed-license source tree. Preserve each component's license and attribution notices.

- FAST-LIO includes a [GPL v2 license](src/FAST_LIO/LICENSE), although its historical package metadata says BSD.
- M-detector includes its [GPL license](src/compare_methods/src/M-detector/LICENSE).
- The spatiotemporal-normal detector includes an [MIT license](src/compare_methods/src/dynamic_object_detection/LICENSE).
- Dynablox, Voxblox, A2 description, Livox SDK, and supporting libraries retain their supplied license texts and copyright notices.
- The deformation packages declare MIT in their package metadata. Some simulator/model assets and dependency wrappers lack a complete license statement; this repository does not assign a new license to those assets.
