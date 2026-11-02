# From Structural Micro-Deformation Detection to Protective Robot Retreat

ROS 1 research software for repeated LiDAR observation, structural micro-deformation estimation, persistent deformation mapping, and a robot retreat demonstration. The repository includes a Gazebo environment, the modified FAST-LIO frontend used by the detector, experiment runners, comparison methods, recorders, and offline evaluation/plotting source code.

The default ICRA pipeline is:

```text
Gazebo A2 + front/rear Mid360
              |
     synchronized rigid merge + front IMU
              |
           FAST-LIO
              |
     registered cloud + pose covariance
              |
   frozen-reference deformation detector
              |
     deformation evolution / retention
              |
       deformation cost overlay
              |
    live SLAM occupancy + Navfn / DWA
              |
       protective return to HOME
```

Gazebo object identities, object poses, bounding boxes, and robot ground truth are used for recording and offline evaluation. The online detector and mission use registered sensor data and FAST-LIO/navigation odometry; object-truth association is not an online detection or return-trigger input.

## Repository contents

This is a **catkin workspace**: build from the repository root, where `src/` resides.

| Path | Role |
|---|---|
| `src/FAST_LIO` | Modified FAST-LIO with registered clouds, odometry, and covariance output required by the detector. |
| `src/livox_ros_driver` | Livox ROS messages/driver and the bundled Livox SDK source. |
| `src/gazebo_test/Mid360_simulation_plugin/livox_laser_simulation` | Ray-based LiDAR simulation, synchronized sensor merge, planar robot motion, controlled target motion, worlds, scan patterns, meshes, and local scene models. |
| `src/gazebo_test/Mid360_simulation_plugin/a2_description` | A2 robot URDF and meshes. |
| `src/deform_monitor_v3` | Current V1/V1.1 detector, messages/services, RViz adapter, recorders, and retained legacy V3 code. The package name is historical; the main launch uses `deform_monitor_v1_node`. |
| `src/deform_evolution_map` | Observable-measurement confirmation, persistent source retention, and short-horizon deformation forecasts. |
| `src/deform_avoidance_map` | Current/predicted deformation cost overlays and map interfaces. |
| `src/deform_monitor_bringup` | Full-system launch files, stationary and mobile missions, occupancy-map integration, experiment recording, and local planning evaluation. |
| `src/compare_methods` | Separate catkin overlay containing M-detector, spatiotemporal normals, Dynablox/Voxblox, M3C2, common benchmark recording, and source dependencies. |
| `src/batch_script` | Repeated stationary and comparison-method speed sweeps. |
| `src/analyze_script` | Analysis of newly recorded stationary, comparison, real-data, and retreat runs. |
| `src/IEEE_ICRA2027`, `src/figure_tools` | Paper-specific statistics, audits, and plotting source. Historical input records and generated paper products are not bundled. |

The root `CATKIN_IGNORE` prevents a parent catkin workspace from discovering this nested checkout. It does not prevent `catkin_make` from building this repository's own `src/`. `src/compare_methods/CATKIN_IGNORE` keeps the comparison overlay separate. The optional upstream `minkindr_python` bindings are also excluded from the overlay build; the C++ libraries needed by Dynablox are included.

## Implemented functionality

- **Sensor simulation:** front Mid360 and optional rear Mid360, an IMU stream, fixed-extrinsic merging, robot odometry truth, controlled target motion, and scene verification. Formal dual-LiDAR runs use a 20 m ray endpoint and finite returns strictly within `0.5 < range < 20.0 m` per sensor. The default front/rear sampling strides are `1/2`.
- **Localization and mapping:** FAST-LIO registers the scans, exports pose uncertainty, and supplies the frame used by detection and navigation. OctoMap and the map proxy provide the navigation occupancy map.
- **Micro-deformation estimation:** stationary reference construction, surfel/GDU support derived from observed geometry, displacement/velocity estimation with uncertainty and observability information, and explicit monitoring phases.
- **Evolution and retention:** newly supported deformation can become a retained source; retained information can survive temporary lack of current motion/visibility under the configured policy. Available forecasts are published with their source snapshot identity.
- **Navigation integration:** retained sources contribute finite deformation costs to navigation maps. The mobile mission builds a map while approaching an observation point, stops to construct a reference, monitors deformation, and returns using Navfn and DWA after the configured persistent trigger.
- **Experiments and evaluation:** structured algorithm/sensor/truth records, checkpoint products, detection and runtime statistics, offline truth alignment, local SLAM-only versus deformation-aware path comparison, and baseline speed sweeps.
- **Real-data ingestion:** detection can consume a separately started FAST-LIO stream or bag playback, with manual reference initialization and optional structured recording.

## Environment and dependencies

The reference platform is **Ubuntu 20.04, ROS Noetic, Gazebo Classic 11, Python 3.8, and a C++14-capable compiler**. This is ROS 1 code; it is not a ROS 2 package. A graphical desktop with OpenGL is required for Gazebo GUI and RViz; the simulation and batch runners also support headless operation.

Install ROS Noetic using its distribution instructions, then install the project dependencies. On the reference Ubuntu/ROS platform:

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

The Livox SDK is built from the included source during the main build. A separate system-wide SDK installation or SDK download is not needed. The comparison overlay uses system Eigen, Glog, Gflags, and Protobuf together with the included catkin wrapper packages.

## Download and build

```bash
git clone https://github.com/zhongguanLiu/From-Structural-Micro-Deformation-Detection-to-Protective-Robot-Retreat.git micro_deformation_ws
cd micro_deformation_ws
source /opt/ros/noetic/setup.bash
# After rosdep has been initialized on this machine:
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release -j2
source devel/setup.bash
```

Use a fresh terminal and this checkout's `devel/setup.bash`; sourcing another workspace with the same package names can select the wrong launch files or binaries. Build outputs stay in `build/` and `devel/` and are excluded from Git.

The simulator sets `GAZEBO_MODEL_PATH` to its bundled `models/` directory. The checked-in scene assets and scan tables are required runtime inputs, not recorded experiment data. No separate `~/.gazebo/models` collection is needed for the included worlds.

### Optional comparison overlay

Build the main workspace first. Then, from the repository root:

```bash
source devel/setup.bash
cd src/compare_methods
catkin_make -DCMAKE_BUILD_TYPE=Release -j2 \
  -DUSE_SYSTEM_EIGEN=ON -DUSE_SYSTEM_GLOG=ON -DUSE_SYSTEM_PROTOBUF=ON
source devel/setup.bash
cd ../..
```

M-detector, spatiotemporal normals, and Dynablox produce dynamic-point outputs; they do not provide the structural displacement vector estimated by the main method. M3C2 estimates signed normal distances with its own significance and reference definitions. Their outputs and timing are recorded separately rather than treated as identical measurements.

## Stationary observation experiment

From the repository root:

```bash
DEFORM_MONITOR_ROS_MASTER_PORT=11321 \
  ./src/deform_monitor_bringup/scripts/run_icra2027_stationary_full_process.sh \
  gui:=true rviz:=true wait_for_operator:=true \
  rviz_config:="$PWD/src/deform_monitor_bringup/config/rviz/icra2027_video.rviz"
```

1. Gazebo, FAST-LIO, the detector, the recorders, and RViz start.
2. Once the initial odometry and map are available, Gazebo pauses and the launch terminal prints **`READY_TO_RECORD`**.
3. Arrange the windows and start your screen recorder. Return to the launch terminal and **press Enter**, or type **`1`** and press Enter.
4. The robot remains at the observation pose. After stabilization and successful reference construction, the controlled experiment runs through the following simulation-time sequence:

| Experiment time | Target motion |
|---|---|
| 0–10 s | Stationary baseline |
| 10–50 s | Translation at the configured speed; default 1 mm/s |
| 50–60 s | Hold the final pose |

The runner records the full observation interval, exports results, and shuts down after finalization. Do not interrupt while it is finalizing. Check the final manifest as well as the terminal completion message.

Initialization/reference construction precedes experiment time zero. Wall-clock duration can be substantially longer than 60 seconds because Gazebo may run below real time.

## Mobile approach and protective retreat

After the stationary experiment has fully exited:

```bash
DEFORM_MONITOR_ROS_MASTER_PORT=11322 \
  ./src/deform_monitor_bringup/scripts/run_icra2027_closed_loop.sh \
  gui:=true rviz:=true wait_for_operator:=true
```

This uses the navigation RViz profile by default. The same `READY_TO_RECORD` gate gives time to arrange windows before the mission begins.

The mission performs online mapping and outbound navigation, stops at the observation area, freezes a reference, and starts the controlled target excitation. The default mobile excitation is a 3 s baseline, 15 s of translation at 1 mm/s, then a 10 s hold. Return requires a data-derived retained source to satisfy the configured 10 s source-time persistence condition and be represented in the decision map. Actual runtime evidence determines whether the trigger is reached.

The return uses the operational Navfn/DWA stack on the fused navigation map, not an offline comparison path. Upon return to HOME and successful finalization, the terminal reports `RUN_COMPLETE`; the environment stays open until `Ctrl+C`.

**Mobile mission does not mean deformation estimation while walking.** The default mission deliberately stops for reference construction and deformation monitoring. The A2 carrier uses planar kinematics; the simulation is not a foot-contact/gait or structural-collapse physics model.

### Runtime controls

- Both wrappers choose a private ROS master unless `DEFORM_MONITOR_ROS_MASTER_PORT` is supplied. They print its URI.
- GUI runs in an interactive terminal wait by default. `wait_for_operator:=true` makes that behavior explicit; it needs a terminal. Noninteractive/headless runs default to automatic start.
- For unattended runs, pass `gui:=false rviz:=false wait_for_operator:=false`.
- Run experiments sequentially: the wrappers isolate ROS, but do not assign a separate Gazebo master port to every run.
- `DEFORM_MONITOR_OUTPUT_ROOT` changes the results root. The default is this workspace's `experiment_output/`.
- `record_experiment:=false` is for disposable diagnostics and produces no complete scientific run manifest.
- Neither wrapper starts a video recorder or `rosbag record`. The Enter/`1` gate releases the experiment; start your recording application yourself.
- Do not reset Gazebo world/time after freezing a reference. Stop and relaunch to record a fresh trial.

To inspect the mobile experiment from another terminal:

```bash
source devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11322
export ROS_HOSTNAME=127.0.0.1
rostopic echo /deform/mission/status
```

## RViz displays and ROS interfaces

Use **`camera_init`** as the RViz fixed frame. The navigation profile is `src/deform_monitor_bringup/config/rviz/icra2027_closed_loop.rviz`; the video profile emphasizes the scene and deformation estimates. The video profile's map/evolution layers are optional displays and can be enabled in **Panels → Displays**.

| Topic | Meaning / RViz display |
|---|---|
| `/livox/lidar`, `/livox/imu` | Merged LiDAR and the primary IMU input to FAST-LIO. |
| `/cloud_registered` | Registered scene point cloud (`PointCloud2`). |
| `/Odometry`, `/lio_odom_cov` | FAST-LIO pose and covariance interface. |
| `/navigation/odometry` | Navigation odometry with derived velocity. |
| `/path` | Recorded FAST-LIO trajectory (`Path`). |
| `/deform/v1/status` | Detector phase, reference state, and processing diagnostics. |
| `/deform/v1/detection_tiles` | Structured detector estimates/evidence; display filtering does not remove these records. |
| `/deform/v1/rviz/deformation_markers` | Reference support, probability coloring, and displacement vectors (`MarkerArray`). |
| `/deform/v1/rviz/status_markers` | Monitoring state and experiment clock (`MarkerArray`). |
| `/deform/evolution/frame`, `/deform/evolution/markers` | Evolution/retention data and visualization. |
| `/deform/avoidance/current_overlay` | Current deformation costs (`Map`). |
| `/deform/avoidance/forecast_0p5s`, `/deform/avoidance/forecast_1p0s`, `/deform/avoidance/forecast_2p0s` | Available forecast cost grids. |
| `/mission/navigation_map` | Occupancy and deformation costs used by navigation (`Map`). |
| `/move_base/NavfnROS/plan` | Operational global navigation plan (`Path`). |
| `/move_base/DWAPlannerROS/local_plan` | Operational local controller trajectory (`Path`). |
| `/deform/mission/status` | Mission transitions and finalization request/status information. |

Displacement arrows are amplified **50× for visualization**, with a 1 m display-length cap. Preserve this label in videos. Physical estimates and recorded ground truth are not amplified. An arrow is an estimated displacement, not by itself a confirmed hazard. Deformation layers can be empty during outbound navigation or before reference/confirmation; plans appear when navigation has an active goal.

## Additional launch modes

```bash
# Standalone simulator and sensor visualization:
roslaunch livox_laser_simulation icra2027_simulation.launch

# Generic complete simulation/detection/evolution/avoidance pipeline:
roslaunch deform_monitor_bringup simulation_full.launch

# Downstream evolution/avoidance components for an already running detector:
roslaunch deform_monitor_bringup analysis_only.launch

# Read-only local SLAM-only versus deformation-aware planning evaluation:
roslaunch deform_monitor_bringup planning_evaluation.launch
```

The two experiment wrappers above are the recommended entries for the stationary and mobile ICRA missions; the generic launch modes do not reproduce all of their orchestration. Standalone motion control is available through:

```bash
rosrun livox_laser_simulation icra2027_motion_cli.py status
```

The CLI also supports `run`, `start`, `stop`, `reset`, and `reload`. Do not reset/reload target geometry after the detector freezes its reference. The main mission wrappers manage motion automatically.

## Real sensor data and bag playback

No bags or real sensor records are included. Supply a compatible LiDAR/IMU stream using the topics/message types expected by this FAST-LIO configuration. The bundled SDK/driver is the legacy Livox driver; compatibility with a particular physical device, especially devices requiring another SDK, must be checked separately.

In separate terminals, source this workspace and start:

```bash
# Frontend, using your compatible LiDAR/IMU stream or bag:
roslaunch fast_lio mapping_mid360_real.launch rviz:=false

# Detector and detection visualization; simulation clock for bag replay:
roslaunch deform_monitor_v3 real.launch record:=true source_bag:=/absolute/path/to/your.bag

# Supply your bag; pause on a stationary segment to establish the reference:
rosbag play --clock --pause /absolute/path/to/your.bag

# After the required stationary input is available:
rosrun deform_monitor_v3 v1_monitoring_cli.py start
rosrun deform_monitor_v3 v1_monitoring_cli.py status
```

Keep the sensor/environment stationary during reference initialization and resume the physical motion or moving bag segment only after the CLI reports an active frozen reference. Inspect or override the CLI's `--roi-min` and `--roi-max` bounds for your scene. For a live stream, set `use_sim_time:=false` on `real.launch`. The real launch does not start a hardware driver, Gazebo, or navigation.

The real recorder stores structured detector products and metadata; it does not substitute for independent displacement ground truth. A display response alone is not a physical accuracy measurement.

## Comparison methods and repeated trials

Build and source the comparison overlay before using these commands. A single run takes a method, speed in mm/s, and a fresh output directory:

```bash
./src/compare_methods/run_single_comparison.sh \
  m_detector 1 "$PWD/experiment_output/comparison_example"
```

Supported method names: `m_detector`, `spatiotemporal_normals`, `dynablox`, `m3c2`. Supported benchmark translation speeds: `1`, `3`, `5`, `10` mm/s. The comparison benchmark uses the **front LiDAR only** and the stationary 0/10/50/60 s timeline. The main formal mission defaults to **dual LiDAR**; do not describe the methods as having identical sensor inputs.

```bash
# Preview the stationary experiment matrix before running it:
./src/batch_script/run_icra2027_stationary_speed_sweep.sh --print-matrix
./src/batch_script/run_icra2027_stationary_speed_sweep.sh

# Three dynamic-point baselines:
./src/batch_script/run_compare_methods_single_lidar_speed_sweep.sh --print-matrix
./src/batch_script/run_compare_methods_single_lidar_speed_sweep.sh

# M3C2 sweep:
./src/batch_script/run_m3c2_speed_sweep.sh
```

The default sweeps use ten repetitions per speed. Batch scripts validate recording completeness; their retry/cleanup behavior can remove incomplete trial directories. Use dedicated output roots and do not point them at unrelated records.

## Output layout and offline analysis

Main experiment runners create timestamped directories under:

```text
experiment_output/
  stationary_full_process/icra2027_<timestamp>/
  closed_loop/icra2027_<timestamp>/
```

Depending on the run and its completion state, these contain:

| Product | Content |
|---|---|
| `scene/`, configuration/provenance files | Loaded scene, motion settings, parameters, and source/version information. |
| `deformation/` | Algorithm displacement/evidence records. |
| `truth/` | Evaluation-only robot/object ground truth. |
| `sensors/`, `runtime/` | IMU, LiDAR merge/cadence diagnostics, Gazebo performance, processing time. |
| `navigation/` | Robot trajectory and navigation-map products. |
| `events/` | Mission events and stationary checkpoint timing. |
| `analysis/` | Detection, truth alignment, runtime, trajectory, map, and planning-derived products. |
| `mission_summary.json` | Mission outcome and state history. |
| `run_manifest.json` | Recorder completion, required artifact checks, and output-contract status. |
| `ros_log_scrub_report.json` | Post-shutdown log sanitization result. |

A created directory or an available video is not proof of a completed experiment. Check the manifest's finalization outcome and artifact errors. The wrappers allow recorder drain/finalization before stopping ROS, then sanitize ROS logs after logger shutdown.

For new records, use the general analysis entry points with explicit input paths:

```bash
python3 -B src/analyze_script/analyze_stationary_speed_sweep.py \
  --batch-dir /absolute/path/to/your/stationary_batch
python3 -B src/analyze_script/analyze_compare_methods.py --help
python3 -B src/analyze_script/analyze_m3c2.py --help
python3 -B src/analyze_script/analyze_v1_real.py --help
python3 -B src/analyze_script/analyze_evacuation.py --help
```

`src/IEEE_ICRA2027/` and `src/figure_tools/` retain the historical paper-analysis source and its named batch paths. Those programs require the corresponding records and intermediate statistics; they cannot regenerate published figures or paper metrics from this data-free checkout alone. Recorded data, aggregate CSV/JSON results, manuscripts/PDFs, proprietary fonts, screenshots, and videos are intentionally absent. The general analysis tools are the entry points for evaluating new runs.

## Checks

After building and sourcing the main workspace:

```bash
python3 -B -m unittest discover -s src/deform_monitor_bringup/test -p 'test_closed_loop_core.py'
python3 -B -m unittest discover -s src/deform_monitor_bringup/test -p 'test_stationary_monitor_cloud_adapter.py'
python3 -B src/deform_monitor_bringup/test/test_operator_start.py
python3 -B -m unittest discover -s src/deform_monitor_v3/test -p 'test_sim_experiment_recorder.py'
python3 -B -m unittest discover -s src/deform_monitor_v3/test -p 'test_launch_output_root.py'
roslaunch --nodes deform_monitor_bringup icra2027_stationary_full_process.launch wait_for_operator:=true
roslaunch --nodes deform_monitor_bringup icra2027_closed_loop.launch wait_for_operator:=true
```

For this source-release snapshot, both catkin workspaces compiled successfully, 168 targeted Python tests passed, and the two main mission launches plus all four comparison-method launches resolved. The source/asset check found no missing literal mesh/model/texture references. A full Gazebo experiment was not rerun as part of packaging.

`roslaunch --nodes` resolves the launch configuration; it does not execute or validate a full navigation mission. Additional C++ unit tests and ROS integration tests are included alongside their packages.

## Measurement and interpretation limits

- Calibration is explicitly marked unvalidated in the supplied research calibration file. Application-specific noise, extrinsics, task scales, and validity decisions require calibration for physical deployment.
- The sensor configuration requests a 10 Hz update schedule. Actual delivered measurements depend on Gazebo workload; the reference dual-ray setup recorded approximately 5 Hz. Use recorded timestamps and diagnostics rather than treating the configured rate as measured rate.
- Experiment manifests include strict sensing/recording contracts, including cadence and pairing criteria. A faster/slower host or changed sensor setup may need a separately justified protocol; a failed check should not be reported as a successful reference experiment.
- Finite deformation costs influence planning. They are not a structural failure model, a guaranteed safe region, or a validated collapse warning.
- Simulated truth supports offline evaluation. Real-data runs without independent truth do not establish absolute deformation accuracy or warning lead time.
- The mobile mission demonstrates approach, stationary sensing, and return. It does not establish continuous moving-platform deformation accuracy or physical quadruped gait performance.

## Repository size and upload

The source release includes its scene assets and LiDAR sampling tables. The unbuilt source tree is approximately 288 MiB; the largest individual files are approximately 33.25 MiB. Ordinary Git upload is sufficient; Git LFS is not required for these files.

Use `git push` for the repository rather than the browser upload form: some required files exceed the browser's 25 MiB limit. GitHub warns for files above 50 MiB and blocks ordinary Git files above 100 MiB. See [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The `.gitignore` excludes build trees, verification scratch files, recordings, experiment outputs, caches, and generated paper products. Do not force-add those outputs or commit the original workstation workspace as a whole. All nested third-party Git metadata has been removed; their source and applicable license files are included directly.

## Licenses and attribution

This is a mixed-license source tree, not a single blanket license grant. Preserve the notices in the individual components:

- FAST-LIO includes a [GPL v2 license file](src/FAST_LIO/LICENSE), despite its historical package metadata saying BSD; the bundled source notices must be reviewed when redistributing derivatives.
- M-detector includes its [GPL license](src/compare_methods/src/M-detector/LICENSE).
- The spatiotemporal-normal detector includes an [MIT license](src/compare_methods/src/dynamic_object_detection/LICENSE).
- Dynablox, Voxblox, A2 description, Livox SDK, and the supporting libraries retain their included license texts and copyright notices.
- The deformation packages declare MIT in their package metadata. Some simulator/model assets and dependency wrappers do not include a complete license statement. No new license is assigned to those assets by this packaging; their redistribution terms need to be established by the repository owner.

Third-party authorship is retained in the original source headers, package metadata, and license files. Please cite the underlying methods and this work when using the corresponding components in research.
