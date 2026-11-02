deform_monitor_v3/V1DetectionCommon common
uint8 phase
deform_monitor_v3/V1StableId128 reference_snapshot_id
bool reference_frozen
uint32 gdu_count
uint32 interface_count
string[] active_gates
uint64 deadline_failure_count
uint64 resource_failure_count
string reason_catalog_version
time last_attempt_stamp
uint64 last_attempt_frame_index
bool last_attempt_published
uint16[] last_attempt_reason_codes
# Wall-clock work of the most recent synchronized input attempt.  It includes
# BuildBundle + V1Pipeline::AcceptFrame and, on a published success, building
# the deformation/interface/tile/status products plus the synchronous roscpp
# handoff of deformation/interface/tile messages.  It excludes transport,
# subscriber work, pre-synchronization waiting and the final status handoff.
float64 last_attempt_processing_time_ms
uint64 last_attempt_input_point_count
