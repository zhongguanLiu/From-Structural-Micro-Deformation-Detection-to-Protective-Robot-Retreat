uint8 NEW_DEFORMATION_SOURCE=0
uint8 UPDATED_DEFORMATION_SOURCE=1
uint8 CLEARED_DEFORMATION_SOURCE=2

std_msgs/Header header
uint64 map_revision
uint8 event_type
string source_id
geometry_msgs/Point position
float64 displacement_rms_m
uint8 magnitude_trend
string policy
