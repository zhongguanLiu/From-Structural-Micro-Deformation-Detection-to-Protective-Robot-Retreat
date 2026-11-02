#include "deform_monitor_v3/core/scalar_measurement_builder.hpp"

#include "deform_monitor_v3/core/observable_subspace.hpp"

#include <cmath>

namespace deform_monitor_v3 {

namespace {

bool IsFiniteScalar(double v) {
  return std::isfinite(v);
}

}  // namespace

void ScalarMeasurementBuilder::SetParams(const ObservationParams& observation_params,
                                         const NoiseParams& noise_params) {
  observation_params_ = observation_params;
  noise_params_ = noise_params;
}

AlignedVector<ScalarMeasurement> ScalarMeasurementBuilder::BuildMeasurements(
    const AnchorReference& anchor,
    const LocalSupportData& support,
    const PoseCov6D& /*pose_cov*/,
    const Eigen::Vector3d& /*lidar_origin_R*/,
    bool apply_innovation_gate) const {
  AlignedVector<ScalarMeasurement> scalars;
  if (!support.valid || support.support_count < observation_params_.min_support_scalar) {
    return scalars;
  }

  auto maybe_add = [&](const Eigen::Vector3d& h,
                       double z,
                       double independent_r,
                       const Eigen::Matrix3d& reference_covariance_R,
                       uint8_t reference_covariance_group,
                       uint8_t type) -> bool {
    const double h_norm = h.norm();
    if(h_norm<0.9||h_norm>1.1||!reference_covariance_R.allFinite()||
       !IsFiniteScalar(independent_r)||independent_r<=0.0)return false;
    const Eigen::Vector3d unit_h=h/h_norm;
    const double reference_r=unit_h.dot(reference_covariance_R*unit_h);
    const double r=independent_r+reference_r;
    if (!IsFiniteScalar(z) || !IsFiniteScalar(r) || r <= 0.0) {
      return false;
    }
    if (support.support_count < observation_params_.min_support_scalar) {
      return false;
    }
    const double nis_like = std::abs(z) / std::sqrt(r);
    if (apply_innovation_gate && nis_like > observation_params_.tau_nis_scalar) {
      return false;
    }
    ScalarMeasurement m;
    m.h_R = unit_h;
    m.z = z;
    m.r = r;
    m.independent_r = independent_r;
    m.reference_covariance_R = reference_covariance_R;
    m.reference_covariance_group = reference_covariance_group;
    m.type = type;
    scalars.push_back(m);
    return true;
  };

  const ObservableSubspace subspace = BuildObservableSubspace(anchor);
  const Eigen::Vector3d n = subspace.basis_R.col(0);

  // View-dependent noise inflation: applies to all anchor types when the
  // sensor has moved, increasing measurement variance proportionally to
  // range change and view angle change.
  const double view_angle_deg = AngleBetweenDeg(support.view_dir_R, anchor.mean_view_dir_R);
  const double range_delta = std::max(0.0, support.range - anchor.mean_range);
  const double view_noise_scale =
      1.0 + noise_params_.kappa_r * range_delta + noise_params_.kappa_v * view_angle_deg;

  const double sigma_plane = noise_params_.sigma_pi0 * view_noise_scale;
  const double z_plane = n.dot(support.centroid_R - anchor.center_R);
  const double independent_r_plane=
      n.dot(support.centroid_cov*n)+sigma_plane*sigma_plane;
  // The normal measurement is the common observable dimension for all anchor
  // types. If it is invalid, a secondary scalar is not allowed to update the
  // state on its own.
  if (!maybe_add(n,z_plane,independent_r_plane,anchor.Sigma_ref_geom,0,0)) {
    return scalars;
  }

  if (anchor.type == AnchorType::EDGE) {
    const Eigen::Vector3d e1 = subspace.basis_R.col(1);
    const int min_edge_support =
        std::max(2, (observation_params_.min_support_scalar + 1) / 2);
    if (support.edge_geometry_valid &&
        support.edge_support_count >= min_edge_support) {
      const double sigma_edge = noise_params_.sigma_edge0 * view_noise_scale;
      const double z_edge = e1.dot(support.edge_centroid_R - anchor.edge_center_R);
      const double independent_r_edge=
          e1.dot(support.edge_centroid_cov*e1)+sigma_edge*sigma_edge;
      maybe_add(e1,z_edge,independent_r_edge,anchor.Sigma_ref_edge,1,1);
    }
  } else if (anchor.type == AnchorType::BAND) {
    const Eigen::Vector3d e2 = subspace.basis_R.col(1);
    const Eigen::Vector3d delta = support.band_centroid_R - anchor.band_center_R;
    const double z_band = e2.dot(delta);
    const double independent_r_band=e2.dot(support.band_centroid_cov*e2)+
        noise_params_.sigma_bc0*noise_params_.sigma_bc0;
    maybe_add(e2,z_band,independent_r_band,anchor.Sigma_ref_geom,0,3);
  }

  return scalars;
}

}  // namespace deform_monitor_v3
