#include "deform_monitor_v3/v1/math.hpp"

#include "deform_monitor_v3/v1/deterministic.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace deform_monitor_v3 {
namespace v1 {

double ClampProbability(double probability, double floor) {
  const double bounded_floor = std::max(1.0e-15, std::min(0.1, floor));
  return std::max(bounded_floor, std::min(1.0 - bounded_floor, probability));
}

double LogSumExp(const std::vector<double>& terms) {
  if (terms.empty()) return -std::numeric_limits<double>::infinity();
  const double maximum = *std::max_element(terms.begin(), terms.end());
  if (!std::isfinite(maximum)) return maximum;
  std::vector<double> scaled;
  scaled.reserve(terms.size());
  for (double term : terms) scaled.push_back(std::exp(term - maximum));
  return maximum + std::log(PairwiseSum(scaled));
}

std::vector<double> Softmax(const std::vector<double>& log_scores) {
  std::vector<double> probabilities(log_scores.size(), 0.0);
  const double normalizer = LogSumExp(log_scores);
  if (!std::isfinite(normalizer)) return probabilities;
  for (size_t i = 0; i < log_scores.size(); ++i) {
    probabilities[i] = std::isfinite(log_scores[i])
                           ? std::exp(log_scores[i] - normalizer) : 0.0;
  }
  const double total = PairwiseSum(probabilities);
  if (total > 0.0) for (double& probability : probabilities) probability /= total;
  return probabilities;
}

double GaussianLogDensity(double residual, double variance) {
  if (!(variance > 0.0) || !std::isfinite(variance) || !std::isfinite(residual))
    return -std::numeric_limits<double>::infinity();
  static const double kLogTwoPi = std::log(2.0 * M_PI);
  return -0.5 * (kLogTwoPi + std::log(variance) + residual * residual / variance);
}

double LogNegativeBinomialPmf(int count, double mean, double dispersion) {
  if (count < 0 || !(mean >= 0.0) || !(dispersion > 0.0) ||
      !std::isfinite(mean) || !std::isfinite(dispersion))
    return -std::numeric_limits<double>::infinity();
  if (mean == 0.0) return count == 0 ? 0.0 : -std::numeric_limits<double>::infinity();
  const double k = static_cast<double>(count);
  const double r = dispersion;
  return std::lgamma(k + r) - std::lgamma(r) - std::lgamma(k + 1.0) +
         r * (std::log(r) - std::log(r + mean)) +
         k * (std::log(mean) - std::log(r + mean));
}

double LogVisibilityMixedNegativeBinomial(int count, double mean, double dispersion,
                                          double visibility, double probability_floor) {
  if (count < 0 || !std::isfinite(visibility) || visibility < 0.0 ||
      visibility > 1.0)
    return -std::numeric_limits<double>::infinity();
  if (visibility == 0.0)
    return count == 0 ? 0.0 : -std::numeric_limits<double>::infinity();
  if (visibility == 1.0)
    return LogNegativeBinomialPmf(count, mean, dispersion);
  const double pi = ClampProbability(visibility, probability_floor);
  if (count > 0) {
    return std::log(pi) + LogNegativeBinomialPmf(count, mean, dispersion);
  }
  return LogSumExp({std::log(pi) + LogNegativeBinomialPmf(0, mean, dispersion),
                    std::log1p(-pi)});
}

double WendlandC2(double normalized_distance) {
  if (!std::isfinite(normalized_distance) || normalized_distance >= 1.0) return 0.0;
  const double r = std::max(0.0, normalized_distance);
  const double one_minus = 1.0 - r;
  return one_minus * one_minus * one_minus * one_minus * (4.0 * r + 1.0);
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(),
            vector.z(), 0.0, -vector.x(),
            -vector.y(), vector.x(), 0.0;
  return matrix;
}

bool ValidateSymmetricPsd(const Eigen::MatrixXd& matrix, double tolerance,
                          double* minimum_eigenvalue) {
  if (matrix.rows() == 0 || matrix.rows() != matrix.cols() || !matrix.allFinite()) return false;
  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  if ((matrix - matrix.transpose()).cwiseAbs().maxCoeff() > tolerance) return false;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) return false;
  const double minimum = solver.eigenvalues().minCoeff();
  if (minimum_eigenvalue) *minimum_eigenvalue = minimum;
  return minimum >= -tolerance;
}

bool StableCholesky(const Eigen::MatrixXd& matrix, const NumericalConfig& numerical,
                    Eigen::MatrixXd* lower, double* jitter_used, int* attempts) {
  if (!lower || matrix.rows() == 0 || matrix.rows() != matrix.cols() || !matrix.allFinite())
    return false;
  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  double jitter = 0.0;
  for (int attempt = 0; attempt <= numerical.cholesky_jitter_attempts; ++attempt) {
    Eigen::MatrixXd candidate = symmetric;
    if (jitter > 0.0) candidate.diagonal().array() += jitter;
    Eigen::LLT<Eigen::MatrixXd> factor(candidate);
    if (factor.info() == Eigen::Success) {
      const Eigen::MatrixXd L = factor.matrixL();
      const double denominator = std::max(1.0, candidate.norm());
      const double residual = (L * L.transpose() - candidate).norm() / denominator;
      if (std::isfinite(residual) && residual <= numerical.solve_residual_tolerance) {
        *lower = L;
        if (jitter_used) *jitter_used = jitter;
        if (attempts) *attempts = attempt + 1;
        return true;
      }
    }
    jitter = attempt == 0 ? numerical.cholesky_jitter_initial
                          : jitter * numerical.cholesky_jitter_multiplier;
  }
  if (jitter_used) *jitter_used = jitter;
  if (attempts) *attempts = numerical.cholesky_jitter_attempts + 1;
  return false;
}

bool DeterministicPsdSquareRoot(const Eigen::MatrixXd& matrix,
                                const NumericalConfig& numerical,
                                Eigen::MatrixXd* lower) {
  if (!lower || matrix.rows() == 0 || matrix.rows() != matrix.cols() ||
      !matrix.allFinite() ||
      !ValidateSymmetricPsd(matrix,
                            numerical.psd_eigenvalue_tolerance)) {
    return false;
  }
  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  const int dimension = symmetric.rows();
  Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(dimension, dimension);
  const double scale = std::max(1.0, symmetric.cwiseAbs().maxCoeff());
  const double absolute_tolerance =
      numerical.psd_eigenvalue_tolerance * scale;
  for (int column = 0; column < dimension; ++column) {
    double diagonal = symmetric(column, column);
    for (int previous = 0; previous < column; ++previous)
      diagonal -= factor(column, previous) * factor(column, previous);
    if (diagonal < -absolute_tolerance || !std::isfinite(diagonal))
      return false;
    // For an exact PSD matrix, a zero pivot implies that every remaining
    // residual in the same column is zero.  Clamp only the admitted negative
    // round-off; a small positive pivot remains a real covariance direction.
    if (diagonal < 0.0) diagonal = 0.0;
    factor(column, column) = std::sqrt(diagonal);
    for (int row = column + 1; row < dimension; ++row) {
      double residual = symmetric(row, column);
      for (int previous = 0; previous < column; ++previous)
        residual -= factor(row, previous) * factor(column, previous);
      if (!std::isfinite(residual)) return false;
      if (factor(column, column) > 0.0) {
        factor(row, column) = residual / factor(column, column);
      } else if (std::abs(residual) > absolute_tolerance) {
        return false;
      }
    }
  }
  const double denominator = std::max(1.0, symmetric.norm());
  const double relative_residual =
      (factor * factor.transpose() - symmetric).norm() / denominator;
  if (!std::isfinite(relative_residual) ||
      relative_residual > numerical.solve_residual_tolerance) {
    return false;
  }
  *lower = factor;
  return true;
}

bool DeterministicPsdInformationFactor(
    const Eigen::MatrixXd& matrix, const NumericalConfig& numerical,
    Eigen::MatrixXd* information_factor) {
  if (!information_factor) return false;
  Eigen::MatrixXd lower;
  if (!DeterministicPsdSquareRoot(matrix, numerical, &lower)) return false;
  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  const double scale = std::max(1.0, symmetric.cwiseAbs().maxCoeff());
  const double root_threshold = std::sqrt(
      numerical.psd_eigenvalue_tolerance * scale);
  std::vector<int> active_columns;
  for (int column = 0; column < lower.cols(); ++column)
    if (lower(column, column) > root_threshold)
      active_columns.push_back(column);
  if (active_columns.empty()) {
    if (symmetric.norm() > numerical.solve_residual_tolerance *
                               std::max(1.0, symmetric.norm()))
      return false;
    information_factor->resize(0, matrix.cols());
    return true;
  }
  Eigen::MatrixXd carrier(lower.rows(),
                          static_cast<int>(active_columns.size()));
  for (size_t index = 0; index < active_columns.size(); ++index)
    carrier.col(static_cast<int>(index)) =
        lower.col(active_columns[index]);
  const Eigen::MatrixXd retained = carrier * carrier.transpose();
  const double covariance_denominator = std::max(1.0, symmetric.norm());
  if ((retained - symmetric).norm() / covariance_denominator >
      numerical.solve_residual_tolerance)
    return false;

  // A normal-equations inverse of carrier^T carrier squares the condition
  // number and rejects valid, strongly anisotropic posteriors.  A fixed-order
  // (unpivoted) Householder QR gives carrier = Q R and the exact rectangular
  // factor carrier^+ = R^{-1} Q^T without choosing an eigenbasis.
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(carrier);
  const int rank = carrier.cols();
  Eigen::MatrixXd upper = qr.matrixQR()
                              .topLeftCorner(rank, rank)
                              .template triangularView<Eigen::Upper>();
  Eigen::MatrixXd thin_q = qr.householderQ() *
      Eigen::MatrixXd::Identity(carrier.rows(), rank);
  if (!upper.allFinite() || !thin_q.allFinite()) return false;
  const double carrier_scale = std::max(1.0, carrier.norm());
  const double diagonal_threshold =
      numerical.psd_eigenvalue_tolerance * carrier_scale;
  for (int diagonal = 0; diagonal < rank; ++diagonal) {
    if (std::abs(upper(diagonal, diagonal)) <= diagonal_threshold)
      return false;
    // Fix the otherwise arbitrary Householder sign so the published factor is
    // canonical for a fixed coordinate ordering.
    if (upper(diagonal, diagonal) < 0.0) {
      upper.row(diagonal) *= -1.0;
      thin_q.col(diagonal) *= -1.0;
    }
  }
  const Eigen::MatrixXd inverse_upper =
      upper.triangularView<Eigen::Upper>().solve(
          Eigen::MatrixXd::Identity(rank, rank));
  const Eigen::MatrixXd candidate = inverse_upper * thin_q.transpose();
  if (!candidate.allFinite()) return false;

  const Eigen::MatrixXd left_identity = candidate * carrier;
  const Eigen::MatrixXd range_projector = carrier * candidate;
  if ((carrier - thin_q * upper).norm() / carrier_scale >
          numerical.solve_residual_tolerance ||
      (left_identity - Eigen::MatrixXd::Identity(rank, rank)).norm() /
              std::max(1.0, std::sqrt(static_cast<double>(rank))) >
          numerical.solve_residual_tolerance ||
      (range_projector - range_projector.transpose()).norm() /
              std::max(1.0, range_projector.norm()) >
          numerical.solve_residual_tolerance) {
    return false;
  }
  *information_factor = candidate;
  return true;
}

bool DeterministicPsdPseudoInverse(const Eigen::MatrixXd& matrix,
                                   const NumericalConfig& numerical,
                                   Eigen::MatrixXd* pseudoinverse) {
  if (!pseudoinverse) return false;
  Eigen::MatrixXd factor;
  if (!DeterministicPsdInformationFactor(matrix, numerical, &factor))
    return false;
  Eigen::MatrixXd candidate = factor.transpose() * factor;
  candidate = 0.5 * (candidate + candidate.transpose());
  if (!candidate.allFinite()) return false;
  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  const double covariance_denominator = std::max(1.0, symmetric.norm());
  const double inverse_denominator = std::max(1.0, candidate.norm());
  if ((symmetric * candidate * symmetric - symmetric).norm() /
          covariance_denominator > numerical.solve_residual_tolerance ||
      (candidate * symmetric * candidate - candidate).norm() /
          inverse_denominator > numerical.solve_residual_tolerance ||
      (symmetric * candidate - candidate * symmetric).norm() /
          std::max(1.0, (symmetric * candidate).norm()) >
          numerical.solve_residual_tolerance)
    return false;
  *pseudoinverse = candidate;
  return true;
}

bool MinimumRetainedInformationRatio(
    const Eigen::MatrixXd& known_information,
    const Eigen::MatrixXd& marginalized_information,
    double relative_rank_tolerance, double* ratio, int* effective_rank) {
  if (!ratio || known_information.rows() == 0 ||
      known_information.rows() != known_information.cols() ||
      marginalized_information.rows() != known_information.rows() ||
      marginalized_information.cols() != known_information.cols() ||
      !known_information.allFinite() || !marginalized_information.allFinite() ||
      !(relative_rank_tolerance >= 0.0) ||
      !std::isfinite(relative_rank_tolerance)) return false;
  const Eigen::MatrixXd known =
      0.5 * (known_information + known_information.transpose());
  const Eigen::MatrixXd marginalized =
      0.5 * (marginalized_information + marginalized_information.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> known_eigen(known);
  if (known_eigen.info() != Eigen::Success) return false;
  const double maximum = known_eigen.eigenvalues().maxCoeff();
  if (!(maximum > 0.0)) return false;
  const double threshold = relative_rank_tolerance * std::max(1.0, maximum);
  std::vector<int> positive_modes;
  for (int mode = 0; mode < known.rows(); ++mode) {
    if (known_eigen.eigenvalues()[mode] > threshold)
      positive_modes.push_back(mode);
  }
  if (positive_modes.empty()) return false;
  Eigen::MatrixXd inverse_sqrt(
      known.rows(), static_cast<int>(positive_modes.size()));
  for (size_t column = 0; column < positive_modes.size(); ++column) {
    const int mode = positive_modes[column];
    inverse_sqrt.col(static_cast<int>(column)) =
        known_eigen.eigenvectors().col(mode) /
        std::sqrt(known_eigen.eigenvalues()[mode]);
  }
  const Eigen::MatrixXd generalized =
      inverse_sqrt.transpose() * marginalized * inverse_sqrt;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> generalized_eigen(
      0.5 * (generalized + generalized.transpose()), Eigen::EigenvaluesOnly);
  if (generalized_eigen.info() != Eigen::Success ||
      !generalized_eigen.eigenvalues().allFinite()) return false;
  *ratio = std::max(0.0, std::min(
      1.0, generalized_eigen.eigenvalues().minCoeff()));
  if (effective_rank)
    *effective_rank = static_cast<int>(positive_modes.size());
  return true;
}

bool MinimumTaskNormalizedInformation(
    const Eigen::MatrixXd& information,
    const Eigen::MatrixXd& task_metric,
    double relative_rank_tolerance, double* minimum_information,
    int* effective_rank, int* task_rank, double* maximum_information) {
  if (!minimum_information || information.rows() == 0 ||
      information.rows() != information.cols() ||
      task_metric.rows() != information.rows() ||
      task_metric.cols() != information.cols() ||
      !information.allFinite() || !task_metric.allFinite() ||
      !(relative_rank_tolerance >= 0.0) ||
      !std::isfinite(relative_rank_tolerance)) return false;
  const Eigen::MatrixXd metric =
      0.5 * (task_metric + task_metric.transpose());
  const Eigen::MatrixXd symmetric_information =
      0.5 * (information + information.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> metric_eigen(metric);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full_information_eigen(
      symmetric_information, Eigen::EigenvaluesOnly);
  if (metric_eigen.info() != Eigen::Success ||
      full_information_eigen.info() != Eigen::Success) return false;
  const double metric_maximum = metric_eigen.eigenvalues().maxCoeff();
  if (!(metric_maximum > 0.0)) return false;
  const double information_scale = std::max(
      1.0, full_information_eigen.eigenvalues().cwiseAbs().maxCoeff());
  if (metric_eigen.eigenvalues().minCoeff() <
          -relative_rank_tolerance * std::max(1.0, metric_maximum) ||
      full_information_eigen.eigenvalues().minCoeff() <
          -relative_rank_tolerance * information_scale)
    return false;
  const double metric_threshold = relative_rank_tolerance *
      std::max(1.0, metric_maximum);
  std::vector<int> physical_modes;
  std::vector<int> task_null_modes;
  for (int mode = 0; mode < metric.rows(); ++mode)
    if (metric_eigen.eigenvalues()[mode] > metric_threshold)
      physical_modes.push_back(mode);
    else
      task_null_modes.push_back(mode);
  if (physical_modes.empty()) return false;
  Eigen::MatrixXd task_basis(
      metric.rows(), static_cast<int>(physical_modes.size()));
  Eigen::VectorXd task_eigenvalues(static_cast<int>(physical_modes.size()));
  for (size_t column = 0; column < physical_modes.size(); ++column) {
    const int mode = physical_modes[column];
    task_basis.col(static_cast<int>(column)) =
        metric_eigen.eigenvectors().col(mode);
    task_eigenvalues[static_cast<int>(column)] =
        metric_eigen.eigenvalues()[mode];
  }
  // Coefficients in null(task_metric) are unconstrained by a unit task
  // amplitude. Minimize them out with a Schur complement before measuring
  // information. Otherwise an observation of only
  // (task coefficient + interface/nuisance coefficient) would be counted as
  // task information even when those terms are wholly confounded.
  Eigen::MatrixXd task_information =
      task_basis.transpose() * symmetric_information * task_basis;
  if (!task_null_modes.empty()) {
    Eigen::MatrixXd null_basis(
        metric.rows(), static_cast<int>(task_null_modes.size()));
    for (size_t column = 0; column < task_null_modes.size(); ++column)
      null_basis.col(static_cast<int>(column)) =
          metric_eigen.eigenvectors().col(task_null_modes[column]);
    const Eigen::MatrixXd task_null_information =
        task_basis.transpose() * symmetric_information * null_basis;
    const Eigen::MatrixXd null_information =
        null_basis.transpose() * symmetric_information * null_basis;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> null_eigen(
        0.5 * (null_information + null_information.transpose()));
    if (null_eigen.info() != Eigen::Success) return false;
    const double null_maximum = std::max(
        0.0, null_eigen.eigenvalues().maxCoeff());
    const double null_threshold = relative_rank_tolerance *
        std::max(1.0, null_maximum);
    Eigen::MatrixXd null_pseudoinverse = Eigen::MatrixXd::Zero(
        null_information.rows(), null_information.cols());
    for (int mode = 0; mode < null_eigen.eigenvalues().size(); ++mode) {
      if (null_eigen.eigenvalues()[mode] <= null_threshold) continue;
      null_pseudoinverse +=
          null_eigen.eigenvectors().col(mode) *
          null_eigen.eigenvectors().col(mode).transpose() /
          null_eigen.eigenvalues()[mode];
    }
    task_information -= task_null_information * null_pseudoinverse *
                        task_null_information.transpose();
  }
  task_information = 0.5 *
      (task_information + task_information.transpose());
  Eigen::MatrixXd inverse_task_scale = Eigen::MatrixXd::Zero(
      task_eigenvalues.size(), task_eigenvalues.size());
  for (int mode = 0; mode < task_eigenvalues.size(); ++mode)
    inverse_task_scale(mode, mode) =
        1.0 / std::sqrt(task_eigenvalues[mode]);
  const Eigen::MatrixXd normalized_information =
      inverse_task_scale * task_information * inverse_task_scale;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> information_eigen(
      0.5 * (normalized_information + normalized_information.transpose()),
      Eigen::EigenvaluesOnly);
  if (information_eigen.info() != Eigen::Success ||
      !information_eigen.eigenvalues().allFinite()) return false;
  const double information_maximum = std::max(
      0.0, information_eigen.eigenvalues().maxCoeff());
  const double information_threshold = relative_rank_tolerance *
      std::max(1.0, information_maximum);
  int positive = 0;
  for (int mode = 0; mode < information_eigen.eigenvalues().size(); ++mode)
    if (information_eigen.eigenvalues()[mode] > information_threshold)
      ++positive;
  *minimum_information = std::max(
      0.0, information_eigen.eigenvalues().minCoeff());
  if (maximum_information)
    *maximum_information = information_maximum;
  if (effective_rank) *effective_rank = positive;
  if (task_rank) *task_rank = static_cast<int>(physical_modes.size());
  return true;
}

Eigen::Matrix3d UacGenerator(const std::array<double, 5>& rates) {
  const double q_ua = rates[0], q_uc = rates[1], q_au = rates[2];
  const double q_ac = rates[3], q_ca = rates[4];
  Eigen::Matrix3d generator;
  generator << -(q_ua + q_uc), q_ua, q_uc,
               q_au, -(q_au + q_ac), q_ac,
               0.0, q_ca, -q_ca;
  return generator;
}

Eigen::Matrix3d CtmcTransition(const Eigen::Matrix3d& generator, double delta_t,
                               double tolerance) {
  if (!(delta_t >= 0.0) || !generator.allFinite())
    return Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  if (delta_t == 0.0) return Eigen::Matrix3d::Identity();
  const double lambda = std::max({-generator(0, 0), -generator(1, 1),
                                  -generator(2, 2), 0.0});
  if (lambda == 0.0) return Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d embedded = Eigen::Matrix3d::Identity() + generator / lambda;
  const double mean = lambda * delta_t;
  double poisson_weight = std::exp(-mean);
  double cumulative_weight = poisson_weight;
  Eigen::Matrix3d power = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d transition = poisson_weight * power;
  const int hard_limit = std::max(64, static_cast<int>(std::ceil(mean + 20.0 * std::sqrt(mean + 1.0))));
  for (int n = 1; n <= hard_limit; ++n) {
    power = power * embedded;
    poisson_weight *= mean / static_cast<double>(n);
    transition += poisson_weight * power;
    cumulative_weight += poisson_weight;
    if (1.0 - cumulative_weight <= tolerance && n > mean) break;
  }
  // Round-off is projected back to the row stochastic simplex only at the
  // final epsilon scale; this is not a model transition or probability clamp.
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column)
      if (transition(row, column) < 0.0 && transition(row, column) > -1.0e-14)
        transition(row, column) = 0.0;
    const double sum = transition.row(row).sum();
    if (sum > 0.0) transition.row(row) /= sum;
  }
  return transition;
}

std::vector<RobustScaleNode> GaussianStudentTScaleMixture(
    double contamination_probability, double degrees_of_freedom,
    int quadrature_order) {
  std::vector<RobustScaleNode> nodes;
  if (!(contamination_probability >= 0.0 && contamination_probability < 1.0) ||
      !(degrees_of_freedom > 0.0) || quadrature_order <= 0) return nodes;
  nodes.push_back(RobustScaleNode{1.0 - contamination_probability, 1.0, 0});
  if (contamination_probability == 0.0) return nodes;

  const double alpha = degrees_of_freedom / 2.0 - 1.0;
  Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(quadrature_order, quadrature_order);
  for (int i = 0; i < quadrature_order; ++i) {
    jacobi(i, i) = 2.0 * static_cast<double>(i) + 1.0 + alpha;
    if (i > 0) {
      const double off = std::sqrt(static_cast<double>(i) *
                                   (static_cast<double>(i) + alpha));
      jacobi(i, i - 1) = off;
      jacobi(i - 1, i) = off;
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(jacobi);
  if (solver.info() != Eigen::Success) return std::vector<RobustScaleNode>();
  for (int i = 0; i < quadrature_order; ++i) {
    const double x = solver.eigenvalues()[i];
    const double normalized_weight = solver.eigenvectors()(0, i) * solver.eigenvectors()(0, i);
    RobustScaleNode node;
    node.prior_weight = contamination_probability * normalized_weight;
    node.precision = 2.0 * x / degrees_of_freedom;
    node.scale_index = i + 1;
    nodes.push_back(node);
  }
  return nodes;
}

double RobustGaussianStudentLogDensity(double residual, double latent_variance,
                                       double measurement_variance,
                                       const CalibrationConfig& calibration) {
  const std::vector<RobustScaleNode> nodes = GaussianStudentTScaleMixture(
      calibration.contamination_probability, calibration.student_t_dof, 8);
  std::vector<double> terms;
  terms.reserve(nodes.size());
  for (const RobustScaleNode& node : nodes) {
    if (!(node.prior_weight > 0.0) || !(node.precision > 0.0)) continue;
    const bool core = node.scale_index == 0;
    const double scale2 = core ? 1.0 : calibration.student_t_scale * calibration.student_t_scale;
    const double variance = latent_variance + measurement_variance * scale2 / node.precision;
    terms.push_back(std::log(node.prior_weight) + GaussianLogDensity(residual, variance));
  }
  return LogSumExp(terms);
}

double InverseNormalCdf(double probability) {
  // Acklam's approximation. Accuracy is substantially below the calibration
  // uncertainty used for the MDE gate.
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
      -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
      -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
      2.445134137142996e+00, 3.754408661907416e+00};
  if (probability <= 0.0) return -std::numeric_limits<double>::infinity();
  if (probability >= 1.0) return std::numeric_limits<double>::infinity();
  if (probability < 0.02425) {
    const double q = std::sqrt(-2.0 * std::log(probability));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  }
  if (probability > 0.97575) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  }
  const double q = probability - 0.5;
  const double r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
         (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
}

double MinimumDetectableEffect(double standard_deviation,
                               double false_alarm_probability,
                               double target_power) {
  if (!(standard_deviation > 0.0) || !std::isfinite(standard_deviation))
    return std::numeric_limits<double>::infinity();
  return MinimumDetectableEffectFromInformation(
      1.0 / (standard_deviation * standard_deviation), 1,
      false_alarm_probability, target_power);
}

double MinimumDetectableEffectFromInformation(
    double directional_information, int effective_rank,
    double false_alarm_probability, double target_power) {
  if (!(directional_information > 0.0) ||
      !std::isfinite(directional_information) || effective_rank <= 0 ||
      !(false_alarm_probability > 0.0 && false_alarm_probability < 1.0) ||
      !(target_power > false_alarm_probability && target_power < 1.0))
    return std::numeric_limits<double>::infinity();
  try {
    const boost::math::chi_squared_distribution<double> null_distribution(
        static_cast<double>(effective_rank));
    const double critical = boost::math::quantile(
        null_distribution, 1.0 - false_alarm_probability);
    auto power = [&](double noncentrality) {
      const boost::math::non_central_chi_squared_distribution<double>
          alternative(static_cast<double>(effective_rank), noncentrality);
      return boost::math::cdf(boost::math::complement(alternative, critical));
    };
    double lower = 0.0;
    double upper = 1.0;
    while (power(upper) < target_power && upper < 1.0e12) upper *= 2.0;
    if (power(upper) < target_power) return std::numeric_limits<double>::infinity();
    // Fixed iterations make the inversion deterministic and far more accurate
    // than any calibrated uncertainty retained by this V1 model.
    for (int iteration = 0; iteration < 96; ++iteration) {
      const double midpoint = 0.5 * (lower + upper);
      if (power(midpoint) >= target_power) upper = midpoint;
      else lower = midpoint;
    }
    return std::sqrt(upper / directional_information);
  } catch (...) {
    return std::numeric_limits<double>::infinity();
  }
}

ProbabilityInterval ConservativeUnion(
    const std::vector<ProbabilityInterval>& probabilities,
    double numerical_error_upper) {
  ProbabilityInterval result;
  if (probabilities.empty()) {
    result.mean = 0.0;
    result.upper = std::max(0.0, std::min(1.0, numerical_error_upper));
    result.calibrated = true;
    result.mean_identified = true;
    return result;
  }
  double lower = 0.0, upper_sum = 0.0, mean_sum = 0.0;
  bool calibrated = true;
  bool all_means_identified = true;
  for (const ProbabilityInterval& probability : probabilities) {
    lower = std::max(lower, probability.lower);
    upper_sum += probability.upper;
    mean_sum += probability.mean;
    calibrated = calibrated && probability.calibrated;
    all_means_identified = all_means_identified &&
                           probability.mean_identified;
  }
  result.lower = std::max(0.0, lower - numerical_error_upper);
  result.upper = std::min(1.0, upper_sum + numerical_error_upper);
  result.calibrated = calibrated;
  if (probabilities.size() == 1U) {
    result.mean = probabilities.front().mean;
    result.mean_identified = all_means_identified;
  } else {
    // The marginals do not identify E[1_{union}] without their joint law.
    // Store the certified upper endpoint in the legacy scalar slot so an
    // unaware consumer cannot obtain a less conservative decision value.
    (void)mean_sum;
    result.mean = result.upper;
    result.mean_identified = false;
  }
  return result;
}

}  // namespace v1
}  // namespace deform_monitor_v3
