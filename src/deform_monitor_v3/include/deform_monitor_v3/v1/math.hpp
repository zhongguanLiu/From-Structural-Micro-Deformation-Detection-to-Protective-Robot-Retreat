#ifndef DEFORM_MONITOR_V3_V1_MATH_HPP
#define DEFORM_MONITOR_V3_V1_MATH_HPP

#include "deform_monitor_v3/v1/types.hpp"

#include <Eigen/Dense>

#include <array>
#include <utility>
#include <vector>

namespace deform_monitor_v3 {
namespace v1 {

double ClampProbability(double probability, double floor);
double LogSumExp(const std::vector<double>& terms);
std::vector<double> Softmax(const std::vector<double>& log_scores);
double GaussianLogDensity(double residual, double variance);
double LogNegativeBinomialPmf(int count, double mean, double dispersion);
double LogVisibilityMixedNegativeBinomial(int count, double mean, double dispersion,
                                          double visibility, double probability_floor);

double WendlandC2(double normalized_distance);
Eigen::Matrix3d Skew(const Eigen::Vector3d& vector);

bool ValidateSymmetricPsd(const Eigen::MatrixXd& matrix, double tolerance,
                          double* minimum_eigenvalue = nullptr);
bool StableCholesky(const Eigen::MatrixXd& matrix, const NumericalConfig& numerical,
                    Eigen::MatrixXd* lower, double* jitter_used = nullptr,
                    int* attempts = nullptr);

// Fixed-coordinate semidefinite Cholesky factor.  Unlike an eigenvector-based
// square root, this factor is unique for the configured coordinate ordering
// even when covariance eigenvalues are repeated.  No jitter is added: callers
// obtain L such that L L^T equals the admitted PSD matrix, or a hard failure.
bool DeterministicPsdSquareRoot(const Eigen::MatrixXd& matrix,
                                const NumericalConfig& numerical,
                                Eigen::MatrixXd* lower);

// Rectangular square-root information factor R for a PSD covariance P.  The
// rows span only the admitted covariance range and satisfy R^T R = P^+.
// Columns below the configured PSD rank threshold are removed in coordinate
// order; no eigenvector/SVD basis is selected inside repeated eigenspaces.
bool DeterministicPsdInformationFactor(const Eigen::MatrixXd& matrix,
                                       const NumericalConfig& numerical,
                                       Eigen::MatrixXd* information_factor);

// Moore-Penrose inverse reconstructed from the rectangular information
// factor.  This helper is retained for algorithms that explicitly require the
// dense inverse; public Gaussian components should publish the factor itself.
bool DeterministicPsdPseudoInverse(const Eigen::MatrixXd& matrix,
                                   const NumericalConfig& numerical,
                                   Eigen::MatrixXd* pseudoinverse);

// Minimum retained information on the positive range of known_information:
// lambda_min(I_known^{-1/2} I_marginal I_known^{-1/2}).  Null directions of
// I_known are deliberately excluded because they were never observable.
bool MinimumRetainedInformationRatio(
    const Eigen::MatrixXd& known_information,
    const Eigen::MatrixXd& marginalized_information,
    double relative_rank_tolerance, double* ratio,
    int* effective_rank = nullptr);

// Weakest Fisher information for one metre of task-field RMS, where
// a^T task_metric a is the squared physical field amplitude.  Coefficients in
// null(task_metric) are minimized out (Schur complement), not fixed to zero.
// This prevents both control-grid refinement and task-null confounding from
// silently changing MDE units or overstating power.
bool MinimumTaskNormalizedInformation(
    const Eigen::MatrixXd& information,
    const Eigen::MatrixXd& task_metric,
    double relative_rank_tolerance,
    double* minimum_information,
    int* effective_rank = nullptr,
    int* task_rank = nullptr,
    double* maximum_information = nullptr);

Eigen::Matrix3d UacGenerator(const std::array<double, 5>& rates);
Eigen::Matrix3d CtmcTransition(const Eigen::Matrix3d& generator, double delta_t,
                               double tolerance = 1.0e-15);

struct RobustScaleNode {
  double prior_weight = 0.0;
  // Precision multiplier lambda. The contaminated measurement variance is
  // kappa^2 R / lambda.
  double precision = 1.0;
  int scale_index = 0;
};

std::vector<RobustScaleNode> GaussianStudentTScaleMixture(
    double contamination_probability, double degrees_of_freedom,
    int quadrature_order);

double RobustGaussianStudentLogDensity(double residual, double latent_variance,
                                       double measurement_variance,
                                       const CalibrationConfig& calibration);

double InverseNormalCdf(double probability);
double MinimumDetectableEffect(double standard_deviation,
                               double false_alarm_probability,
                               double target_power);
double MinimumDetectableEffectFromInformation(double directional_information,
                                               int effective_rank,
                                               double false_alarm_probability,
                                               double target_power);

ProbabilityInterval ConservativeUnion(
    const std::vector<ProbabilityInterval>& probabilities,
    double numerical_error_upper);

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_MATH_HPP
