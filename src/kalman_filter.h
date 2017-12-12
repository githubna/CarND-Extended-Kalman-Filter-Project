#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

class KalmanFilter {
public:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // process noise covariance matrix
  Eigen::MatrixXd Q_;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   */
  void Init(Eigen::VectorXd& x_in,
            Eigen::MatrixXd& P_in,
            Eigen::MatrixXd& F_in,
            Eigen::MatrixXd& Q_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   */
  void Predict();

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param y The difference in between measurement value and predicted
   *measurement value at k+1
   * @param Hj The Jacobian measurement matrix at K+1
   * @param R The measurement noise covariance matrix
   */
  void MeasurementUpdate(const Eigen::VectorXd& y,
                         const Eigen::MatrixXd& H,
                         const Eigen::MatrixXd& R);
};

#endif /* KALMAN_FILTER_H_ */
