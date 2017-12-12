#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
    0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
    0, 0.0009, 0,
    0, 0, 0.09;

  // measurement matrix - laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
    0, 1, 0, 0;

  // measurement Jacobian matrix - radar
  Hj_radar_ = MatrixXd::Zero(3, 4);

  z_pred_laser_ = VectorXd(2);
  z_pred_radar_ = VectorXd(3);

  y_laser_ = VectorXd(2);
  y_radar_ = VectorXd(3);

  // create a 4D state vector
  ekf_.x_ = VectorXd(4);

  // state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1000, 0,
    0, 0, 0, 1000;

  // the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,
    0, 0, 0, 1;

  ekf_.Q_  = MatrixXd::Zero(4, 4);
  noise_ax = 9;
  noise_ay = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage& measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float px = measurement_pack.raw_measurements_[0] * cos(
        measurement_pack.raw_measurements_[1]);
      float py = measurement_pack.raw_measurements_[0] * sin(
        measurement_pack.raw_measurements_[1]);
      ekf_.x_ << px, py, 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_[0],
        measurement_pack.raw_measurements_[1], 0, 0;
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_     = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

  // save some computation cycles if dt is too small
  if (dt > 0.001) {
    previous_timestamp_ = measurement_pack.timestamp_;

    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_3 * dt;

    // Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<  dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
      0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
      dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

    ekf_.Predict();
  }

  /*****************************************************************************
   *  Measurement Update
   ****************************************************************************/
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    auto x = ekf_.x_;

    // calculate z_predict using it nonlinear functions
    double c0 = sqrt(x[0] * x[0] + x[1] * x[1]);

    if (c0 == 0) {
      z_pred_radar_ << 0, 0, 0;
    } else {
      z_pred_radar_ << c0, atan2(x[1], x[0]), (x[0] * x[2] + x[1] * x[3]) / c0;
    }

    y_radar_ = measurement_pack.raw_measurements_ - z_pred_radar_;

    // adjust delta phi if needed
    y_radar_[1] = atan2(sin(y_radar_[1]), cos(y_radar_[1]));

    // calculate measurement Jacobian matrix
    Hj_radar_ = tools.CalculateJacobian(ekf_.x_);

    // Radar updates
    ekf_.MeasurementUpdate(y_radar_,
                           Hj_radar_,
                           R_radar_);
  } else {
    // Laser Updates
    z_pred_laser_ = H_laser_ * ekf_.x_;
    y_laser_      = measurement_pack.raw_measurements_ - z_pred_laser_;
    ekf_.MeasurementUpdate(y_laser_,
                           H_laser_,
                           R_laser_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
