#include "ConvexMPCLocomotion.h"

#include <iostream>
#include <qpOASES.hpp>
// #include "Utilities/Timer.h"
// #include "Utilities/Utilities_print.h"
#include "convexMPC_interface.h"

#include <chrono>
using namespace std::chrono;
using namespace std::chrono_literals;
// #include "../../../../common/FootstepPlanner/GraphSearch.h"

// #include "Gait.h"

//#define DRAW_DEBUG_SWINGS
//#define DRAW_DEBUG_PATH

////////////////////
// Controller
// 参考Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive
// Control 一个步态周期由horizonLength(10)个mpc周期组成 步态按1KHz处理
// mpc计数间隔为30左右 一毫秒计数一次来控制频率 即一个mpc周期为30ms 则步态周期为
// 10*30 =300ms
////////////////////
Eigen::Matrix3d skewMat(Eigen::Vector3d input) {
  Eigen::Matrix3d skewMat;
  skewMat << 0.0, -input(2), input(1), input(2), 0.0, -input(0), -input(1),
      input(0), 0.0;
  return skewMat;
}

Eigen::Vector3d quat_to_ypr(Eigen::Quaterniond q) {
  Eigen::Vector3d output;
  double as = fmin(-2. * (q.x() * q.z() - q.w() * q.y()), .99999);
  output(0) =
      atan2(2. * (q.x() * q.y() + q.w() * q.z()),
            q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z());
  output(1) = asin(as);
  output(2) =
      atan2(2. * (q.y() * q.z() + q.w() * q.x()),
            q.w() * q.w() - q.x() * q.x() - q.y() * q.y() + q.z() * q.z());

  return output;
}

template <typename Derived>
Eigen::MatrixXd createBlockVertical(const Eigen::MatrixBase<Derived> &a,
                                    const unsigned count) {
  Eigen::MatrixXd bvm = Eigen::MatrixXd::Zero(a.rows() * count, a.cols());
  for (size_t i = 0; i < count; ++i) {
    bvm.block(i * a.rows(), 0, a.rows(), a.cols()) = a;
  }

  return bvm;
}

template <typename Derived>
Eigen::MatrixXd createBlockDiagonal(const Eigen::MatrixBase<Derived> &a,
                                    const unsigned count) {
  Eigen::MatrixXd bdm =
      Eigen::MatrixXd::Zero(a.rows() * count, a.cols() * count);
  for (size_t i = 0; i < count; ++i) {
    bdm.block(i * a.rows(), i * a.cols(), a.rows(), a.cols()) = a;
  }

  return bdm;
}

ConvexMPCLocomotion::ConvexMPCLocomotion(float _dt, int _iterations_between_mpc)
    : iterationsBetweenMPC(_iterations_between_mpc), //控制频率用  15
      horizonLength(10), dt(_dt),                    // 0.002
      trotting(horizonLength,
               Vec4<int>(0, horizonLength / 2.0, horizonLength / 2.0, 0),
               Vec4<int>(horizonLength / 2.0, horizonLength / 2.0,
                         horizonLength / 2.0, horizonLength / 2.0),
               "Trotting"),
      bounding(horizonLength, Vec4<int>(5, 5, 0, 0), Vec4<int>(4, 4, 4, 4),
               "Bounding"),
      // bounding(horizonLength,
      // Vec4<int>(5,5,0,0),Vec4<int>(3,3,3,3),"Bounding"),
      pronking(horizonLength, Vec4<int>(0, 0, 0, 0), Vec4<int>(4, 4, 4, 4),
               "Pronking"),
      jumping(horizonLength, Vec4<int>(0, 0, 0, 0), Vec4<int>(2, 2, 2, 2),
              "Jumping"),
      galloping(horizonLength, Vec4<int>(0, 3, 6, 9), Vec4<int>(5, 5, 5, 5),
                "Galloping"),
      standing(horizonLength, Vec4<int>(0, 0, 0, 0), Vec4<int>(10, 19, 10, 10),
               "Standing"),
      trotRunning(horizonLength, Vec4<int>(0, 5, 5, 0), Vec4<int>(4, 4, 4, 4),
                  "Trot Running"),
      walking(horizonLength,
              Vec4<int>(0, horizonLength / 2.0, horizonLength / 4.0,
                        3.0 * horizonLength / 4.0),
              Vec4<int>(3.0 * horizonLength / 4.0, 3.0 * horizonLength / 4.0,
                        3.0 * horizonLength / 4.0, 3.0 * horizonLength / 4.0),
              "Walking"),
      walking2(horizonLength, Vec4<int>(0, 5, 5, 0), Vec4<int>(7, 7, 7, 7),
               "Walking2"),
      pacing(horizonLength, Vec4<int>(6, 0, 6, 0), Vec4<int>(6, 6, 6, 6),
             "Pacing"),
      aio(horizonLength, Vec4<int>(0, 0, 0, 0), Vec4<int>(10, 10, 10, 10),
          "aio") {
  dtMPC = dt * iterationsBetweenMPC; // 0.03
  default_iterations_between_mpc = iterationsBetweenMPC;
  printf("[Convex MPC] dt: %.3f iterations: %d, dtMPC: %.3f\n", dt,
         iterationsBetweenMPC, dtMPC); // 0.002, 15, 0.03
  // setup_problem(dtMPC, horizonLength, 0.4, 120);
  // setup_problem(dtMPC, horizonLength, 0.4, 650); // DH

  for (int i = 0; i < 4; i++)
    firstSwing[i] = true;

  // initSparseMPC();

  pBody_des.setZero();
  vBody_des.setZero();
  for (int i = 0; i < 4; i++)
    f_ff[i].setZero();
}

void ConvexMPCLocomotion::initialize() {
  for (int i = 0; i < 4; i++)
    firstSwing[i] = true;
  firstRun = true;
}

void ConvexMPCLocomotion::recompute_timing(int iterations_per_mpc) {
  iterationsBetweenMPC = iterations_per_mpc;
  dtMPC = dt * iterations_per_mpc;
}

//设置期望值
void ConvexMPCLocomotion::_SetupCommand(
    StateEstimatorContainer<float> &_stateEstimator,
    std::vector<double> gamepadCommand) {
  _body_height = 0.29;

  float x_vel_cmd, y_vel_cmd, yaw_vel_cmd;
  float x_filter(0.01), y_filter(0.006), yaw_filter(0.03);

  //手柄数据先暂时设置为0，后面再给手柄赋值   旋转角速度和x,y方向上的线速度
  x_vel_cmd = gamepadCommand[0];
  y_vel_cmd = gamepadCommand[1];
  yaw_vel_cmd = gamepadCommand[2];

  _x_vel_des = x_vel_cmd; //一阶低通数字滤波
  _y_vel_des = y_vel_cmd;
  _yaw_turn_rate = yaw_vel_cmd;
  _yaw_des = _stateEstimator.getResult().rpy[2] +
             dt * _yaw_turn_rate; //涉及到了状态估计中的欧拉角

  _roll_des = 0.;
  _pitch_des = 0.;
}

template <>
void ConvexMPCLocomotion::run(Quadruped<float> &_quadruped,
                              LegController<float> &_legController,
                              StateEstimatorContainer<float> &_stateEstimator,
                              DesiredStateCommand<float> &_desiredStateCommand,
                              std::vector<double> gamepadCommand, int gaitType,
                              int robotMode) {
  bool omniMode = false;
  // Command Setup
  _SetupCommand(_stateEstimator, gamepadCommand);

  auto &seResult = _stateEstimator.getResult(); //状态估计器

  // Check if transition to standing 检查是否过渡到站立
  if (((gaitNumber == 4) && current_gait != 4) || firstRun) {
    world_position_desired[0] = seResult.position[0];
    world_position_desired[1] = seResult.position[1];
  }

  // pick gait
  Gait *gait = &trotting;

  gait->setIterations(iterationsBetweenMPC, iterationCounter); //步态周期计算

  // integrate position setpoint
  Vec3<float> v_des_robot(_x_vel_des, _y_vel_des,
                          0); //身体坐标系下的期望线速度
  Vec3<float> v_des_world = seResult.rBody.transpose() * v_des_robot;
  Vec3<float> v_robot = seResult.vWorld; //世界坐标系下的机器人实际速度
  for (int i = 0; i < 4; i++) {
    pFoot[i] = seResult.position +
               seResult.rBody.transpose() *
                   (_quadruped.getHipLocation(i) + _legController.datas[i].p);
    // pFoot[i] = _legController.datas[i].p;
  }

  Vec3<float> error;
  if (gait != &standing) { //非站立下的期望位置，通过累加期望速度完成
    world_position_desired +=
        dt * Vec3<float>(v_des_world[0], v_des_world[1], 0);
  }

  // some first time initialization
  if (firstRun) {
    world_position_desired[0] = seResult.position[0];
    world_position_desired[1] = seResult.position[1];
    world_position_desired[2] = seResult.rpy[2];

    for (int i = 0; i < 4; i++) //足底摆动轨迹
    {
      footSwingTrajectories[i].setHeight(0.06);
      footSwingTrajectories[i].setInitialPosition(pFoot[i]); // set p0
      footSwingTrajectories[i].setFinalPosition(pFoot[i]);   // set pf
    }
    firstRun = false;
  }

  // foot placement
  for (int l = 0; l < 4; l++) {
    swingTimes[l] = gait->getCurrentSwingTime(
        dtMPC, l); // return dtMPC * _stance  0.026 * 5 = 0.13
                   // dtMPC的值变为了0.026，外部给修改的赋值
  }

  float side_sign[4] = {-1, 1, -1, 1};
  float v_abs = std::fabs(v_des_robot[0]);
  for (int i = 0; i < 4; i++) {
    if (firstSwing[i]) {
      swingTimeRemaining[i] = swingTimes[i];
    } else {
      swingTimeRemaining[i] -= dt;
    }

    footSwingTrajectories[i].setHeight(0.06);
    Vec3<float> offset(0, side_sign[i] * .065, 0);

    Vec3<float> pRobotFrame =
        (_quadruped.getHipLocation(i) + offset); //得到身体坐标系下的hip关节坐标

    float stance_time =
        gait->getCurrentStanceTime(dtMPC, i); // stance_time = 0.13

    Vec3<float> pYawCorrected =
        coordinateRotation(CoordinateAxis::Z,
                           -_yaw_turn_rate * stance_time / 2) *
        pRobotFrame; //机身旋转yaw后，得到在机身坐标系下的hip坐标

    Vec3<float> des_vel;
    des_vel[0] = _x_vel_des;
    des_vel[1] = _y_vel_des;
    des_vel[2] = 0.0;

    //世界坐标系下hip坐标 以剩余摆动时间内匀速运动来估计
    Vec3<float> Pf = seResult.position +
                     seResult.rBody.transpose() *
                         (pYawCorrected + des_vel * swingTimeRemaining[i]);
    float p_rel_max = 0.3f;

    float pfx_rel = seResult.vWorld[0] * .5 * stance_time +
                    .03f * (seResult.vWorld[0] - v_des_world[0]) +
                    (0.5f * sqrt(seResult.position[2] / 9.81f)) *
                        (seResult.vWorld[1] * _yaw_turn_rate);

    float pfy_rel = seResult.vWorld[1] * .5 * stance_time +
                    .03f * (seResult.vWorld[1] - v_des_world[1]) +
                    (0.5f * sqrt(seResult.position[2] / 9.81f)) *
                        (-seResult.vWorld[0] * _yaw_turn_rate);
    pfx_rel = fminf(fmaxf(pfx_rel, -p_rel_max), p_rel_max);
    pfy_rel = fminf(fmaxf(pfy_rel, -p_rel_max), p_rel_max);
    Pf[0] += pfx_rel;
    Pf[1] += pfy_rel;
    Pf[2] = 0.0;
    footSwingTrajectories[i].setFinalPosition(
        Pf); //最终得到足底的位置，并作为轨迹终点 世界坐标系下的落足点
  }
  // std::cout << std::endl;

  // calc gait
  iterationCounter++;

  // load LCM leg swing gains
  Kp << 700, 0, 0, 0, 700, 0, 0, 0, 200;
  Kp_stance = 0.0 * Kp;

  Kd << 10, 0, 0, 0, 10, 0, 0, 0, 10;
  Kd_stance = 1.0 * Kd;
  // gait
  Vec4<float> contactStates = gait->getContactState();
  Vec4<float> swingStates = gait->getSwingState();
  int *mpcTable = gait->getMpcTable();

  Eigen::MatrixXi gaitTable(10, 4);
  for (int ver = 0; ver < 10; ver++) {
    for (int hor = 0; hor < 4; hor++) {
      gaitTable(ver, hor) = mpcTable[ver * 4 + hor];
    }
  }
  // std::cout << gaitTable << std::endl;
  UserInput userInput;
  userInput.x_vel_cmd = _x_vel_des;
  userInput.y_vel_cmd = _y_vel_des;
  userInput.yaw_turn_rate = _yaw_turn_rate;
  userInput.yaw_des = _yaw_des;
  userInput.p_des[0] = world_position_desired[0];
  userInput.p_des[1] = world_position_desired[1];
  userInput.p_des[2] = 0.29;
  RobotSystemOutputData robotSystemData;
  robotSystemData.worldToBaseRotMat(0, 0) = seResult.rBody(0, 0);
  robotSystemData.worldToBaseRotMat(1, 0) = seResult.rBody(1, 0);
  robotSystemData.worldToBaseRotMat(2, 0) = seResult.rBody(2, 0);
  robotSystemData.worldToBaseRotMat(0, 1) = seResult.rBody(0, 1);
  robotSystemData.worldToBaseRotMat(1, 1) = seResult.rBody(1, 1);
  robotSystemData.worldToBaseRotMat(2, 1) = seResult.rBody(2, 1);
  robotSystemData.worldToBaseRotMat(0, 2) = seResult.rBody(0, 2);
  robotSystemData.worldToBaseRotMat(1, 2) = seResult.rBody(1, 2);
  robotSystemData.worldToBaseRotMat(2, 2) = seResult.rBody(2, 2);
  robotSystemData.baseToWorldRotMat =
      robotSystemData.worldToBaseRotMat.transpose();
  robotSystemData.worldPosition[0] = (double)seResult.position[0];
  robotSystemData.worldPosition[1] = (double)seResult.position[1];
  robotSystemData.worldPosition[2] = (double)seResult.position[2];
  robotSystemData.worldLinearVelocity[0] = (double)seResult.vWorld[0];
  robotSystemData.worldLinearVelocity[1] = (double)seResult.vWorld[1];
  robotSystemData.worldLinearVelocity[2] = (double)seResult.vWorld[2];
  robotSystemData.worldAngularVelocity[0] = (double)seResult.omegaWorld[0];
  robotSystemData.worldAngularVelocity[1] = (double)seResult.omegaWorld[1];
  robotSystemData.worldAngularVelocity[2] = (double)seResult.omegaWorld[2];
  robotSystemData.baseRotation.w() = seResult.orientation[0];
  robotSystemData.baseRotation.x() = seResult.orientation[1];
  robotSystemData.baseRotation.y() = seResult.orientation[2];
  robotSystemData.baseRotation.z() = seResult.orientation[3];
  if ((iterationCounter % iterationsBetweenMPC) == 0) {
    std::cout << "**************\n";
    auto start2 = high_resolution_clock::now();

    updateMPCIfNeeded(mpcTable, _stateEstimator, omniMode);
    for (int i = 0; i < 4; i++) {
      std::cout << f_ff[i].transpose() << std::endl;
    }
    auto end2 = high_resolution_clock::now();
    std::cout << "time taken: "
              << duration_cast<microseconds>(end2 - start2).count() << "us"
              << std::endl;
    std::cout << "@@@@@@@@@@@@@@@@@@@\n";
    auto start = high_resolution_clock::now();

    auto mpcData = updateMPC(robotSystemData, userInput, gaitTable);
    auto preRotationTorque = solveMPC(mpcData);
    if (robotMode == 1) {
      for (int i = 0; i < 4; i++) {
        auto tempTorque = -robotSystemData.baseToWorldRotMat *
                          preRotationTorque.block<3, 1>(i * 3, 0);
        for (int j = 0; j < 3; j++)
          f_ff[i][j] = tempTorque[j];
        std::cout << f_ff[i].transpose() << std::endl;
      }
    } else {
      for (int i = 0; i < 4; i++) {
        auto tempTorque = -robotSystemData.baseToWorldRotMat *
                          preRotationTorque.block<3, 1>(i * 3, 0);
        for (int j = 0; j < 3; j++)
          f_ff2[i][j] = tempTorque[j];
        std::cout << f_ff2[i].transpose() << std::endl;
      }
    }
    auto end = high_resolution_clock::now();
    std::cout << "time taken: "
              << duration_cast<microseconds>(end - start).count() << "us"
              << std::endl;
    std::cout << "**************\n\n";
  }
  //  StateEstimator* se = hw_i->state_estimator;
  Vec4<float> se_contactState(0, 0, 0, 0);

  bool use_wbc = false;

  for (int foot = 0; foot < 4; foot++) {
    float contactState = contactStates[foot];
    float swingState = swingStates[foot];
    if (swingState > 0) // foot is in swing
    {
      if (firstSwing[foot]) {
        firstSwing[foot] = false;
        footSwingTrajectories[foot].setInitialPosition(pFoot[foot]);
      }

      footSwingTrajectories[foot].computeSwingTrajectoryBezier(
          swingState, swingTimes[foot]);

      Vec3<float> pDesFootWorld = footSwingTrajectories[foot].getPosition();
      Vec3<float> vDesFootWorld = footSwingTrajectories[foot].getVelocity();

      Vec3<float> pDesLeg =
          seResult.rBody *
              (pDesFootWorld -
               seResult.position) //侧摆关节坐标系下的足端坐标
                                  //(此处先改为身体坐标系下的足端坐标)
          - _quadruped.getHipLocation(foot);
      Vec3<float> vDesLeg = seResult.rBody * (vDesFootWorld - seResult.vWorld);

      // Update for WBC
      pFoot_des[foot] = pDesFootWorld;
      vFoot_des[foot] = vDesFootWorld;
      aFoot_des[foot] = footSwingTrajectories[foot].getAcceleration();

      if (!use_wbc) {
        // Update leg control command regardless of the usage of WBIC
        _legController.commands[foot].pDes = pDesLeg;
        _legController.commands[foot].vDes = vDesLeg;
        if (foot == 1 || foot == 3) {
          _legController.commands[foot].kpCartesian = Kp;
          _legController.commands[foot].kdCartesian = Kd;
        } else {
          _legController.commands[foot].kpCartesian = 1 * Kp;
          _legController.commands[foot].kdCartesian = 1 * Kd;
        }
      }
    } else // foot is in stance
    {
      firstSwing[foot] = true;

      Vec3<float> pDesFootWorld = footSwingTrajectories[foot].getPosition();
      Vec3<float> vDesFootWorld = footSwingTrajectories[foot].getVelocity();
      Vec3<float> pDesLeg =
          seResult.rBody * (pDesFootWorld - seResult.position) -
          _quadruped.getHipLocation(foot);
      Vec3<float> vDesLeg = seResult.rBody * (vDesFootWorld - seResult.vWorld);

      if (!use_wbc) {
        _legController.commands[foot].pDes = pDesLeg;
        _legController.commands[foot].vDes = vDesLeg;

        if (foot == 1 || foot == 3) {
          _legController.commands[foot].kdCartesian = Kd_stance;
        } else {
          _legController.commands[foot].kdCartesian = 1 * Kd_stance;
        }

        _legController.commands[foot].forceFeedForward = f_ff[foot];
        _legController.commands[foot].kdJoint = Mat3<float>::Identity() * 0.2;

      } else { // Stance foot damping
        _legController.commands[foot].pDes = pDesLeg;
        _legController.commands[foot].vDes = vDesLeg;
        _legController.commands[foot].kpCartesian = 0. * Kp_stance;
        _legController.commands[foot].kdCartesian = Kd_stance;
      }
      se_contactState[foot] = contactState;

      // Update for WBC
      // Fr_des[foot] = -f_ff[foot];
    }
  }
  // se->set_contact_state(se_contactState); todo removed
  _stateEstimator.setContactPhase(se_contactState);

  // Update For WBC
  pBody_des[0] = world_position_desired[0];
  pBody_des[1] = world_position_desired[1];
  pBody_des[2] = _body_height;

  vBody_des[0] = v_des_world[0];
  vBody_des[1] = v_des_world[1];
  vBody_des[2] = 0.;

  pBody_RPY_des[0] = 0.;
  pBody_RPY_des[1] = 0.;
  pBody_RPY_des[2] = _yaw_des;

  vBody_Ori_des[0] = 0.;
  vBody_Ori_des[1] = 0.;
  vBody_Ori_des[2] = _yaw_turn_rate;
}

void ConvexMPCLocomotion::updateMPCIfNeeded(
    int *mpcTable, StateEstimatorContainer<float> &_stateEstimator,
    bool omniMode) {
  // iterationsBetweenMPC = 30;
  if ((iterationCounter % iterationsBetweenMPC) == 0) {
    auto seResult = _stateEstimator.getResult();
    float *p = seResult.position.data();

    Vec3<float> v_des_robot(_x_vel_des, _y_vel_des, 0);
    Vec3<float> v_des_world = seResult.rBody.transpose() * v_des_robot;
    // float trajInitial[12] = {0,0,0, 0,0,.25, 0,0,0,0,0,0};

    // printf("Position error: %.3f, integral %.3f\n", pxy_err[0],
    // x_comp_integral);

    if (current_gait == 4) {
      float trajInitial[12] = {
          _roll_des,
          _pitch_des /*-hw_i->state_estimator->se_ground_pitch*/,
          seResult.rpy[2] /*+(float)stateCommand->data.stateDes[11]*/,
          seResult.position[0] /*+(float)fsm->main_control_settings.p_des[0]*/,
          seResult.position[1] /*+(float)fsm->main_control_settings.p_des[1]*/,
          0.29 /*fsm->main_control_settings.p_des[2]*/,
          0,
          0,
          0,
          0,
          0,
          0};

      for (int i = 0; i < horizonLength; i++)
        for (int j = 0; j < 12; j++)
          trajAll[12 * i + j] = trajInitial[j];
    }

    else {
      const float max_pos_error = .1;
      float xStart = world_position_desired[0];
      float yStart = world_position_desired[1];

      if (xStart - p[0] > max_pos_error)
        xStart = p[0] + 0.1;
      if (p[0] - xStart > max_pos_error)
        xStart = p[0] - 0.1;

      if (yStart - p[1] > max_pos_error)
        yStart = p[1] + 0.1;
      if (p[1] - yStart > max_pos_error)
        yStart = p[1] - 0.1;

      world_position_desired[0] = xStart;
      world_position_desired[1] = yStart;

      float trajInitial[12] = {0.0,      // 0
                               0.0,      // 1
                               _yaw_des, // 2
                               // yawStart,    // 2
                               xStart,              // 3
                               yStart,              // 4
                               (float)_body_height, // 5
                               0,                   // 6
                               0,                   // 7
                               _yaw_turn_rate,      // 8
                               v_des_world[0],      // 9
                               v_des_world[1],      // 10
                               0};                  // 11

      for (int i = 0; i < horizonLength; i++) {
        for (int j = 0; j < 12; j++)
          trajAll[12 * i + j] = trajInitial[j];

        if (i == 0) // start at current position  TODO consider not doing this
        {
          trajAll[2] = seResult.rpy[2];
        } else {
          trajAll[12 * i + 3] =
              trajAll[12 * (i - 1) + 3] + dtMPC * v_des_world[0];
          trajAll[12 * i + 4] =
              trajAll[12 * (i - 1) + 4] + dtMPC * v_des_world[1];
          trajAll[12 * i + 2] =
              trajAll[12 * (i - 1) + 2] + dtMPC * _yaw_turn_rate;
        }
      }
      // for (int i = 0; i < 10; i++) {
      //   for (int j = 0; j < 12; j++) {
      //     std::cout << trajAll[12 * i + j] << " ";
      //   }
      //   std::cout << std::endl;
      // }
    }
  }
  solveDenseMPC(mpcTable, _stateEstimator);
}

void ConvexMPCLocomotion::solveDenseMPC(
    int *mpcTable, StateEstimatorContainer<float> &_stateEstimator) {
  auto seResult = _stateEstimator.getResult();
  float Q[12] = {2.5, 2.5, 10, 50, 50, 100, 0, 0, 0.5, 0.2, 0.2, 0.1};

  float yaw = seResult.rpy[2];
  float *weights = Q;
  float alpha = 4e-5; // make setting eventually
  // float alpha = 4e-7; // make setting eventually: DH
  float *p = seResult.position.data();
  float *v = seResult.vWorld.data();
  float *w = seResult.omegaWorld.data();
  float *q = seResult.orientation.data();

  float r[12];
  for (int i = 0; i < 12; i++)
    r[i] = pFoot[i % 4][i / 4] - seResult.position[i / 4];

  // printf("current posistion: %3.f %.3f %.3f\n", p[0], p[1], p[2]);

  if (alpha > 1e-4) {
    std::cout << "Alpha was set too high (" << alpha << ") adjust to 1e-5\n";
    alpha = 1e-5;
  }

  Vec3<float> pxy_act(p[0], p[1], 0);
  Vec3<float> pxy_des(world_position_desired[0], world_position_desired[1], 0);
  float pz_err = p[2] - _body_height;
  Vec3<float> vxy(seResult.vWorld[0], seResult.vWorld[1], 0);

  dtMPC = dt * iterationsBetweenMPC;
  setup_problem(dtMPC, horizonLength, 0.4, 120);

  int jcqp_max_iter = 10000;
  double jcqp_rho = 0.0000001;
  double jcqp_sigma = 0.00000001;
  double jcqp_alpha = 1.5;
  double jcqp_terminate = 0.1;
  double use_jcqp = 0.0;

  update_solver_settings(jcqp_max_iter, jcqp_rho, jcqp_sigma, jcqp_alpha,
                         jcqp_terminate, use_jcqp);
  update_problem_data_floats(p, v, q, w, r, yaw, weights, trajAll, alpha,
                             mpcTable);

  for (int leg = 0; leg < 4; leg++) {
    Vec3<float> f;
    for (int axis = 0; axis < 3; axis++)
      f[axis] = get_solution(leg * 3 + axis);
    f_ff[leg] = -seResult.rBody * f;

    Fr_des[leg] = f;
  }
  myflags = myflags + 1;
}

MPCData
ConvexMPCLocomotion::updateMPC(const RobotSystemOutputData &robotSystemData,
                               const UserInput &userInput,
                               const Eigen::MatrixXi &gaitTable) {
  Eigen::Vector3d vBody_des = Eigen::Vector3d::Zero();
  vBody_des[0] = userInput.x_vel_cmd;
  vBody_des[1] = userInput.y_vel_cmd;
  vBody_des = robotSystemData.baseToWorldRotMat * vBody_des;
  MPCData updateMPCData;
  updateMPCData.position = robotSystemData.worldPosition;
  // std::cout<<"updateMPCData.position:
  // "<<updateMPCData.position.transpose()<<std::endl;
  updateMPCData.velocity = robotSystemData.worldLinearVelocity;
  updateMPCData.orientation = robotSystemData.baseRotation;
  updateMPCData.omega = robotSystemData.worldAngularVelocity;
  updateMPCData.alpha = 4e-5;

  std::array<double, 13> initialMPCComponents = {0,
                                                 0,
                                                 userInput.yaw_des,
                                                 userInput.p_des[0],
                                                 userInput.p_des[1],
                                                 userInput.p_des[2],
                                                 0,
                                                 0,
                                                 userInput.yaw_turn_rate,
                                                 vBody_des[0],
                                                 vBody_des[1],
                                                 0,
                                                 0};
  updateMPCData.stateTrajectory = Eigen::MatrixXd(13 * 10, 1);
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 13; j++)
      updateMPCData.stateTrajectory(13 * i + j, 0) = initialMPCComponents[j];
    if (i > 0) {
      updateMPCData.stateTrajectory(13 * i + 2, 0) =
          updateMPCData.stateTrajectory(13 * (i - 1) + 2, 0) +
          0.026 * userInput.yaw_turn_rate;
      updateMPCData.stateTrajectory(13 * i + 3, 0) =
          updateMPCData.stateTrajectory(13 * (i - 1) + 3, 0) +
          0.026 * vBody_des[0];
      updateMPCData.stateTrajectory(13 * i + 4, 0) =
          updateMPCData.stateTrajectory(13 * (i - 1) + 4, 0) +
          0.026 * vBody_des[1];
    }
  }
  // for (int i = 0; i < 10; i++) {
  //   std::cout
  //       << updateMPCData.stateTrajectory.block(i * 13, 0, 13, 1).transpose()
  //       << std::endl;
  // }
  updateMPCData.weights = Eigen::MatrixXd(13, 1);
  updateMPCData.weights << 0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.2, 0.2, 0.1,
      0.0;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      updateMPCData.distanceFromCOM[i][j] =
          pFoot[i][j] - robotSystemData.worldPosition[j];
    }
  }
  // std::cout<<"updateMPCData.stateTrajectory:
  // \n"<<updateMPCData.stateTrajectory<<std::endl;
  updateMPCData.gaitTable = gaitTable;
  updateMPCData.horizon = 10;
  updateMPCData.dtMPC = 0.026;
  return updateMPCData;
}

std::pair<A_dt_t, B_dt_t> ConvexMPCLocomotion::computeAdt_Bdt(
    const Eigen::Quaterniond &baseRotation, const double dtMPC,
    const std::array<Eigen::Vector3d, 4> &footDistFromCOM) {

  Eigen::Matrix<double, 13, 13> A_dt;
  Eigen::Matrix<double, 13, 12> B_dt;
  auto rotMat = baseRotation.toRotationMatrix();

  // TODO: Remove hardcoded mass value
  const double mass = 9.0;

  // TODO: remove hardcoded I_body
  Eigen::Matrix3d I_body;
  I_body << 0.07, 0, 0, 0, 0.26, 0, 0, 0, 0.242;

  Eigen::Matrix3d I_world = rotMat * I_body * rotMat.transpose();
  Eigen::Matrix3d I_world_inv;
  I_world_inv = I_world.inverse();

  A_dt.setZero();
  B_dt.setZero();
  A_dt.setIdentity();
  A_dt.block<3, 3>(0, 6) = rotMat * dtMPC;
  A_dt.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity() * dtMPC;
  A_dt(11, 12) = dtMPC;

  for (int i = 0; i < 4; i++) {
    B_dt.block(6, i * 3, 3, 3) = I_world_inv * skewMat(footDistFromCOM[i]);
    B_dt.block(9, i * 3, 3, 3) = Eigen::Matrix3d::Identity() / mass;
  }

  B_dt *= dtMPC;

  return std::make_pair(A_dt, B_dt);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
ConvexMPCLocomotion::computeAqp_Bqp(const A_dt_t &A_dt, const B_dt_t &B_dt,
                                    const unsigned horizon) {
  std::vector<Eigen::Matrix<double, 13, 13>> powerMats;
  powerMats.resize(horizon + 1);
  powerMats[0].setIdentity();
  for (unsigned i = 1; i < horizon + 1; i++) {
    powerMats[i] = A_dt * powerMats[i - 1];
  }

  Eigen::MatrixXd A_qp(13 * horizon, 13);
  Eigen::MatrixXd B_qp = Eigen::MatrixXd::Zero(13 * horizon, 12 * horizon);

  Eigen::MatrixXd B_qp_bottomRow(13, 12 * horizon);
  for (unsigned i = 0; i < horizon; i++) {
    const unsigned index = horizon - i - 1;
    B_qp_bottomRow.block<13, 12>(0, 12 * i) = powerMats[index] * B_dt;
  }

  for (unsigned r = 0; r < horizon; r++) {
    A_qp.block(13 * r, 0, 13, 13) = powerMats[r + 1];
    const unsigned index = (horizon - r - 1) * 12;
    B_qp.block(13 * r, 0, 13, (r + 1) * 12) =
        B_qp_bottomRow.block(0, index, 13, (r + 1) * 12);
  }

  return std::make_pair(A_qp, B_qp);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ConvexMPCLocomotion::computeqH_qg(
    const Eigen::MatrixXd &A_qp, const Eigen::MatrixXd &B_qp,
    const Eigen::MatrixXd &weights, const Eigen::MatrixXd &x_0,
    const Eigen::MatrixXd &x_d, unsigned horizon) {
  Eigen::MatrixXd q_H = Eigen::MatrixXd::Zero(12 * horizon, 12 * horizon);
  Eigen::MatrixXd q_g = Eigen::MatrixXd::Zero(12 * horizon, 1);
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(13 * horizon, 13 * horizon);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(12 * horizon, 12 * horizon);
  R *= 4e-5;
  S.diagonal() = weights.replicate(horizon, 1);
  q_H = B_qp.transpose() * S * B_qp + R;
  q_g = B_qp.transpose() * S * (A_qp * x_0 - x_d);
  return std::make_pair(q_H, q_g);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
ConvexMPCLocomotion::computeReducedqH_qg(const Eigen::MatrixXd &q_H,
                                         const Eigen::MatrixXd &q_g,
                                         const Eigen::MatrixXi &gait) {
  const int activeLegs = (gait.array() != 0).count();
  Eigen::MatrixXd red_q_H =
      Eigen::MatrixXd::Zero(activeLegs * 3, activeLegs * 3);
  Eigen::MatrixXd red_q_g = Eigen::MatrixXd::Zero(activeLegs * 3, 1);
  std::vector<int> indexList{};
  auto gaitMap = Eigen::Map<const Eigen::VectorXi>(gait.data(), gait.size());
  for (unsigned i = 0; i < gait.size(); i++) {
    if (gaitMap[i] == 1) {
      indexList.emplace_back(3 * i);
      indexList.emplace_back(3 * i + 1);
      indexList.emplace_back(3 * i + 2);
    }
  }

  for (size_t i = 0; i < indexList.size(); i++) {
    auto indexA = indexList[i];
    red_q_g(i, 0) = q_g(indexA, 0);
    for (size_t j = 0; j < indexList.size(); j++) {
      auto indexB = indexList[j];
      red_q_H(i, j) = q_H(indexA, indexB);
    }
  }
  return std::make_pair(red_q_H, red_q_g);
}

std::tuple<RowMatrixXd, Eigen::VectorXd, Eigen::VectorXd>
ConvexMPCLocomotion::generateConstraints(const Eigen::MatrixXi &gaitTable) {
  int activeLegs = (gaitTable.array() != 0).count();

  double mu = 0.2;
  Eigen::Matrix<double, 6, 3> f_block;
  f_block << 1, 0, mu, -1, 0, mu, 0, 1, mu, 0, -1, mu, 0, 0, 1.0, 0, 0, -1.0;
  Eigen::Matrix<double, 6, 1> lb_block =
      (Eigen::Matrix<double, 6, 1>() << 0, 0, 0, 0, 0, -1500.0).finished();
  RowMatrixXd constraints = createBlockDiagonal(f_block, activeLegs);
  Eigen::VectorXd lb = createBlockVertical(lb_block, activeLegs);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(
      activeLegs * 6, std::numeric_limits<double>::max());
  return {constraints, lb, ub};
}

Eigen::Matrix<double, 12, 1>
ConvexMPCLocomotion::solveMPC(const MPCData &input) {
  const int activeLegs = (input.gaitTable.array() != 0).count();
  const unsigned num_variables = input.gaitTable.size() * 3;
  Eigen::MatrixXd x_0 = Eigen::MatrixXd(13, 1);
  Eigen::Vector3d rpy_MPC = quat_to_ypr(input.orientation);
  x_0 << rpy_MPC[2], rpy_MPC[1], rpy_MPC[0], input.position, input.omega,
      input.velocity, -9.8;
  const auto [A_dt, B_dt] =
      computeAdt_Bdt(input.orientation, input.dtMPC, input.distanceFromCOM);
  const auto [A_qp, B_qp] = computeAqp_Bqp(A_dt, B_dt, input.horizon);
  const auto [q_H, q_g] = computeqH_qg(A_qp, B_qp, input.weights, x_0,
                                       input.stateTrajectory, input.horizon);
  const auto [red_q_H, red_q_g] =
      computeReducedqH_qg(q_H, q_g, input.gaitTable);
  const auto [A_mat, lb, ub] = generateConstraints(input.gaitTable);

  qpOASES::QProblem qProblem(activeLegs * 3, num_variables);
  qpOASES::Options options;
  options.printLevel = qpOASES::PL_NONE;
  qProblem.setOptions(options);
  qpOASES::int_t nWSR = 1000;
  qProblem.init(red_q_H.data(), red_q_g.data(), A_mat.data(), nullptr, nullptr,
                lb.data(), ub.data(), nWSR);
  Eigen::VectorXd solved_x = Eigen::VectorXd(activeLegs * 3);
  qProblem.getPrimalSolution(solved_x.data());

  Eigen::Matrix<double, 12, 1> solution_Fr_des;
  int count = 0;
  for (int i = 0; i < 4; i++) {
    if (input.gaitTable(0, i)) {
      solution_Fr_des.block<3, 1>(i * 3, 0) = solved_x.segment(count * 3, 3);
      count++;
    } else {
      solution_Fr_des.block<3, 1>(i * 3, 0) = Eigen::Vector3d::Zero();
    }
  }
  return solution_Fr_des;
}
