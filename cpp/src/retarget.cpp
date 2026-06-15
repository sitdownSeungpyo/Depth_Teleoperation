#include "itu/retarget.hpp"

#include "itu/mathutil.hpp"

#include <Eigen/Geometry>  // Vector3d::cross
#include <cmath>

namespace itu {

namespace {
constexpr double kEps = 1e-6;

Eigen::Vector3d project_perpendicular(const Eigen::Vector3d& v, const Eigen::Vector3d& axis) {
  return v - v.dot(axis) * axis;  // axis assumed unit length
}

double sign(double x) { return (x > 0.0) - (x < 0.0); }

const Eigen::Vector3d kWorldDown(0.0, -1.0, 0.0);

// Look up a keypoint; nullopt if absent.
std::optional<Eigen::Vector3d> get_kp(const KeypointMap& kp, const std::string& name) {
  auto it = kp.find(name);
  if (it == kp.end()) return std::nullopt;
  return it->second;
}

double estimate_shoulder_yaw(const Eigen::Vector3d& shoulder, const Eigen::Vector3d& elbow,
                             const Eigen::Vector3d& wrist, Side side) {
  auto upper = mathutil::safe_normalize(elbow - shoulder);
  if (!upper) return 0.0;
  auto forearm = mathutil::safe_normalize(wrist - elbow);
  if (!forearm) return 0.0;

  auto elbow_axis = mathutil::safe_normalize(upper->cross(*forearm), 0.1);
  if (!elbow_axis) return 0.0;  // nearly straight arm — yaw not observable

  auto ref_axis = mathutil::safe_normalize(upper->cross(kWorldDown), 0.1);
  if (!ref_axis) return 0.0;  // upper arm parallel to gravity

  const double sin_yaw = ref_axis->cross(*elbow_axis).dot(*upper);
  const double cos_yaw = ref_axis->dot(*elbow_axis);
  const double yaw = std::atan2(sin_yaw, cos_yaw);
  return side == Side::Right ? yaw : -yaw;
}

std::pair<double, double> estimate_wrist_yaw_pitch(const Eigen::Vector3d& elbow,
                                                   const Eigen::Vector3d& wrist,
                                                   const Eigen::Vector3d& hand_middle,
                                                   const Eigen::Vector3d& hand_index,
                                                   const Eigen::Vector3d& hand_pinky, Side side) {
  auto forearm = mathutil::safe_normalize(wrist - elbow);
  if (!forearm) return {0.0, 0.0};
  auto hand_dir = mathutil::safe_normalize(hand_middle - wrist, 1e-4);
  if (!hand_dir) return {0.0, 0.0};
  auto palm_across = mathutil::safe_normalize(hand_index - hand_pinky, 1e-4);
  if (!palm_across) return {0.0, 0.0};

  const double pitch_mag = mathutil::safe_arccos(forearm->dot(*hand_dir));
  const Eigen::Vector3d pitch_axis = forearm->cross(*hand_dir);
  double s = sign(pitch_axis.dot(*palm_across));
  if (s == 0.0) s = 1.0;
  const double wrist_pitch = (side == Side::Right ? s : -s) * pitch_mag;

  auto pap = mathutil::safe_normalize(project_perpendicular(*palm_across, *forearm), 1e-4);
  if (!pap) return {0.0, wrist_pitch};
  auto ref = mathutil::safe_normalize(forearm->cross(kWorldDown), 0.1);
  if (!ref) return {0.0, wrist_pitch};
  const double sin_yaw = ref->cross(*pap).dot(*forearm);
  const double cos_yaw = ref->dot(*pap);
  const double yaw = std::atan2(sin_yaw, cos_yaw);
  const double wrist_yaw = side == Side::Right ? yaw : -yaw;
  return {wrist_yaw, wrist_pitch};
}
}  // namespace

Calibration Calibration::from_tpose_frames(const std::vector<AlignedFrame>& frames) {
  std::vector<double> lengths;
  for (const auto& frame : frames) {
    for (Side side : {Side::Left, Side::Right}) {
      const std::string s = side_name(side);
      const Eigen::Vector3d shoulder = frame.keypoints.at(s + "_shoulder");
      const Eigen::Vector3d wrist = frame.keypoints.at(s + "_wrist");
      lengths.push_back((wrist - shoulder).norm());
    }
  }
  if (lengths.empty()) throw SingularConfigurationError("calibration received no frames");
  Calibration c;
  c.operator_arm_length = mathutil::mean(lengths);
  return c;
}

JointMap retarget_arm(const AlignedFrame& aligned, Side side, const RobotGeometry& robot,
                      const Calibration& calibration, bool decouple_pitch_elbow) {
  const std::string s = side_name(side);
  const Eigen::Vector3d shoulder = aligned.keypoints.at(s + "_shoulder");
  const Eigen::Vector3d elbow = aligned.keypoints.at(s + "_elbow");
  const Eigen::Vector3d wrist = aligned.keypoints.at(s + "_wrist");

  if (elbow.norm() < kEps || wrist.norm() < kEps)
    throw SingularConfigurationError(s + " elbow/wrist keypoint missing (zero vector)");

  const Eigen::Vector3d upper = elbow - shoulder;
  const Eigen::Vector3d lower = wrist - elbow;
  const Eigen::Vector3d s_to_w = wrist - shoulder;
  const double upper_n = upper.norm();
  const double lower_n = lower.norm();
  const double c = s_to_w.norm();
  if (upper_n < kEps || lower_n < kEps || c < kEps)
    throw SingularConfigurationError("degenerate " + s + " arm geometry");

  // scale is exposed in Python for future Cartesian targets; angles are
  // scale-invariant, so it is computed-and-discarded there. Omitted here.

  // Eq. 1 — elbow flexion (angle between upper-arm and forearm).
  const double cos_elbow = upper.dot(lower) / (upper_n * lower_n);
  const double theta_5 = mathutil::safe_arccos(cos_elbow);

  // Eq. 2 — shoulder elevation. Aligner is +y up, paper is +y down -> negate.
  const double y_paper = -s_to_w[1];
  const double theta_3 = mathutil::safe_arccos(y_paper / c);

  // Eq. 3 — shoulder pitch. Mirror x for the left arm. +0.0 canonicalises -0.0.
  const double x = (side == Side::Left ? -s_to_w[0] : s_to_w[0]) + 0.0;
  const double z = s_to_w[2] + 0.0;
  const double azimuth = std::atan2(z, x);
  const double theta_1 = decouple_pitch_elbow ? azimuth : (azimuth - theta_5);

  const std::string prefix = side_prefix(side);
  const auto& offsets = calibration.rest_offsets;
  auto out_val = [&](const std::string& name, double raw) {
    double off = 0.0;
    if (auto it = offsets.find(name); it != offsets.end()) off = it->second;
    return mathutil::wrap_pi(raw - off);
  };

  JointMap out;
  out[prefix + "_shoulder_pitch"] = out_val(prefix + "_shoulder_pitch", theta_1);
  out[prefix + "_shoulder_roll"] = out_val(prefix + "_shoulder_roll", theta_3);
  out[prefix + "_elbow"] = out_val(prefix + "_elbow", theta_5);

  const double sh_yaw = estimate_shoulder_yaw(shoulder, elbow, wrist, side);
  out[prefix + "_shoulder_yaw"] = out_val(prefix + "_shoulder_yaw", sh_yaw);

  auto hm = get_kp(aligned.keypoints, s + "_hand_middle_mcp");
  auto hi = get_kp(aligned.keypoints, s + "_hand_index_mcp");
  auto hp = get_kp(aligned.keypoints, s + "_hand_pinky_mcp");
  if (hm && hi && hp && (*hm - wrist).norm() > 1e-4) {
    auto [w_yaw, w_pitch] = estimate_wrist_yaw_pitch(elbow, wrist, *hm, *hi, *hp, side);
    out[prefix + "_wrist_yaw"] = out_val(prefix + "_wrist_yaw", w_yaw);
    out[prefix + "_wrist_pitch"] = out_val(prefix + "_wrist_pitch", w_pitch);
  }
  return out;
}

JointMap retarget_full_upper_body(const AlignedFrame& aligned, const RobotGeometry& robot,
                                  const Calibration& calibration, bool decouple_pitch_elbow) {
  JointMap out;
  for (Side side : {Side::Right, Side::Left}) {
    try {
      JointMap arm = retarget_arm(aligned, side, robot, calibration, decouple_pitch_elbow);
      for (auto& [k, v] : arm) out[k] = v;
    } catch (const SingularConfigurationError&) {
      // skip the failing arm; downstream filter holds its previous value.
    }
  }
  out["torso_yaw"] = aligned.rpy[2];
  out["head_pitch"] = aligned.rpy[1];
  // Passthrough defaults for joints the IK doesn't compute.
  for (const char* j : {"r_shoulder_yaw", "l_shoulder_yaw", "r_wrist_yaw", "l_wrist_yaw",
                        "r_wrist_pitch", "l_wrist_pitch", "neck_yaw"}) {
    out.emplace(j, 0.0);  // setdefault: only inserts if absent
  }
  return out;
}

double estimate_arm_length(const KeypointMap& keypoints) {
  std::vector<double> lengths;
  for (Side side : {Side::Left, Side::Right}) {
    const std::string s = side_name(side);
    const Eigen::Vector3d shoulder = keypoints.at(s + "_shoulder");
    const Eigen::Vector3d wrist = keypoints.at(s + "_wrist");
    lengths.push_back((wrist - shoulder).norm());
  }
  return mathutil::mean(lengths);
}

void CalibrationCollector::push(const AlignedFrame& frame) {
  buf_.push_back(frame);
  while (static_cast<int>(buf_.size()) > target_) buf_.pop_front();
}

Calibration CalibrationCollector::finalise(const std::optional<RobotGeometry>& robot,
                                           bool decouple_pitch_elbow) {
  if (!ready())
    throw SingularConfigurationError("need " + std::to_string(target_) + " frames, have " +
                                     std::to_string(buf_.size()));
  std::vector<AlignedFrame> frames(buf_.begin(), buf_.end());
  Calibration cal = Calibration::from_tpose_frames(frames);
  if (!robot) return cal;

  Calibration temp;
  temp.operator_arm_length = cal.operator_arm_length;
  std::map<std::string, std::vector<double>> accum;
  for (const auto& frame : frames) {
    JointMap angles;
    try {
      angles = retarget_full_upper_body(frame, *robot, temp, decouple_pitch_elbow);
    } catch (const SingularConfigurationError&) {
      continue;
    }
    for (const auto& [joint, value] : angles) accum[joint].push_back(value);
  }
  for (const auto& [joint, values] : accum) {
    if (!values.empty()) cal.rest_offsets[joint] = mathutil::circular_mean(values);
  }
  return cal;
}

}  // namespace itu
