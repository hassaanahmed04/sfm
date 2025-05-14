#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <crow_all.h>
#include <Eigen/Dense>
#include <memory>

using namespace Eigen;

struct ReprojectionError {
  ReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
      : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {
    const T* R = camera;
    const T* t = camera + 9;

    T p[3] = { point[0] - t[0], point[1] - t[1], point[2] - t[2] };

    T world_point[3] = {
      R[0] * p[0] + R[1] * p[1] + R[2] * p[2],
      R[3] * p[0] + R[4] * p[1] + R[5] * p[2],
      R[6] * p[0] + R[7] * p[1] + R[8] * p[2]
    };

    T inv_z = T(1.0) / ceres::fmax(T(1e-10), world_point[2]);
    T u = fx * world_point[0] * inv_z + cx;
    T v = fy * world_point[1] * inv_z + cy;

    residuals[0] = u - T(observed_x);
    residuals[1] = v - T(observed_y);

    return true;
  }

  static ceres::CostFunction* Create(double observed_x, double observed_y, double fx, double fy, double cx, double cy) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 12, 3>(
      new ReprojectionError(observed_x, observed_y, fx, fy, cx, cy));
  }

  const double observed_x, observed_y;
  const double fx, fy, cx, cy;
};

struct BundleAdjustmentData {
  std::vector<double> observations;
  std::vector<double> camera_params;
  std::vector<double> points_3d;
  double fx, fy, cx, cy;
  size_t num_points;
  size_t num_observations;

  bool validate() const {
    return (points_3d.size() % 3 == 0) && 
           (observations.size() % 2 == 0) &&
           (camera_params.size() == 12) &&
           (num_points == points_3d.size() / 3) &&
           (num_observations == observations.size() / 2);
  }
};

BundleAdjustmentData parse_bundle_adjust_request(const crow::request& req) {
  auto data = crow::json::load(req.body);
  if (!data) throw std::runtime_error("Invalid JSON input");

  BundleAdjustmentData result;

  auto obs = data["observations"];
  if (!obs) throw std::runtime_error("Missing observations");
  for (auto val : obs) result.observations.push_back(val.d());
  result.num_observations = result.observations.size() / 2;

  auto cam_params = data["camera_params"];
  if (!cam_params || cam_params.size() != 12) throw std::runtime_error("Camera parameters must have exactly 12 elements");
  for (auto val : cam_params) result.camera_params.push_back(val.d());

  auto pts_3d = data["points_3d"];
  if (!pts_3d || pts_3d.size() % 3 != 0) throw std::runtime_error("3D points must be a multiple of 3");
  for (auto val : pts_3d) result.points_3d.push_back(val.d());
  result.num_points = result.points_3d.size() / 3;

  auto intrinsics = data["intrinsics"];
  if (!intrinsics || intrinsics.size() != 4) throw std::runtime_error("Intrinsics must have exactly 4 elements");
  result.fx = intrinsics[0].d();
  result.fy = intrinsics[1].d();
  result.cx = intrinsics[2].d();
  result.cy = intrinsics[3].d();

  if (!result.validate()) throw std::runtime_error("Invalid data dimensions");
  return result;
}

int main() {
  crow::SimpleApp app;

  CROW_ROUTE(app, "/bundle_adjust").methods("POST"_method)([](const crow::request& req) {
    try {
      auto data = parse_bundle_adjust_request(req);
      ceres::Problem problem;
      double* camera_params = data.camera_params.data();
      double* points_3d = data.points_3d.data();

      for (size_t i = 0; i < data.num_observations; i++) {
        size_t pt_idx = i % data.num_points;
        auto cost_fn = ReprojectionError::Create(
          data.observations[2 * i],
          data.observations[2 * i + 1],
          data.fx, data.fy, data.cx, data.cy);

        problem.AddResidualBlock(cost_fn, new ceres::CauchyLoss(1.0), camera_params, points_3d + 3 * pt_idx);
      }

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.minimizer_progress_to_stdout = false;
      options.max_num_iterations = 200;
      options.function_tolerance = 1e-8;
      options.gradient_tolerance = 1e-10;
      options.parameter_tolerance = 1e-12;
      options.num_threads = std::thread::hardware_concurrency();

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      
      crow::json::wvalue response;
      response["optimized_camera"] = std::vector<double>(camera_params, camera_params + 12);
      response["optimized_points"] = data.points_3d;
      response["summary"] = {
        {"brief_report", summary.BriefReport()},
        {"total_time", summary.total_time_in_seconds},
        {"initial_cost", summary.initial_cost},
        {"final_cost", summary.final_cost},
        {"iterations", summary.iterations.size()}
      };

      return crow::response(response);
    } catch (const std::exception& e) {
      crow::json::wvalue err_json({{"error", e.what()}});
      return crow::response(400, err_json);
    }
  });

  CROW_ROUTE(app, "/health")([]() {
    return crow::response(200, "OK");
  });

  std::cout << "Starting Ceres optimization server on port 8080..." << std::endl;
  app.port(8080).multithreaded().run();
  return 0;
}
