// Copyright 2022 Alexis Kanter
//
// This is a simple version of a hand wave counter using running mean.
// When the hand goes back and forth the mean we can count this as 1 
// hand wave movement. Obviously, this may not be the best solution
// in terms of accuracy but it is the simplest and fastest solution
// to the task with the given dataset. Essentially, the solution counts
// the times the hand passes through the averaged middle angle.


#include <cmath>
#include <string.h>
#include <list>
#include <iterator>

#include "mediapipe/calculators/util/pose_landmarks_to_wave_counter_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/render_data.pb.h"

#include <opencv2/core/matx.hpp>
#include <opencv2/video.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cvdef.h"

namespace mediapipe {

namespace {

constexpr char kSceneLabelAngle[] = "Angle";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kRenderDataTag[] = "RENDER_DATA";
constexpr int kNumLandmarks = 33;
constexpr double kHalfCircleDegrees = 180.0;
constexpr double kFullCircleDegrees = 360.0;
constexpr int kNumPreviousAngles = 5;
constexpr double kNextTextLineScale = 1.2;

struct Measurement {
  Timestamp timestamp = Timestamp::Unset();
  double angle = 0.0;
  double momentum = 0.0;
};

}  // namespace

class PoseLandmarksToWaveCounterKfCalculator : public CalculatorBase {

 private:
  PoseLandmarksToWaveCounterCalculatorOptions options_;
  int total_wave_count_ = 0;

  Measurement initial_measurement_;
  bool set_initial_ = true;


  std::list<Measurement> prev_measurements_;
  double angular_distance = 0.0;

  // EKF
  cv::KalmanFilter KF;
  cv::Mat state; /* (phi, delta_phi) */
  cv::Mat processNoise;
  cv::Mat measurement;

  // Logging var
  bool show_estimation_msg_ = true;

 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // Validate input and Output types
    cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    // Get the Graph options
    options_ = cc->Options<PoseLandmarksToWaveCounterCalculatorOptions>();

    while (!prev_measurements_.empty())
     prev_measurements_.pop_front();

    //KF
    KF.init(2, 1, 0, CV_64F);
    state = cv::Mat(2, 1, CV_64F); /* (phi, delta_phi) */
    processNoise = cv::Mat(2, 1, CV_64F);
    measurement = cv::Mat::zeros(1, 1, CV_64F);
    state.at<double>(0) = 0.0f;
    state.at<double>(1) = 0.0f;
    KF.transitionMatrix = (cv::Mat_<double>(2, 2) << 1, 1, 0, 1);
    cv::setIdentity(KF.measurementMatrix);
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));

    // Logging variable
    show_estimation_msg_ = true;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Get the norm landmarks are not empty
    if (cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
      return absl::OkStatus();
    }

    // Get the right hand landmarks
    const std::vector<int> kRightHandLandmarks {12, 14, 16};
    const auto right_landmarks = GetPartialLandmarks(cc, kRightHandLandmarks);

    // Calculate the right hand "elbow" angle
    auto right_angle = CalculateVectorAngle(right_landmarks);


    // Update history of measurements
    UpdateMeasurement(&prev_measurements_, right_angle, cc->InputTimestamp());


    // EKF
    double stateAngle = state.at<double>(0);
    cv::Mat prediction = KF.predict();
    double predictAngle = prediction.at<double>(0);
    double predictAngular = prediction.at<double>(1);

    measurement.at<double>(0) = prev_measurements_.begin()->angle;
    measurement.at<double>(1) = angular_momentum;
    measurement += KF.measurementMatrix*state;
    double measAngle = measurement.at<double>(0);
    // correct the state estimates based on measurements
    // updates statePost & errorCovPost
    KF.correct(measurement);
    state = KF.transitionMatrix*state + processNoise;

    double improvedAngle = KF.statePost.at<double>(0);
    double improvedAngular = KF.statePost.at<double>(1);

    std::cout<<"predAngle  "<<  predictAngle <<" state " << improvedAngle << "  meas: "<< right_angle <<std::endl ;
    std::cout<<"predAMomentum "<< predictAngular <<" state " <<improvedAngular << "  Amomentum: "<< angular_momentum <<std::endl ;

  
    
    // Output hand 
    auto render_data = absl::make_unique<RenderData>();
    AddRenderData(render_data.get(), int(right_middle_ / 2), double(0.0 * 100));

    cc->Outputs()
        .Tag(kRenderDataTag)
        .Add(render_data.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

 private:
  NormalizedLandmarkList GetPartialLandmarks(CalculatorContext* cc, const std::vector<int> kPartialLandmarks) {
    // Get the landmarks of interest
    const auto& landmarks =
        cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();
    NormalizedLandmarkList partial_landmarks;
    for (int i : kPartialLandmarks) {
      *partial_landmarks.add_landmark() = landmarks.landmark(i);
    }
    return partial_landmarks;
  }


  
  void AddRenderData(RenderData* render_data, double right_angle, double percentage) {
    // Add Text for hand wave counter in the video
    // Set up label
    auto* counter_label_annotation = render_data->add_render_annotations();
    counter_label_annotation->set_scene_tag(kSceneLabelAngle);
    counter_label_annotation->mutable_color()->set_r(options_.color().r());
    counter_label_annotation->mutable_color()->set_g(options_.color().g());
    counter_label_annotation->mutable_color()->set_b(options_.color().b());
    counter_label_annotation->set_thickness(options_.thickness());

    // Update Text and format it based on the options provided in the graph config pbtxt
    auto* counter_text = counter_label_annotation->mutable_text();
    counter_text->set_display_text("Wave Counter: " + std::to_string(right_angle));
    counter_text->set_normalized(false);
    counter_text->set_left(options_.horizontal_offset_px());
    counter_text->set_baseline(options_.vertical_offset_px());
    counter_text->set_font_height(options_.font_height_px());
    counter_text->set_font_face(options_.font_face());

    // Add Text for percentage in the video
    // Set up label
    auto* perc_label_annotation = render_data->add_render_annotations();
    perc_label_annotation->set_scene_tag(kSceneLabelAngle);
    perc_label_annotation->mutable_color()->set_r(options_.color().r());
    perc_label_annotation->mutable_color()->set_g(options_.color().g());
    perc_label_annotation->mutable_color()->set_b(options_.color().b());
    perc_label_annotation->set_thickness(options_.thickness());

    // Update Text and format it based on the options provided in the graph config pbtxt
    auto* perc_text = perc_label_annotation->mutable_text();
    perc_text->set_display_text("Wave Cycle Completion: " + std::to_string(percentage) + "%");
    perc_text->set_normalized(false);
    perc_text->set_left(options_.horizontal_offset_px());
    perc_text->set_baseline(options_.vertical_offset_px() + options_.font_height_px() * kNextTextLineScale);
    perc_text->set_font_height(options_.font_height_px());
    perc_text->set_font_face(options_.font_face());
  }

  double CalculateVectorAngle(const NormalizedLandmarkList& landmarks) {
    // Get the 3 points of interest
    auto shoulder = landmarks.landmark(0);
    auto elbow = landmarks.landmark(1);
    auto wrist = landmarks.landmark(2);

    // Create the two 3d vectors, using the elbow as the initial point 
    // and the shoulder as well as the elbow as the end points of the two vectors
    cv::Vec3d vector_a {shoulder.x() - elbow.x(),
                        shoulder.y() - elbow.y(),
                        shoulder.z() - elbow.z()};
    cv::Vec3d vector_b {wrist.x() - elbow.x(),
                        wrist.y() - elbow.y(),
                        wrist.z() - elbow.z()};

    // Calculate the angle between the 3D vectors
    //  a . b  = |a| |b| cos(theta)            .       => dot product and theta is the angle between the two vectors
    //  theta = cos^-1 (a . b / (|a| |b|))     cos^-1  => arc_cos(x)
    double angle = std::acos(vector_a.dot(vector_b) / (cv::norm(vector_a) * cv::norm(vector_b)));

    // Convert radians to degrees and limit them between 0 - 180 
    angle *= kHalfCircleDegrees / CV_PI;
    if (angle > kHalfCircleDegrees)
       angle = kFullCircleDegrees - angle;

    return angle;
  }

  void UpdateMeasurement(std::list<Measurement>* history, double angle, Timestamp timestamp) {
    Measurement new_measurement;
    new_measurement.timestamp = timestamp;
    new_measurement.angle = angle;
    double momentum = 0.0;

    // Calculate Angular Momentum
    auto last_measurement = history->begin();
    if (history->size() <= 1)
      momentum = 0.0;
    else 
      momentum = (angle - last_measurement->angle) / double((timestamp - last_measurement->timestamp).Microseconds());
    
    // Push new measurement to history container
    history->push_front(new_measurement);
    while(history->size() == options_.pre_settled_time() + 1)
      history->pop_back();
  }

};

REGISTER_CALCULATOR(PoseLandmarksToWaveCounterCalculator);

}  // namespace mediapipe
