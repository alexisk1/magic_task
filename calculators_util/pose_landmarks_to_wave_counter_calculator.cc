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

namespace mediapipe {

namespace {

constexpr char kSceneLabelAngle[] = "Angle";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kRenderDataTag[] = "RENDER_DATA";
constexpr int kNumLandmarks = 33;
constexpr double kHalfCircleDegrees = 180.0;
constexpr double kFullCircleDegrees = 360.0;
constexpr int kNumPreviousAngles = 5;
constexpr double kNextTextLineScale = 1.2;

}  // namespace

class PoseLandmarksToWaveCounterCalculator : public CalculatorBase {

 private:
  // Calculator options set by the graph config and defined in 
  PoseLandmarksToWaveCounterCalculatorOptions options_;

  // Running mean - hand wave counter vars
  double running_right_mean_sum_ = 0.0;
  int running_right_samples_ = 0;
  int right_positive_ = 0;
  int right_negative_ = 0;
  int right_middle_ = 0;
  double init_angular_distance_ = 0.0;
  double angular_distance_ = 0.0;

  // Angles history
  std::list<double> prev_right_angles_;

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
    // Initialize Timestamp
    cc->SetOffset(TimestampDiff(0));

    // Get the Graph options
    options_ = cc->Options<PoseLandmarksToWaveCounterCalculatorOptions>();

    // Initialize running mean hand wave counter
    running_right_mean_sum_ = 0;
    running_right_samples_ = 0;
    right_positive_ = 0;
    right_negative_ = 0;
    right_middle_ = 0;
  
    // Initialize angle history container
    while (!prev_right_angles_.empty())
     prev_right_angles_.pop_front();

    // Initialize log helping var
    show_estimation_msg_ = true;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Get the norm landmarks are not empty
    if (cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
      return absl::OkStatus();
    }
    auto tm = cc->InputTimestamp();
    // Get the right hand landmarks
    const std::vector<int> kRightHandLandmarks {12, 14, 16};
    const auto right_landmarks = GetPartialLandmarks(cc, kRightHandLandmarks);

    // Calculate the right hand "elbow" angle
    auto right_angle = CalculateVectorAngle(right_landmarks);

    // Update list with previous angles for percentage calculation
    prev_right_angles_.push_front(right_angle);
    if(prev_right_angles_.size() > kNumPreviousAngles)
      prev_right_angles_.pop_back();

    // Update Angular distance
    auto angle_last = prev_right_angles_.front();
    auto angle_prev = (++prev_right_angles_.front());
    angular_distance_ += std::abs(angle_last - angle_prev);
 
    // Update running mean of angle
    running_right_mean_sum_ += right_angle;
    running_right_samples_ ++;
    auto right_mean = running_right_mean_sum_ / running_right_samples_;

    // Update position counters for hand based on the new mean
    if (running_right_samples_ > options_.pre_settled_time()){
      if (right_angle > right_mean) 
        right_positive_++;
      else 
        right_negative_++;
    }

    // Calculate percentage of gesture completion by using the angular 
    double percentage = 0.0;    
    if (right_middle_) {
      // Limit percentage up to 100 %
      percentage = std::min(angular_distance_ / init_angular_distance_, 1.0);

      // We estimate the half period distance
      percentage /= 2;

      // If counter is odd then we are in the second half period
      if (right_middle_ % 2 == 1)
        percentage += 0.5;
    } else if(show_estimation_msg_) {
      LOG(INFO) << "First Hand Wave cycle -- waiting for first repetition to finish... "; 
      show_estimation_msg_ = false;
    }



    // Check if a full wave cycle has been performed    
    if(right_positive_ > options_.min_half_cycle() && 
      right_negative_ > options_.min_half_cycle() &&
      running_right_samples_ > options_.min_classify_samples())
    {
      // Update angular distance for the first time or when a longer 
      // distance is covered
      if (!right_middle_ || angular_distance_ > init_angular_distance_){
        init_angular_distance_ = angular_distance_;
      }
      // Update half period counter
      right_middle_++;

      // Initialize all counters
      right_positive_ = 0;
      right_negative_ = 0;
      angular_distance_ = 0.0;
    }

    LOG(INFO) << "Hand Wave Counter: " << int(right_middle_ / 2) << "  Hand Wave progress: " << percentage * 100 << " %"; 

    // Output hand  wave counter and percentage of completion so
    // that they are rendered in the video by the annotation
    // overlay component.
    auto render_data = absl::make_unique<RenderData>();
    AddRenderData(render_data.get(), int(right_middle_ / 2), double(percentage * 100));
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

};

REGISTER_CALCULATOR(PoseLandmarksToWaveCounterCalculator);

}  // namespace mediapipe
