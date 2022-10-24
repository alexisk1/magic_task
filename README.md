# Magic Interview Task Solution

This is the Readme file for Alexi's solution to the Magic Interview Task.

## Introduction

The interview's main task, according to my understanding, is to use a pose estimation CNN model and the mediapipe library to perform hand wave counting on a video stream. 

In order to get the wave counter application working, we need to break down the functionality of the application into the following steps:
- (a) Pose Estimation
  There are a number of different and well optimized CNN's that do regression of body landmarks. Mediapipe uses BladePose with three different variants the heavy, full and the lite. The comparison of the three models seems to show that for mobile applications the full model has the best trade-off between accuracy and performance. We could change this to the lite version. 
- (b) Calculate the Elbow angle
  The elbow angle is defined by two vectors which are defined by three landmarks of interests named as right elbow, right wirst, right shoulder. The first vector is defined by the Elbow to Wrist 3D points / landmarks and the second one is defined by Elbow to Shoulder 3D points / landmarks. To calcuate the angle we can use the dot product equation to get the cosine value of the theta angle defined between the two aforementioned 3D vectors.
- (c) Detect the full wave period using the angle
  Now, we need to somehow to track the "position" of the hand using the elbow angle and detect any periodical "waving". To do that, I defined two different solutions explained in the following sections (Solution A && Solution B).
  
  
## Mediapipe - Main Application and graph

I used a C++ module to proceed with my solution. I have already used mediapipe with Android SDK but since I don't have a recent installation and setup of Android SDK, I tried to define first the C++ desktop application. I manage to solve the task using a simple running mean approach (Hand Wave Counter) but I also tried to make the algorithm more robust using a Kalman Filter(Hand Wave Counter KF). Therefore, I did the following:
- I defined a new bazel Build file for each solution (hand_wave_counter, hand_wave_counter_kf) in mediapipe/examples/desktop/ to conform with the in-built examples for C++.
- I defined a new main C++ file to remove the use of the camera feed in case of any input video file issues and to log an appropriate Log/Console message in case of inappropriate use, setting logging to stderr console to INFO so all Logging messages are passed to the console too. Therefore, I updated the 
- I defined a simple graph for each solution with no subgraphs used. Initially, I tried to define the new calculators/components as modules and use them as subgraphs but that seemed a bit more complicated than just to define a new calculator and I didn't have the required time to proceed.
- The main components of the graphs are:
-- FlowLimiterCalculator to limit the input frame rate to match the computation rate of the downstream
-- PoseLandmarkCpu Module as well as a ConstantSidePacketCalculator to use the Pose Landmark on CPU module defined already in the Mediapipe library without the calculating the segmentation mask.
-- Create two new Calculators namely pose_landmarks_to_wave_counter_calculator for the first solution and pose_landmarks_to_wave_counter_kf_calculator for the second solution that take as input the Normalized Landmarks and outputs the as render data and console information the "hand wave counter" and the "hand wave progress" variables. I added the calculators in the mediapipe/calculators/util/ folder update the Build file too. I also included a new protobuf file for the Options of the calculators named as 

Unfortunately, I wanted to incorporate the graph in an Android application but I didn't have time to proceed. I have done this in the past and I knew it would take valuable time away from my preparation.

## Options for calculators 

We define the common options data structure in the protobuf message pose_landmarks_to_wave_counter_options_calculator.proto , these are the following shared options:

There are two labels(Text) added on the video, the hand wave counter and the hand wave progress. The Label font options are:
- color: The text color to be used.
- thickness: The thickness of the letters
- font_height_px: The height of the letters
- horizontal_offset_px: The offset in pixels from the left border.
- vertical_offset_px: The offset in pixels from the top border.

Running Mean(Solution A) options:
- pre_settled_time: Number of frames to wait the average angle measurement to settle down.
- min_half_cycle: Minimum number of frames for the elbow angle to be either in the positive or in the negative side.
- min_classify_cycles: Minimum number of frames to wait before starting classifing the half period as done. This should be at least Min() = pre_settled_time + 2 * min_half_cycle

## Solution A - Hand Wave Counter - Running mean

Here, the solutions calculates on the fly the average elbow angle. Then, we can count the frames that the elbow is in a negative position and the times it is in a positive position relative to the mean angle. This will help us detect and average a half period of a hand wave period. Counting the times this loop has been performed and dividing by 2 will give us the required hand wave counter.
The main drawbacks of this technique is that it has two perform half a period in order to actually start settling down. Therefore, the counting is not that robust for the first and for the last period. In addition, the whole process requires parameterization and it is not based on a probalistic model for tracking the angle and angular velocity (momentum). 
Therefore, a better approach is to use a Kalman Filter or an Extended Kalman Filter to measure and correct the Elbow angle and angular velocity using the timestamp from the mediapipe graph.

Then, we accummulate the delta(angle) to estimate the angular distance covered in degrees of a half cycle. This want is done by just using the maximum value it is time. Therefore, we can estimate for each half period the progress of the waving functionality. In the end, the total progress of the waving function is 0% - 50% if we have counted even number of periods or 50% to 100% in case we have an odd number of total periods. A btter approach is two average all the counted half length.

[Video of solution A on the two input videos](https://youtu.be/PUqMW7HJ7uQ)

### Building & Running 
After copying the necessary folders and files into the existing folder structure of mediapipe.  

Build:

  bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1  mediapipe/examples/desktop/hand_wave_counter_kf:hand_wave_counter_kf

Run Example:

  bazel-bin/mediapipe/examples/desktop/hand_wave_counter/hand_wave_counter  --calculator_graph_config_file=mediapipe/graphs/hand_wave_counter/hand_wave_counter.pbtxt --input_video_path=/home/alex/Downloads/B.mp4 --output_video_path=./B_out.mp4


## Solution A  - Hand Wave Counter KF - Kalman Filter

Here, we try to pass the angle and the angular velocity (momentum) through a Kalman Filter. Using a better angle and angular velocity we can define better when the hand reaches the initial position with the initial velocity direction (not value, we only care about the direction).


### Building & Running 
After copying the necessary folders and files into the existing folder structure of mediapipe.  

Build:

  bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1  mediapipe/examples/desktop/hand_wave_counter_kf:hand_wave_counter_kf

Run:

  bazel-bin/mediapipe/examples/desktop/hand_wave_counter/hand_wave_counter_kf  --calculator_graph_config_file=mediapipe/graphs/hand_wave_counter_kf/hand_wave_counter_kf.pbtxt --input_video_path=/home/alex/Downloads/A.mp4 --output_video_path=./A_out.mp4


