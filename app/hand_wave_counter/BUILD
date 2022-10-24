# Copyright 2022 Alexis Kanter
# 
# My solution for Magic CV/ML Engineer Interview Task

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "hand_wave_counter",
    data = [
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:hand_wave_counter_main",
        "//mediapipe/graphs/hand_wave_counter:hand_wave_counter_deps",
    ],
)
