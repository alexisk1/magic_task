# Copyright 2022 Alexis Kanter
#
# My 2nd solution for Magic Interview Task

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hand_wave_counter_kf_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
        "//mediapipe/calculators/util:pose_landmarks_to_wave_counter_kf_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
    ],
)

mediapipe_binary_graph(
    name = "hand_wave_counter_kf_binary_graph",
    graph = "hand_wave_counter_kf.pbtxt",
    output_name = "hand_wave_counter_kf.binarypb",
    deps = [":hand_wave_counter_kf_deps"],
)
