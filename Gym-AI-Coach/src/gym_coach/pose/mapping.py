"""Versioned subset of MediaPipe PoseLandmark; no backend import on module load."""

from enum import StrEnum
from types import MappingProxyType

from ..contracts import Exercise

JOINT_MAPPING_VERSION = "mediapipe-pose-13-v1"


class JointName(StrEnum):
    NOSE = "nose"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


JOINT_INDEX = MappingProxyType(dict(zip(JointName, (0, 11, 12, 13, 14, 15, 16,
                                                    23, 24, 25, 26, 27, 28), strict=True)))
REQUIRED_JOINTS = MappingProxyType({
    Exercise.BODY_WEIGHT_SQUATS: (
        JointName.LEFT_SHOULDER, JointName.RIGHT_SHOULDER,
        JointName.LEFT_HIP, JointName.RIGHT_HIP, JointName.LEFT_KNEE,
        JointName.RIGHT_KNEE, JointName.LEFT_ANKLE, JointName.RIGHT_ANKLE,
    ),
    Exercise.HANDSTAND_PUSHUPS: (
        JointName.LEFT_SHOULDER, JointName.RIGHT_SHOULDER,
        JointName.LEFT_ELBOW, JointName.RIGHT_ELBOW, JointName.LEFT_WRIST,
        JointName.RIGHT_WRIST, JointName.LEFT_HIP, JointName.RIGHT_HIP,
    ),
})
