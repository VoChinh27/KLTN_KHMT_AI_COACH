"""Test-only deterministic input: genuine QualityGate, never patched eligibility."""
from pathlib import Path

from gym_coach.contracts import Viewpoint
from gym_coach.features import DEFAULT_REGISTRY, SUPPORTED_KNEE_FEATURE_IDS
from gym_coach.pose.mapping import JointName as J
from gym_coach.preprocessing import load_preprocessing_profile, preprocess_pose
from gym_coach.quality import QualityGate, QualityProfile

from test_preprocessing import make_frames, metadata

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ID = "stage5_6_synthetic_contract.avi"


def knee_requirements():
    return {key: DEFAULT_REGISTRY.get(key).required_joints for key in SUPPORTED_KNEE_FEATURE_IDS}


def knee_viewpoints():
    return {key: DEFAULT_REGISTRY.get(key).viewpoint_policy.supported for key in SUPPORTED_KNEE_FEATURE_IDS}


def fixture_input(*, n=31, fps=30.0, viewpoint=Viewpoint.SIDE, changes=None, requirements=None):
    viewpoint = Viewpoint(viewpoint)
    coordinates = {}
    for i in range(n):
        for joint, x, y in ((J.LEFT_KNEE, 0.4, 0.4), (J.LEFT_ANKLE, 0.2, 0.4),
                            (J.RIGHT_KNEE, 0.6, 0.4), (J.RIGHT_ANKLE, 0.8, 0.4)):
            coordinates[i, joint] = ("observed", x, y)
    coordinates.update(changes or {})
    raw = tuple(frame.model_copy(update={"video_id": VIDEO_ID, "viewpoint": viewpoint})
                for frame in make_frames(n=n, fps=fps, changes=coordinates))
    info = metadata(raw, fps=fps).model_copy(update={"video_id": VIDEO_ID, "source_path": Path(VIDEO_ID)})
    # None selects canonical per-feature declarations; {} explicitly tests the
    # QualityGate default movement_analysis-only behavior, without editing output.
    report = QualityGate(QualityProfile()).evaluate(raw,
        metric_requirements=knee_requirements() if requirements is None else requirements,
        metric_viewpoints=knee_viewpoints())
    return raw, info, report


def fixture_result(**options):
    raw, info, report = fixture_input(**options)
    profile = load_preprocessing_profile(ROOT/"configs/preprocessing_s3_v1.toml")
    return preprocess_pose(raw, report, info, profile=profile)
