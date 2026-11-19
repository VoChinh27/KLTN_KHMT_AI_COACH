"""Independent 2D reference intake/evaluation. No inference or gate recomputation.

Pixel distances compare observed raw landmarks; angle errors compare ONLY saved
available processed measurements. These branches deliberately have different masks.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Literal

from pydantic import Field, StrictBool, model_validator

from ..contracts import ContractModel
from ..flags import GymError
from ..geometry import joint_angle

VERSION = "squat-2d-reference-v1"
PROTOCOL = "squat-blinded-pixel-reference-v1"
SIDES = ("left", "right")
PARTS = ("hip", "knee", "ankle")
JOINTS = tuple(f"{s}_{j}" for s in SIDES for j in PARTS)
STATUS = dict(validation_status="NOT_VALIDATED", official_eligibility=False,
              assessment=False, g57="BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT", g6="IN_PROGRESS")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode())


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class FrameIdentity(ContractModel):
    sample_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    video_path: str = Field(pattern=r"^Data/UCF-101_GYM/BodyWeightSquats/[^/\\]+\.avi$")
    video_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    frame_index: int = Field(ge=0, strict=True)
    timestamp_seconds: float = Field(ge=0, strict=True)
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class LandmarkLabel(ContractModel):
    state: Literal["pending", "visible", "occluded", "out_of_frame", "uncertain"]
    x: float | None = Field(default=None, strict=True)
    y: float | None = Field(default=None, strict=True)
    reason: str | None = None

    @model_validator(mode="after")
    def coordinates(self):
        if self.state == "visible":
            if self.x is None or self.y is None or self.reason is not None:
                raise ValueError("Visible landmark requires finite xy and no unavailable reason")
        elif self.x is not None or self.y is not None or not self.reason or not self.reason.strip():
            raise ValueError("Unavailable/pending landmark requires null xy and explicit reason")
        return self


class ReferenceFrame(ContractModel):
    identity: FrameIdentity
    annotation_status: Literal["pending", "completed"]
    annotator: str | None = None
    method: Literal["manual_blinded_pixel", "synthetic_fixture"] | None = None
    annotated_at: datetime | None = None
    independent_of_model: StrictBool | None = None
    blinded_to_predictions: StrictBool | None = None
    provenance_note: str | None = None
    landmarks: dict[str, LandmarkLabel]

    @model_validator(mode="after")
    def reference_policy(self):
        if set(self.landmarks) != set(JOINTS):
            raise ValueError("Exactly six named anatomical-side joints required; no side inference")
        for label in self.landmarks.values():
            if label.state == "visible" and not (0 <= label.x <= self.identity.width-1 and
                                                   0 <= label.y <= self.identity.height-1):
                raise ValueError("Reference visible pixel coordinates outside image centers")
        if self.annotation_status == "pending":
            if any(v is not None for v in (self.annotator, self.method, self.annotated_at,
                                          self.independent_of_model, self.blinded_to_predictions, self.provenance_note)):
                raise ValueError("Pending template must not invent annotator/provenance")
            if any(p.state != "pending" for p in self.landmarks.values()):
                raise ValueError("Pending frame must remain unfilled; submit completed frames separately")
        else:
            if (not self.annotator or not self.annotator.strip() or not self.method or
                    not self.annotated_at or self.annotated_at.utcoffset() is None or
                    self.independent_of_model is not True or self.blinded_to_predictions is not True or
                    not self.provenance_note or not self.provenance_note.strip() or
                    any(p.state == "pending" for p in self.landmarks.values())):
                raise ValueError("Completed reference requires independent/blinded source, person/method/time/note")
        return self


class ReferenceDocument(ContractModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    protocol_version: Literal["squat-blinded-pixel-reference-v1"] = PROTOCOL
    protocol_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sampling_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    coordinate_convention: Literal["pixel_centers_x_right_y_down_raw_norm_times_width_height"] = (
        "pixel_centers_x_right_y_down_raw_norm_times_width_height")
    frames: tuple[ReferenceFrame, ...]

    @model_validator(mode="after")
    def unique_frames(self):
        ids = [f.identity.sample_id for f in self.frames]
        keys = [(f.identity.video_sha256, f.identity.frame_index) for f in self.frames]
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise ValueError("Duplicate reference sample/frame; evaluate repeated raters separately")
        return self


def statistics_for(values):
    """Unweighted observations; p90 uses linear interpolation at 0.9*(n-1)."""
    values = sorted(values)
    if not values:
        return dict(n=0, mae=None, median_absolute_error=None, p90_absolute_error=None)
    index = .9*(len(values)-1)
    lo, hi = math.floor(index), math.ceil(index)
    return dict(n=len(values), mae=statistics.mean(values), median_absolute_error=statistics.median(values),
                p90_absolute_error=values[lo]+(values[hi]-values[lo])*(index-lo))


def evaluate(manifest: dict, predictions: dict, reference: ReferenceDocument) -> dict:
    """All identities checked before errors. No labels => null metrics, never zero error.

    File-backed CLI additionally verifies bytes against package manifests. This
    pure entry point accepts already verified snapshots for synthetic unit tests.
    """
    # Frozen Pydantic instances can still contain mutable nested dictionaries.
    # Repeat strict intake here for callers of the Python API as well as the CLI.
    reference=ReferenceDocument.model_validate_json(reference.model_dump_json(warnings="error"),strict=True)
    if manifest["data_kind"] not in ("real_development","synthetic"):
        raise ValueError("Unknown reference dataset kind")
    if (manifest["version"] != VERSION or predictions["version"] != VERSION or
            reference.protocol_sha256 != manifest["protocol_sha256"] or
            reference.sampling_sha256 != manifest["sampling_sha256"] or
            predictions["manifest_hash"] != canonical_hash(manifest)):
        raise ValueError("Version/protocol/sampling/prediction identity mismatch")
    for key, value in STATUS.items():
        if manifest[key] != value or predictions[key] != value:
            raise ValueError("Disabled gate/validation status mismatch")
    targets = [FrameIdentity.model_validate(s["identity"]) for s in manifest["samples"]]
    expected = {t.sample_id: t for t in targets}
    if len(expected) != len(targets) or len({(t.video_sha256,t.frame_index) for t in targets}) != len(targets):
        raise ValueError("Duplicate target sample")
    supplied = {}
    for frame in reference.frames:
        if frame.identity != expected.get(frame.identity.sample_id):
            raise ValueError("Reference video/hash/frame/dimensions/timestamp/image linkage mismatch")
        if (manifest["data_kind"] == "real_development" and frame.annotation_status == "completed"
                and frame.method != "manual_blinded_pixel"):
            raise ValueError("Synthetic/model source cannot certify real reference")
        supplied[frame.identity.sample_id] = frame
    model = {}
    for row in predictions["frames"]:
        identity = FrameIdentity.model_validate(row["identity"])
        if identity != expected.get(identity.sample_id) or identity.sample_id in model:
            raise ValueError("Prediction identity mismatch/duplicate")
        if set(row["raw_joints"]) != set(JOINTS) or set(row["processed"]) != set(SIDES):
            raise ValueError("Missing/wrong model joints/sides")
        for side, obs in row["processed"].items():
            if (obs["feature_id"] != f"squat_knee_angle_{side}_deg" or obs["side"] != side or
                    obs["unit"] != "degrees" or obs["status"] not in ("available","unavailable") or
                    obs["gate_action"] not in ("allow","degrade","skip")):
                raise ValueError("Processed feature/side/unit/gate mismatch")
            v=obs["value"]
            if obs["status"] == "available":
                if (obs["gate_action"] != "allow" or not isinstance(v,(float,int)) or isinstance(v,bool)
                        or not math.isfinite(v) or not 0 <= v <= 180):
                    raise ValueError("Available angle must retain saved allow/finite degrees")
            elif v is not None or not obs["reasons"]:
                raise ValueError("Unavailable measurement must preserve null and reasons")
        for joint in row["raw_joints"].values():
            if joint["status"] not in ("observed","low_visibility","missing"):
                raise ValueError("Raw provenance invalid")
            xy=(joint["x"],joint["y"])
            if joint["status"] == "missing":
                if xy != (None,None):raise ValueError("Missing raw coordinates must be null")
            elif any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in xy):
                raise ValueError("Raw coordinates must be finite numeric")
        model[identity.sample_id]=row
    if set(model) != set(expected):
        raise ValueError("Prediction target denominator incomplete")
    rows=[]
    for target in targets:
        annotation=supplied.get(target.sample_id);prediction=model[target.sample_id]
        complete=annotation is not None and annotation.annotation_status == "completed"
        for side in SIDES:
            for part in PARTS:
                name=f"{side}_{part}"; label=annotation.landmarks[name] if complete else None
                raw=prediction["raw_joints"][name]; model_reasons=[]; ref_reasons=[]
                if raw["status"] != "observed":model_reasons.append("raw_"+raw["status"])
                elif not (0 <= raw["x"]*target.width <= target.width-1 and
                          0 <= raw["y"]*target.height <= target.height-1):
                    model_reasons.append("raw_out_of_frame")
                if label is None:ref_reasons.append("reference_pending")
                elif label.state != "visible":ref_reasons.extend(["reference_"+label.state,label.reason])
                error=None
                if not model_reasons and not ref_reasons:
                    error=math.hypot(raw["x"]*target.width-label.x,raw["y"]*target.height-label.y)
                rows.append(dict(sample_id=target.sample_id,video_id=target.video_id,side=side,joint=name,
                    branch="raw_landmark_px",absolute_error=error,unit="pixels",
                    model_unavailable_reasons=model_reasons,reference_unavailable_reasons=ref_reasons))
            saved=prediction["processed"][side];model_reasons=list(saved["reasons"]) if saved["status"]!="available" else []
            labels=[annotation.landmarks[f"{side}_{p}"] for p in PARTS] if complete else []
            ref_reasons=[];angle=None
            if not complete:ref_reasons=["reference_pending"]
            elif any(p.state!="visible" for p in labels):
                ref_reasons=sorted({"reference_"+p.state for p in labels if p.state!="visible"})
            else:
                try:angle=joint_angle(*[(p.x,p.y) for p in labels])
                except GymError as exc:ref_reasons=["reference_"+exc.flag.code.value]
            error=abs(saved["value"]-angle) if not model_reasons and not ref_reasons else None
            rows.append(dict(sample_id=target.sample_id,video_id=target.video_id,side=side,joint=None,
                branch="processed_angle_deg",absolute_error=error,unit="degrees",reference_angle=angle,
                model_unavailable_reasons=model_reasons,reference_unavailable_reasons=ref_reasons))
    def summarize(selected):
        return dict(**statistics_for([r["absolute_error"] for r in selected if r["absolute_error"] is not None]),
            denominator=len(selected),reference_available=sum(not r["reference_unavailable_reasons"] for r in selected),
            model_available=sum(not r["model_unavailable_reasons"] for r in selected),
            model_unavailable=sum(bool(r["model_unavailable_reasons"]) for r in selected),
            reference_unavailable=sum(bool(r["reference_unavailable_reasons"]) for r in selected),
            exclusion_reasons=dict(Counter(x for r in selected for x in set(
                r["model_unavailable_reasons"]+r["reference_unavailable_reasons"]))))
    branches={}
    for branch in ("raw_landmark_px","processed_angle_deg"):
        subset=[r for r in rows if r["branch"]==branch]
        groups={}
        for video in sorted({t.video_id for t in targets}):
            for side in SIDES:
                groups[f"{video}/{side}"]=summarize([r for r in subset if r["video_id"]==video and r["side"]==side])
        means=[g["mae"] for g in groups.values() if g["n"]]
        branches[branch]=dict(overall=summarize(subset),by_clip_side=groups,
            by_side={s:summarize([r for r in subset if r["side"]==s]) for s in SIDES},
            clip_side_macro_mae=statistics.mean(means) if means else None,contributing_clip_sides=len(means))
    completed=sum(f.annotation_status=="completed" for f in reference.frames)
    return dict(version=VERSION,**STATUS,status="REFERENCE_PENDING" if not completed else "REFERENCE_DIAGNOSTIC_ONLY",
        manifest_hash=canonical_hash(manifest),reference_hash=canonical_hash(reference.model_dump(mode="json")),
        counts=dict(sample_frames=len(targets),completed_reference_frames=completed,pending_reference_frames=len(targets)-completed),
        weighting="overall micro = each comparable joint/angle equally; macro = each nonempty clip-side MAE equally; empty groups retained",
        percentile="linear interpolation at 0.9*(n-1)",branches=branches,rows=rows,
        raw_diagnostic_angle_computed=False,accuracy_acceptance_threshold=None,
        limitation="Self-declared annotation provenance is not authenticated ground truth; no form/physical-angle/holdout claim")


def checked_file(root, relative, expected):
    relative=Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Local package relative path required")
    path=(Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError("Package path escaped")
    data=path.read_bytes()
    if digest(data)!=expected["sha256"] or len(data)!=expected["size_bytes"]:
        raise ValueError(f"File hash/size mismatch: {relative}")
    return data


def evaluate_package(package, annotation, output):
    package=Path(package);output=Path(output)
    if output.exists():raise FileExistsError(output)
    integrity=json.loads((package/"package-integrity.json").read_bytes())
    for name,expected in integrity["files"].items():checked_file(package,name,expected)
    sampling_bytes=checked_file(package,"sampling.json",integrity["sampling"])
    checked_file(package,"source-verification.json",integrity["source_verification"])
    sampling=json.loads(sampling_bytes)
    manifest=json.loads((package/"annotation_package/manifest.json").read_bytes())
    if (manifest["sampling_sha256"]!=digest(sampling_bytes) or
            manifest["protocol_sha256"]!=digest((package/"annotation_package/PROTOCOL.md").read_bytes())):
        raise ValueError("Frozen sampling/protocol bytes mismatch")
    planned={c["video_id"]:c for c in sampling["selection"]}
    for sample in manifest["samples"]:
        identity=FrameIdentity.model_validate(sample["identity"])
        clip=planned.get(identity.video_id)
        if clip is None or any((identity.video_path!=clip["path"],identity.video_sha256!=clip["video_sha256"],
                identity.width!=clip["width"],identity.height!=clip["height"],
                identity.frame_index not in clip["frame_indices"],identity.timestamp_seconds!=identity.frame_index/clip["fps"])):
            raise ValueError("Image/sample not linked to frozen selection")
        name="annotation_package/"+sample["image_path"]
        image=checked_file(package,name,integrity["files"][name])
        # PNG IHDR is fixed-position; file hash protects the complete image.
        import struct
        if (digest(image)!=identity.image_sha256 or image[:8]!=b"\x89PNG\r\n\x1a\n" or
                struct.unpack(">II",image[16:24])!=(identity.width,identity.height)):
            raise ValueError("Reference image hash/native dimensions mismatch")
    predictions=json.loads((package/"evaluation_inputs/predictions.json").read_bytes())
    reference=ReferenceDocument.model_validate_json(Path(annotation).read_bytes(),strict=True)
    result=evaluate(manifest,predictions,reference)
    result["annotation_file_sha256"]=digest(Path(annotation).read_bytes())
    result["package_integrity_sha256"]=digest((package/"package-integrity.json").read_bytes())
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"evaluation.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    result=evaluate_package(args.package,args.annotations,args.output_dir)
    print(json.dumps(dict(status=result["status"],counts=result["counts"])))


if __name__=="__main__":main()
