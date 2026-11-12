"""Read-only squat feedback projection and a separate synthetic temporal gate."""
from pydantic import ValidationError

from ..contracts import ErrorCategory,Exercise,SourceTier,Stage
from ..features import GEOMETRY_REGISTRY,FeatureInputContext,observe_geometry_feature
from ..features.context import _strict_json
from ..features.models import content_hash
from ..flags import FlagCode,FlagLocation,GymError,Severity,StructuredFlag
from ..preprocessing.models import PreprocessingResult,content_hash as preprocessing_hash
from ..segmentation import SquatSignalSample,SquatTemporalInput,SquatSegmentationProfile,SquatSegmentationResult,segment_squat
from ..segmentation.squat import temporal_flag
from .squat_models import (EvidenceInterval,FeedbackMetric,FeedbackFrame,FeedbackSource,
                           FeedbackMessage,SquatFeedbackResult,SQUAT_IDS)


def feedback_error(reason,detail):
    return GymError(StructuredFlag(code=FlagCode.INVALID_FEEDBACK_INPUT,category=ErrorCategory.INPUT_ERROR,
        severity=Severity.ERROR,stage=Stage.REPORTING,source_tier=SourceTier.S3,message=detail,
        evidence={'reason':reason,'component':'squat_feedback_builder'}))


def interval(start,end,fps):
    return EvidenceInterval(start_frame=start,end_frame=end,start_seconds=start/fps,end_seconds=end/fps)


def phase_at(result,index,fps):
    if result.status=='blocked':return 'Unavailable',None,tuple(dict.fromkeys(f.code for f in result.flags))
    for rep in result.repetitions:
        if index==rep.start_frame:return 'Start/Top',interval(index,index,fps),()
        if index==rep.end_frame:return 'Return/Top',interval(index,index,fps),()
        if index==rep.bottom_frame:return 'Bottom',interval(index,index,fps),()
        if rep.start_frame<index<rep.bottom_frame:return 'Descent',interval(rep.start_frame,rep.bottom_frame,fps),()
        if rep.bottom_frame<index<rep.end_frame:return 'Ascent',interval(rep.bottom_frame,rep.end_frame,fps),()
    if result.input is not None and result.input.samples[index].value is None:
        return 'Unavailable',None,result.input.samples[index].reasons
    if result.repetitions:
        code=FlagCode.PARTIAL_REP_START if index<result.repetitions[0].start_frame else FlagCode.PARTIAL_REP_END
    else:return 'Unavailable',None,tuple(dict.fromkeys(f.code for f in result.flags))
    return 'Unavailable',None,(code,)


def blocked_segmentation(flags,profile):
    return SquatSegmentationResult(input=None,input_hash=None,profile=profile,profile_hash=profile.content_hash(),
        status='blocked',smoothed_values=(),repetitions=(),candidate_count=0,flags=tuple(flags))


def synthetic_temporal_gate(source):
    """A test capability, not permission inferred from numeric0.4 observations.

    Production/official temporal admission is deliberately not implemented yet.
    This gate checks raw source identity, sequence and exact bilateral metrics.
    """
    q=source.quality;flags=[]
    def deny(reason):flags.append(temporal_flag(FlagCode.TEMPORAL_GATE_BLOCKED,reason,q.video_id))
    if not source.metadata.video_id.startswith('synthetic-'):deny('Real temporal validation remains blocked at Stage5.7')
    if q.exercise!=Exercise.BODY_WEIGHT_SQUATS:deny('Only BodyWeightSquats is supported')
    if q.viewpoint.value!='side':deny('Synthetic viewpoint must satisfy the declared side policy')
    if not q.eligibility.eligible or q.eligibility.action!='allow':deny('Sequence is not allow; framewise metric allow is insufficient')
    for key in ('squat_knee_angle_left_deg','squat_knee_angle_right_deg'):
        metric=q.metric_eligibility.get(key)
        if (metric is None or not metric.eligible or metric.action!='allow'
                or set(metric.required_joints)!=set(GEOMETRY_REGISTRY.get(key).required_joints)
                or len(metric.required_joints)!=3):deny('Exact bilateral metric allow required: '+key)
    return tuple(flags)


def build_squat_feedback(result:PreprocessingResult,*,profile:SquatSegmentationProfile|None=None,side='left')->SquatFeedbackResult:
    """Project actual registry observations from an explicitly synthetic fixture.

    Stored real data must use the blocked preview builder in the demo module.
    No existing report is reevaluated/promoted, and numeric0.4 keeps its false
    temporal permission. Separate new temporal input is synthetic-only.
    """
    try:
        if not isinstance(result,PreprocessingResult):raise ValueError('Expected PreprocessingResult')
        result=PreprocessingResult.model_validate_json(_strict_json(result),strict=True)
        if not result.metadata.video_id.startswith('synthetic-'):raise ValueError('Only explicit synthetic feedback fixtures are admitted here')
        if result.quality.exercise!=Exercise.BODY_WEIGHT_SQUATS:raise ValueError('Only BodyWeightSquats feedback is implemented')
        if side not in ('left','right'):raise ValueError('Choose one fixed side explicitly; no automatic fallback')
        profile=SquatSegmentationProfile.model_validate_json(_strict_json(profile or SquatSegmentationProfile()),strict=True)
    except (ValidationError,ValueError,TypeError,AttributeError) as exc:raise feedback_error('invalid_synthetic_source',str(exc)) from exc
    info,q=result.metadata,result.quality
    before=preprocessing_hash(result);all_observations=[];frames=[];samples=[]
    for raw,processed in zip(result.raw_frames,result.frames):
        context=FeatureInputContext(frame=processed,quality=q)
        observations=tuple(observe_geometry_feature(key,context) for key in SQUAT_IDS)
        all_observations.append(observations)
        obs=next(o for o in observations if o.feature_id==f'squat_knee_angle_{side}_deg')
        samples.append(SquatSignalSample(frame_index=obs.frame_index,timestamp_seconds=obs.timestamp_seconds,value=obs.value,
            side_source=side,joint_provenance=tuple(j.status for j in obs.joint_inputs),
            imputation_references=tuple(j.interpolation for j in obs.joint_inputs if j.interpolation is not None),
            source_observation_hash=obs.content_hash(),reasons=obs.reasons))
    blocked=synthetic_temporal_gate(result)
    if blocked:segmentation=blocked_segmentation(blocked,profile)
    else:
        temporal=SquatTemporalInput(video_id=info.video_id,fps=info.fps,source_hash=before,
            evidence_basis='genuine_synthetic_qualitygate',quality=q,quality_hash=content_hash(q),samples=tuple(samples))
        segmentation=segment_squat(temporal,profile=profile)
    for raw,processed,observations in zip(result.raw_frames,result.frames,all_observations):
        phase,evidence,reasons=phase_at(segmentation,processed.frame_index,info.fps)
        metrics=tuple(FeedbackMetric(feature_id=obs.feature_id,definition_hash=obs.availability.definition_hash,
            status=obs.status,value=obs.value,evidence=interval(obs.frame_index,obs.frame_index,info.fps),
            quality_action=q.metric_eligibility[obs.feature_id].action if obs.feature_id in q.metric_eligibility else 'not_run',
            observation_hash=obs.content_hash() if obs.feature_id in q.metric_eligibility else None,
            input_hash=obs.availability.input_hash if obs.feature_id in q.metric_eligibility else None,
            joint_inputs=obs.joint_inputs if obs.feature_id in q.metric_eligibility else (),reasons=obs.reasons,flags=obs.flags) for obs in observations)
        frames.append(FeedbackFrame(frame_index=raw.frame_index,timestamp_seconds=raw.timestamp_seconds,
            phase=phase,phase_evidence=evidence,phase_reasons=reasons,source_frame_hash=processed.raw_frame_hash,
            overlay_source='raw_pose',overlay_joints=raw.joints,metrics=metrics))
    source=FeedbackSource(kind='synthetic_contract',evidence_status='synthetic_fixture',video_id=info.video_id,
        fps=info.fps,frame_count=info.frame_count,width=info.width,height=info.height,preview_sampling='all_synthetic_frames',
        preprocessing_hash=before,raw_sequence_hash=result.raw_sequence_hash,quality_hash=content_hash(q))
    feedback=SquatFeedbackResult(status='synthetic_experiment',registry_hash=GEOMETRY_REGISTRY.content_hash(),
        source=source,source_hash=content_hash(source),quality=q,segmentation=segmentation,segmentation_hash=segmentation.content_hash(),
        frames=tuple(frames),messages=(FeedbackMessage(kind='experimental_description',text='Pha và chu kỳ ứng viên từ dữ liệu synthetic; chưa kiểm chứng trên vận động viên.'),
            FeedbackMessage(kind='assessment_blocked',text='Đánh giá kỹ thuật chưa khả dụng; không có điểm form hoặc kết luận đúng/sai.')),
        flags=blocked)
    assert preprocessing_hash(result)==before
    return feedback
