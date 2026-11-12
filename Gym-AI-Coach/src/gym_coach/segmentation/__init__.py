"""Experimental synthetic squat temporal API; real validation remains blocked."""
from .models import (SquatSignalSample,SquatTemporalInput,SquatSegmentationProfile,
                     SquatSegmentationResult,RepetitionCandidate)
from .squat import segment_squat

__all__=['SquatSignalSample','SquatTemporalInput','SquatSegmentationProfile',
         'SquatSegmentationResult','RepetitionCandidate','segment_squat']
