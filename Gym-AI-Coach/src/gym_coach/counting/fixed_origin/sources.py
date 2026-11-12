"""Verify locked parent and saved comparisons; never replay/train during source intake."""
import importlib.util
from ..research.data import local, identity
from ..models import canonical_hash

PARENT='outputs/squat_count_body_normalized/prepare-5b23ef3e8ed54cc5bacfe226db03f2c6/manifest.json'
DIAGNOSIS='outputs/squat_body_normalized_fold1_diagnosis/run-c18372526a494a48868f3c57f9771cfc'
REVIEW='outputs/squat_body_normalized_fold1_diagnosis/review-62cbae1b12b44a519b8a71375764c504/verification.json'


def verify(root):
    path=local(root,'scripts/diagnose_squat_body_normalized_fold1.py')
    spec=importlib.util.spec_from_file_location('fixed_origin_parent_verifier',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ledger=module.helper().Sources(root)
    ledger.add(path)
    parent,data,saved,checkpoints,verified=module.verify(root,ledger)
    ledger.inventory(DIAGNOSIS)
    old=ledger.json(DIAGNOSIS+'/verification.json',True)
    review=ledger.json(REVIEW)
    if old!=verified or review.get('status')!='VERIFIED' or not review['paired_metrics_exact']:
        raise ValueError('Parent diagnosis/review mismatch')
    if review['epoch_logs']!=3000 or review['body_replay']!=66 or not review['replay_exact']:
        raise ValueError('Parent review lacks verified endpoints/replay')
    controls={}
    for seed,q in saved['seeds'].items():
        controls[seed]=dict(raw=[dict(r,method='gru' if r['method']=='raw58' else r['method']) for r in q['records'] if r['method']!='body_normalized'],
                            body=[r for r in q['records'] if r['method']=='body_normalized'])
    evidence=dict(parent_manifest=identity(root,local(root,PARENT)),parent_manifest_sha256=canonical_hash(parent),
        body_comparison=identity(root,local(root,module.COMPARISON)/'result.json'),
        diagnosis=identity(root,local(root,DIAGNOSIS)/'verification.json'),review=identity(root,local(root,REVIEW)),
        controls=parent['sources']['controls'],diagnostic_tool=parent['sources']['diagnostic_tool'],
        sources=ledger.preserve()['sources'],parent_verification=verified,new_model_forwards=0,training_steps=0,baseline_fits=0)
    return parent,data,controls,evidence
