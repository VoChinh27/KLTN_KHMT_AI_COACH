"""Run orchestration using JSON mock checkpoints, never a learning loop."""
import copy
from pathlib import Path
import subprocess
import pytest
from gym_coach.counting.fixed_origin import orchestrate as o,evaluation as ev,manifest as m
from gym_coach.counting.fixed_origin.contract import SEEDS
from gym_coach.counting.fixed_origin.safety import no_learning
from gym_coach.counting.inner_cv.manifest import sealed,unseal
from gym_coach.counting.readiness import write_json,file_info
from gym_coach.counting.research.data import read,identity
from gym_coach.counting.models import canonical_hash
import test_count_fixed_origin as f
import test_count_train_capacity as diagnosis


@pytest.fixture(autouse=True)
def guard():
    with no_learning():yield


class Mock:
    def __init__(self,root,man):self.root=root;self.man=man;self.calls=[];self.interrupt=None;self.aggregates=0
    def load(self,seed,path,selected,fold):
        if file_info(path)!=read(str(path)+'.sha256.json'):raise ValueError('Mock latest corrupt; no fallback')
        cp=read(path)
        if cp['kind']!='MOCK_NO_TRAINING' or cp['identity']!=m.fold_identity(selected,fold):raise ValueError('Mock identity')
        return cp
    def train(self,seed,fold,dest,resume):
        self.calls.append((seed,fold,dest,resume));dest.mkdir(parents=True,exist_ok=False)
        sel=m.select(self.man,seed);ident=m.fold_identity(sel,fold)
        first=read(resume)['epoch']+1 if resume else 0
        write_json(dest/'run.json',dict(identity=ident,config=sel['config'],
            initial_fresh_state_sha256=self.man['sources']['controls'][str(seed)]['initializations'][str(fold)],
            resume_checkpoint=str(resume) if resume else None,resume_identity=file_info(resume) if resume else None,first_epoch=first))
        e=3 if self.interrupt==(seed,fold) else 499
        path=dest/f'epoch-{e:03d}.pt';write_json(path,dict(kind='MOCK_NO_TRAINING',identity=ident,epoch=e))
        write_json(str(path)+'.sha256.json',file_info(path))
        if e!=499:raise RuntimeError('Simulated user interruption')
        write_json(dest/'complete.json',dict(identity=ident,checkpoint=file_info(path),final_checkpoint=path.name,endpoint=self.man['endpoint']))
    def aggregate(self,paths):
        self.aggregates+=1;assert len(paths)==6
        tr,held,c=f.inputs(self.man)
        q={str(s):ev.compare(self.man,s,tr,held,c,diagnosis.d) for s in SEEDS}
        return dict(status='COMPLETE_DEVELOPMENT_ONLY',manifest_sha256=canonical_hash(self.man),seeds=q,combined=ev.combine(q),limitation='MOCK fixture, no training')


def setup(tmp_path):
    man=f.manifest();run=tmp_path/'run';run.mkdir();(run/'attempts').mkdir();mock=Mock(tmp_path,man)
    return man,run,mock


def execute(root,run,man,mock):return o.execute(root,run,man,{}, {},mock.train,mock.load,mock.aggregate)


def test_fresh_serial_six_skip_verified_complete_and_keep_comparison(tmp_path):
    man,run,mock=setup(tmp_path);comparison=execute(tmp_path,run,man,mock)
    assert [(s,f) for s,f,_,_ in mock.calls]==[(s,f) for s in SEEDS for f in range(3)]
    assert all(r is None for _,_,_,r in mock.calls) and mock.aggregates==1
    before={p:file_info(p) for p in run.rglob('*') if p.is_file()}
    assert execute(tmp_path,run,man,mock)==comparison and len(mock.calls)==6 and mock.aggregates==1
    assert all(file_info(p)==h for p,h in before.items())


def test_interrupt_resume_new_folder_preserve_previous_and_record_lineage(tmp_path):
    man,run,mock=setup(tmp_path);mock.interrupt=(SEEDS[0],1)
    with pytest.raises(RuntimeError,match='interruption'):execute(tmp_path,run,man,mock)
    old={p:file_info(p) for p in run.rglob('*') if p.is_file()};incomplete=mock.calls[-1][2]
    mock.interrupt=None;comparison=execute(tmp_path,run,man,mock)
    assert comparison.is_dir() and len(mock.calls)==7
    resumed=mock.calls[2];assert resumed[2]!=incomplete and resumed[3]==incomplete/'epoch-003.pt'
    assert all(file_info(p)==h for p,h in old.items())
    assert read(resumed[2]/'run.json')['first_epoch']==4


@pytest.mark.parametrize('case',['corrupt','orphan','wrong_identity','bad_resume','incomplete_endpoint','duplicate_endpoint','unknown_dir','wrong_complete'])
def test_resume_fails_closed_without_old_checkpoint_fallback(tmp_path,case):
    man,run,mock=setup(tmp_path);mock.interrupt=(SEEDS[0],0)
    with pytest.raises(RuntimeError):execute(tmp_path,run,man,mock)
    dest=mock.calls[0][2];p=dest/'epoch-003.pt'
    if case=='corrupt':p.write_bytes(b'corrupt')
    if case=='orphan':write_json(dest/'epoch-004.json',{})
    if case=='wrong_identity':
        cp=read(p);cp['identity']['seed']=SEEDS[1];p.write_text(__import__('json').dumps(cp));Path(str(p)+'.sha256.json').write_text(__import__('json').dumps(file_info(p)))
    if case=='bad_resume':
        event=unseal(read(run/'attempts/0001.json'));event['sequence']=2;event['output']=str((run/'seed-20261026/fold-0/attempt-0002').relative_to(tmp_path));event['resume']=None
        write_json(run/'attempts/0002.json',sealed(event))
    if case in ('incomplete_endpoint','duplicate_endpoint','wrong_complete'):
        cp=read(p);cp['epoch']=499;write_json(dest/'epoch-499.pt',cp);write_json(dest/'epoch-499.pt.sha256.json',file_info(dest/'epoch-499.pt'))
        if case!='incomplete_endpoint':write_json(dest/'complete.json',dict(identity=cp['identity'],checkpoint=file_info(dest/'epoch-499.pt'),final_checkpoint='bad.pt' if case=='wrong_complete' else 'epoch-499.pt',endpoint=man['endpoint']))
        if case=='duplicate_endpoint':
            event=unseal(read(run/'attempts/0001.json'));event['sequence']=2;event['output']=str((run/'seed-20261026/fold-0/attempt-0002').relative_to(tmp_path));write_json(run/'attempts/0002.json',sealed(event))
    if case=='unknown_dir':(run/'seed-20261026/fold-0/attempt-untracked').mkdir()
    before=len(mock.calls);mock.interrupt=None
    with pytest.raises((ValueError,OSError)):execute(tmp_path,run,man,mock)
    assert len(mock.calls)==before and mock.aggregates==0


def test_aggregate_failure_new_output_on_retry_no_overwrite(tmp_path):
    man,run,mock=setup(tmp_path)
    def fail(paths):raise ValueError('Mock aggregation failure')
    with pytest.raises(ValueError):o.execute(tmp_path,run,man,{}, {},mock.train,mock.load,fail)
    first=list(run.glob('comparison-*'));assert len(first)==1
    comparison=execute(tmp_path,run,man,mock)
    assert comparison not in first and first[0].exists() and len(mock.calls)==6


def test_command_requires_explicit_user_run_before_any_output(tmp_path):
    with pytest.raises(SystemExit):o.main(['--root',str(tmp_path),'--manifest','missing'])
    assert list(tmp_path.iterdir())==[]


def test_run_lock_rejects_concurrent_owner_then_releases(tmp_path):
    with o.exclusive_run(tmp_path):
        with pytest.raises(OSError):
            with o.exclusive_run(tmp_path):pytest.fail('Concurrent run admitted')
    with o.exclusive_run(tmp_path):pass


def test_powershell_parser_only_never_invokes_training():
    script=Path(__file__).parents[1]/'scripts/Run-SquatFixedOrigin.ps1'
    command="$t=$null;$e=$null;$null=[System.Management.Automation.Language.Parser]::ParseFile('"+str(script)+"',[ref]$t,[ref]$e);if($e.Count){throw $e};Write-Output 'PARSE_ONLY_OK'"
    r=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True)
    assert r.returncode==0 and 'PARSE_ONLY_OK' in r.stdout,r.stderr
