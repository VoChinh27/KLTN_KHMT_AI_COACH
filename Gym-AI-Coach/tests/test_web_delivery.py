"""Handoff links, packaging and explicitly retired entry points."""
from pathlib import Path
import re
import os
import subprocess
import sys
import tomllib
from urllib.parse import unquote

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('relative',['README.md','docs/DOCUMENTATION_INDEX.md',
    'docs/SQUAT_WEB_DEMO_REPORT.md','docs/PROJECT_CLEANUP_REPORT.md'])
def test_active_handoff_local_links_exist(relative):
    path=ROOT/relative
    for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        if '://' in link or link.startswith('#'):continue
        target=unquote(link.split('#',1)[0]).strip('<>')
        assert (path.parent/target).exists(),(relative,target)


def test_active_index_classifies_every_markdown_document():
    index=(ROOT/'docs/DOCUMENTATION_INDEX.md').read_text(encoding='utf-8')
    for path in (ROOT/'docs').glob('*.md'):
        if path.name!='DOCUMENTATION_INDEX.md':assert path.name in index,path.name


def test_web_assets_declared_for_package_and_no_active_tk_consumer():
    project=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))
    patterns=project['tool']['setuptools']['package-data']['gym_coach']
    for suffix in ('html','js','css'):
        assert 'web_assets/*.'+suffix in patterns
        assert list((ROOT/'src/gym_coach').glob('web_assets/*.'+suffix))
    assert not (ROOT/'src/gym_coach/desktop.py').exists()
    assert not (ROOT/'src/gym_coach/desktop_controller.py').exists()
    for path in (ROOT/'src/gym_coach').rglob('*.py'):
        assert 'import tkinter' not in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('entry',['gym_coach.web','gym_coach.reporting.single_video',
    'gym-video-info','gym-pose-extract','gym-quality-audit','gym-pose-preprocess'])
def test_retained_main_entrypoint_help(entry):
    if entry.startswith('gym_coach.'):
        command=[sys.executable,'-B','-m',entry,'--help']
    else:
        project=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))
        assert entry in project['project']['scripts']
        executable=Path(sys.executable).parent/(entry+('.exe' if os.name=='nt' else ''))
        assert executable.is_file()
        command=[str(executable),'--help']
    process=subprocess.run(command,capture_output=True,timeout=30)
    assert process.returncode==0,process.stderr
    assert b'usage:' in process.stdout.lower()
