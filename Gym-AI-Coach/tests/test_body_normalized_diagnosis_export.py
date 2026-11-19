"""Offline export checks; no video decoding or model execution in tests."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('body_export', Path(__file__).parents[1] / 'scripts/export_body_normalized_diagnosis.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_raw_height_units_invert_to_pixels_without_aspect_double_scale():
    row = [0.] * 58
    row[0], row[1] = (100 - 160) / 240, (50 - 120) / 240
    assert e.pixel_xy(row, 0, 240, 320) == pytest.approx((100, 50))
    assert e.JOINTS == ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle')


def test_linkage_rejects_source_mutation_and_workspace_escape(tmp_path):
    p = tmp_path / 'source.json'
    p.write_text('{}')
    link = dict(path=p.name, **e.file_info(p))
    assert e.linked(tmp_path, link) == p
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        e.linked(tmp_path, link)
    with pytest.raises(ValueError, match='outside workspace'):
        e.linked(tmp_path, dict(path='../outside.json', sha256='0' * 64))


def test_table_preserves_signs_zero_and_escapes_field_separators():
    table = e.markdown_table(['Video', 'ΔAE'], [['v|x\ny', -1.25], ['zero', 0.]])
    assert 'v\\|x y' in table and '-1.250000' in table and '0.000000' in table
