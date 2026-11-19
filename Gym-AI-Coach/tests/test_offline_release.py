"""Saved-synthetic ZIP integrity checks; no browser, model, or Data access."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

from gym_coach.reporting.offline_release import (
    ZIP_NAME, inventory, main, validate_bundle, verify_archive, write_release,
)

SOURCE = Path(__file__).resolve().parents[1] / (
    "outputs/squat_demo_handoff/run-ab56536633a042e5ac3032bc279f6124/bundle"
)


def files(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("zip-release")
    source = workspace / "outputs" / "saved-handoff"
    source.parent.mkdir()
    shutil.copytree(SOURCE, source)
    output = workspace / "outputs" / "release-new"
    before = files(source)
    manifest = write_release(source, output, workspace=workspace)
    return workspace, source, output, before, manifest


def test_zip_root_exact_bytes_tree_and_preserved_opening_guide(release):
    _, source, output, before, manifest = release
    with zipfile.ZipFile(output / ZIP_NAME) as archive:
        assert set(archive.namelist()) == set(before)
        assert "START_HERE.html" in archive.namelist()
        assert all(archive.read(name) == raw for name, raw in before.items())
        assert archive.testzip() is None
    assert files(source) == before == files(output / "extracted")
    assert before["HUONG_DAN_MO_DEMO.txt"].startswith(b"\xef\xbb\xbf")
    assert manifest["files"] == inventory(before)
    assert manifest["file_count"] == 20
    assert manifest["total_uncompressed_bytes"] == sum(map(len, before.values()))
    assert manifest["source"]["path"] == "outputs/saved-handoff"


def test_zip_manifest_checksums_receipt_and_disabled_gates(release):
    _, _, output, before, manifest = release
    archive = output / ZIP_NAME
    assert manifest["zip"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert manifest["zip"]["size_bytes"] == archive.stat().st_size
    assert manifest == json.loads((output / "release-manifest.json").read_bytes())
    receipt_raw = (output / "verification.json").read_bytes()
    assert manifest["verification"]["sha256"] == hashlib.sha256(receipt_raw).hexdigest()
    receipt = json.loads(receipt_raw)
    assert receipt["source_unchanged"] is True
    assert receipt["detector_run"] is receipt["inference_run"] is receipt["noise_audit_repeated"] is False
    for check in (receipt["source"], receipt["extracted"]):
        assert check["status"] == "PASSED" and check["file_count"] == len(before)
        assert check["local_links_checked"] > 0 and check["snapshot_links_checked"] == 12
        assert check["unavailable_metrics_checked"] == 362 and check["point_cases_checked"] == 9
        assert check["disabled_gates_preserved"] is check["embedded_json_equal"] is True
        assert check["browser_qa"] == "NOT_RUN" and check["validation_status"] == "NOT_VALIDATED"
        assert check["official_eligibility"] is check["assessment_enabled"] is False
        assert check["g57"] == "BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT" and check["g6"] == "IN_PROGRESS"
    assert receipt["extracted"]["extracted_hashes_checked"] == 20
    lines = (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 21
    for line in lines:
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_reads_only_source_and_own_new_release_never_original_paths(release, monkeypatch):
    workspace, source, _, before, _ = release
    output = workspace / "outputs" / "isolated-release"
    from gym_coach import geometry
    from gym_coach.segmentation import squat
    from gym_coach.reporting import squat_builder
    def forbidden(*args, **kwargs):
        raise AssertionError("No geometry, segmentation, feedback producer or subprocess")
    monkeypatch.setattr(geometry, "joint_angle", forbidden)
    monkeypatch.setattr(squat, "segment_squat", forbidden)
    monkeypatch.setattr(squat_builder, "build_squat_feedback", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    old_open = io.open
    def guarded(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)):
            path = Path(file).resolve()
            assert path.is_relative_to(source) or path.is_relative_to(output)
        return old_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(io, "open", guarded)
    write_release(source, output, workspace=workspace)
    assert files(source) == before


def test_existing_output_and_extraction_never_overwritten(release):
    workspace, source, output, _, manifest = release
    before = files(output)
    with pytest.raises(FileExistsError):
        write_release(source, output, workspace=workspace)
    with pytest.raises(FileExistsError):
        verify_archive(output / ZIP_NAME, manifest["files"], output / "extracted")
    assert files(output) == before


def test_repeat_extraction_into_fresh_directory_is_independent(release, tmp_path):
    _, _, output, _, manifest = release
    target = tmp_path / "new-extraction"
    result = verify_archive(output / ZIP_NAME, manifest["files"], target)
    assert result["extracted_hashes_checked"] == 20
    assert inventory(files(target)) == manifest["files"]


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:/escape.txt",
                                    "sources\\escape.txt", "sources/../escape.txt", "Data/a.json"])
def test_unsafe_zip_member_rejected_before_extraction(release, tmp_path, member):
    _, _, _, _, manifest = release
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "x") as stream:
        stream.writestr(member, b"bad")
    extraction = tmp_path / "absent"
    # zipfile itself normalizes backslashes on Windows; either unsafe-name or
    # unexpected-member rejection is valid, before any extraction takes place.
    with pytest.raises(ValueError):
        verify_archive(archive, manifest["files"], extraction)
    assert not extraction.exists()


@pytest.mark.parametrize("case", ["missing", "extra", "case_collision", "symlink", "device", "size", "sha256"])
def test_zip_member_inventory_types_and_hash_tampering(release, tmp_path, case):
    _, _, _, original, manifest = release
    contents = dict(original)
    if case == "missing":
        del contents["START_HERE.html"]
    elif case == "extra":
        contents["extra.txt"] = b"extra"
    elif case == "case_collision":
        contents["start_here.html"] = contents["START_HERE.html"]
    elif case == "size":
        contents["START_HERE.html"] += b"x"
    elif case == "sha256":
        raw = contents["START_HERE.html"]
        contents["START_HERE.html"] = b"x" + raw[1:]
    archive = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive, "x") as stream:
        for name, raw in contents.items():
            if case in ("symlink", "device") and name == "START_HERE.html":
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = ((stat.S_IFLNK if case == "symlink" else stat.S_IFCHR) | 0o777) << 16
                stream.writestr(info, raw)
            else:
                stream.writestr(name, raw)
    with pytest.raises(ValueError):
        verify_archive(archive, manifest["files"], tmp_path / "extract-new")


def test_duplicate_zip_entries_rejected_before_writes(release, tmp_path):
    _, _, _, original, manifest = release
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "x") as stream:
        for name, raw in original.items():
            stream.writestr(name, raw)
        with pytest.warns(UserWarning, match="Duplicate name"):
            stream.writestr("START_HERE.html", original["START_HERE.html"])
    target = tmp_path / "absent"
    with pytest.raises(ValueError, match="duplicate"):
        verify_archive(archive, manifest["files"], target)
    assert not target.exists()


def modified_bundle(original, name, raw):
    contents = dict(original)
    contents[name] = raw
    manifest = json.loads(contents["manifest.json"])
    manifest["files"][name] = hashlib.sha256(raw).hexdigest()
    contents["manifest.json"] = json.dumps(manifest).encode()
    return contents


@pytest.mark.parametrize("case", ["gate", "version", "linkage", "missing", "hash", "source_ref",
                                  "link", "fragment", "external_link", "path_link", "embedded", "escaping"])
def test_bad_saved_bundle_never_packaged(release, case):
    _, _, _, original, _ = release
    contents = dict(original)
    manifest = json.loads(contents["manifest.json"])
    if case in ("gate", "version", "linkage", "hash"):
        if case == "gate":
            manifest["assessment_enabled"] = True
        elif case == "version":
            manifest["handoff_version"] = "unknown"
        elif case == "linkage":
            manifest["linkage"]["synchronized"] = True
        else:
            manifest["files"]["index.html"] = "0" * 64
        contents["manifest.json"] = json.dumps(manifest).encode()
    elif case == "missing":
        del contents["sources/feedback-0.json"]
    elif case == "source_ref":
        data = json.loads(contents["presentation.json"])
        data["source_refs"][0]["sha256"] = "0" * 64
        contents = modified_bundle(contents, "presentation.json", json.dumps(data).encode())
    else:
        page = contents["index.html"].decode("utf-8")
        if case == "embedded":
            page = page.replace('"presentation_version":"gym-squat-showcase-v1"', '"presentation_version":"wrong"')
        elif case == "escaping":
            page = page.replace('"exercise":"BodyWeightSquats"', '"exercise":"<script>"')
        else:
            target = {"link": "missing.json", "fragment": "#missing", "external_link": "https://example.invalid/",
                      "path_link": "../outside.json"}[case]
            page += f'<a href="{target}">bad link</a>'
        contents = modified_bundle(contents, "index.html", page.encode())
    with pytest.raises(ValueError):
        validate_bundle(contents)


@pytest.mark.parametrize("case", ["point", "pause", "metric", "presentation_gate"])
def test_nulls_and_official_states_cannot_be_promoted(release, case):
    _, _, _, contents, _ = release
    data = json.loads(contents["presentation.json"])
    if case == "point":
        data["point_cases"][0]["accepted_point_count"] = 1
    elif case == "pause":
        data["point_cases"][0]["pause_duration_seconds"] = 0
    elif case == "presentation_gate":
        data["official_eligibility"] = True
    else:
        metric = next(m for c in data["feedback_cases"] for f in c["frames"] for m in f["metrics"]
                      if m["status"] != "available")
        metric["value"] = 0
    bad = modified_bundle(contents, "presentation.json", json.dumps(data).encode())
    with pytest.raises(ValueError):
        validate_bundle(bad)


@pytest.mark.parametrize("case", ["Data", "outside", "nested_output"])
def test_source_scope_and_separate_output_required_before_writes(release, monkeypatch, case):
    workspace, source, _, _, _ = release
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject paths before reading/writing")
    monkeypatch.setattr(io, "open", forbidden)
    bad_source = workspace / "Data" if case == "Data" else workspace.parent / "outside"
    if case == "nested_output":
        bad_source = source
        target = source / "bad-release"
    else:
        target = workspace / "outputs" / "bad-release"
    with pytest.raises(ValueError):
        write_release(bad_source, target, workspace=workspace)


def test_stdlib_only_import_and_cli_dispatch(release, monkeypatch, capsys):
    done = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import gym_coach.reporting.offline_release; "
        "assert not any(m.split('.')[0] in {'numpy','scipy','cv2','mediapipe','torch'} for m in sys.modules)"],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    import gym_coach.reporting.offline_release as module
    _, _, _, _, manifest = release
    calls = []
    def fake(source, target):
        calls.append((source, target))
        return manifest
    monkeypatch.setattr(module, "write_release", fake)
    main(["--bundle", "outputs/source", "--output-dir", "outputs/new"])
    assert calls == [(Path("outputs/source"), Path("outputs/new"))]
    assert json.loads(capsys.readouterr().out) == manifest
