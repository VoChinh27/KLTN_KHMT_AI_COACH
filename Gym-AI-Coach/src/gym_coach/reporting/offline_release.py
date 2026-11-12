"""Byte-preserving ZIP release of a saved synthetic handoff; stdlib only.

No renderer, producer, source selection, noise audit, inference, or network.
python -m gym_coach.reporting.offline_release --bundle <saved> --output-dir <new>
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import unquote, urlsplit
import zipfile

RELEASE_VERSION = "gym-squat-offline-release-v1"
ZIP_NAME = "BodyWeightSquats-offline-demo.zip"
BLOCKED = "BLOCKED_NEEDS_DOCUMENTED_SIDE_VIEW_INPUT"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


def _name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or PurePosixPath(name).is_absolute()
            or any(part.lower() == "data" for part in name.split("/"))):
        raise ValueError(f"Unsafe bundle member: {name!r}")
    return name


def _reparse(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _under_outputs(path, workspace):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    if ".." in candidate.parts or any(p.lower() == "data" for p in candidate.parts):
        raise ValueError("Only paths beneath workspace outputs/ are allowed")
    resolved = candidate.resolve()
    outputs = (workspace / "outputs").resolve()
    if resolved == outputs or not resolved.is_relative_to(outputs):
        raise ValueError("Only paths beneath workspace outputs/ are allowed")
    for part in (candidate, *candidate.parents):
        if part == workspace:
            break
        if part.exists() and _reparse(part):
            raise ValueError("Symlink/junction paths are not release inputs or outputs")
    return resolved


def _tree(bundle):
    files = {}
    def visit(directory):
        for path in sorted(directory.iterdir()):
            if _reparse(path):
                raise ValueError("Bundle cannot contain symlinks or junctions")
            if path.is_dir():
                visit(path)
            elif path.is_file():
                name = _name(path.relative_to(bundle).as_posix())
                if name.casefold() in {key.casefold() for key in files}:
                    raise ValueError("Case-colliding bundle members")
                files[name] = path.read_bytes()
            else:
                raise ValueError("Bundle contains a non-regular file")
    visit(bundle)
    return files


def inventory(files):
    return {name: {"sha256": _sha(raw), "size_bytes": len(raw)}
            for name, raw in sorted(files.items())}


class _HTML(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attribute in ("href", "src"):
            if attribute in attrs:
                self.links.append(attrs[attribute])


def validate_bundle(files):
    """Check saved bytes/links/gates only; never follow original source paths."""
    required = {"START_HERE.html", "index.html", "HUONG_DAN_MO_DEMO.txt",
                "manifest.json", "presentation.json", "noise-review.json",
                "selection.json", "upstream-manifest.json"}
    if not required <= files.keys():
        raise ValueError("Missing required handoff files")
    manifest = _json(files["manifest.json"])
    if (manifest.get("handoff_version") != "gym-squat-demo-handoff-v1"
            or manifest.get("synthetic_only") is not True
            or manifest.get("official_eligibility") is not False
            or manifest.get("assessment_enabled") is not False
            or manifest.get("validation_status") != "NOT_VALIDATED"
            or manifest.get("g57") != BLOCKED or manifest.get("g6") != "IN_PROGRESS"
            or manifest.get("browser_qa") != "NOT_RUN"):
        raise ValueError("Expected synthetic handoff version with disabled official gates")
    expected = manifest["files"]
    if set(files) != set(expected) | {"manifest.json"}:
        raise ValueError("Bundle file inventory differs from handoff manifest")
    for name, digest in expected.items():
        _name(name)
        if not re.fullmatch(r"[a-f0-9]{64}", digest) or _sha(files[name]) != digest:
            raise ValueError(f"Handoff hash mismatch: {name}")
    data = _json(files["presentation.json"])
    if (data.get("presentation_version") != "gym-squat-showcase-v1"
            or data.get("exercise") != "BodyWeightSquats"
            or data.get("experimental_label") != "EXPERIMENTAL / NOT VALIDATED"
            or data.get("official_eligibility") is not False
            or data.get("assessment_enabled") is not False
            or data.get("official_gate") != BLOCKED or data.get("g6_status") != "IN_PROGRESS"):
        raise ValueError("Presentation gates/version must remain disabled and experimental")
    for item in (manifest, data):
        linkage = item.get("linkage", {})
        if (linkage.get("mode") != "independent_panels"
                or linkage.get("synchronized") is not False
                or linkage.get("temporal_overlay_applied_to_skeleton") is not False):
            raise ValueError("Panel linkage must remain independent")
    if data["source_refs"] != manifest["source_artifacts"]:
        raise ValueError("Presentation/source reference mismatch")
    roles = set()
    for ref in data["source_refs"]:
        name = _name(ref["snapshot_path"])
        if ref["role"] in roles or name not in files or _sha(files[name]) != ref["sha256"]:
            raise ValueError("Missing/duplicate/mismatched source sidecar")
        roles.add(ref["role"])
    if {name for name in files if name.startswith("sources/")} != {
            ref["snapshot_path"] for ref in data["source_refs"]}:
        raise ValueError("Source snapshot inventory mismatch")
    null_metrics = 0
    for case in data["feedback_cases"]:
        if case["legacy_phase_displayed"] is not False:
            raise ValueError("Legacy phase must remain hidden")
        for frame in case["frames"]:
            for metric in frame["metrics"]:
                if metric["status"] != "available":
                    if metric["value"] is not None or not metric["reasons"]:
                        raise ValueError("Unavailable metric must preserve null and reasons")
                    null_metrics += 1
    for case in data["point_cases"]:
        if (case["accepted_point_count"] != 0 or case["motion_status"] != "unavailable"
                or any(case[key] is not None for key in ("arrival", "departure", "pause_duration_seconds"))):
            raise ValueError("Point/phase/bounds/pause must remain unavailable")
        for check in case["checks"]:
            if (check["status"] != "unavailable" or not check["reasons"]
                    or any(check[key] is not None for key in
                           ("point_turn_event", "arrival", "departure", "pause_duration_seconds"))):
                raise ValueError("Point check must preserve abstention/null/reasons")
    pages = {name: _HTML(raw.decode("utf-8")) for name, raw in files.items() if name.endswith(".html")}
    link_count = 0
    for name, page in pages.items():
        for href in page.links:
            target = urlsplit(href)
            if target.scheme or target.netloc or target.query:
                raise ValueError("Only local file/fragment links are permitted")
            linked = _name((PurePosixPath(name).parent / unquote(target.path)).as_posix()) if target.path else name
            if linked not in files:
                raise ValueError(f"Broken local link: {name} -> {href}")
            if target.fragment and (linked not in pages or unquote(target.fragment) not in pages[linked].ids):
                raise ValueError(f"Broken local fragment: {href}")
            link_count += 1
    markup = files["index.html"].decode("utf-8")
    embedded = re.findall(r'<script id="showcase-data" type="application/json">(.*?)</script>', markup, re.S)
    if (len(embedded) != 1 or _json(embedded[0]) != data
            or any(char in embedded[0] for char in "<>&\u2028\u2029")
            or "connect-src 'none'" not in markup or "fetch(" in markup or "innerHTML" in markup):
        raise ValueError("Embedded JSON/escaping/offline policy mismatch")
    return dict(status="PASSED", file_count=len(files), local_links_checked=link_count,
                snapshot_links_checked=len(roles), html_pages_checked=len(pages),
                unavailable_metrics_checked=null_metrics, point_cases_checked=len(data["point_cases"]),
                embedded_json_equal=True, disabled_gates_preserved=True, independent_panels=True,
                browser_qa="NOT_RUN", validation_status="NOT_VALIDATED",
                official_eligibility=False, assessment_enabled=False, g57=BLOCKED, g6="IN_PROGRESS")


def verify_archive(archive, expected, extraction_dir):
    """Extract only expected safe regular files into a previously absent directory."""
    extraction = Path(extraction_dir)
    if extraction.exists():
        raise FileExistsError(f"Refusing existing extraction directory: {extraction}")
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        names = [_name(info.filename) for info in members]
        if len({name.casefold() for name in names}) != len(names) or set(names) != set(expected):
            raise ValueError("ZIP has duplicate, missing, or unexpected members")
        for info in members:
            mode = info.external_attr >> 16
            if (info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or info.flag_bits & 1
                    or info.file_size != expected[info.filename]["size_bytes"]):
                raise ValueError("ZIP member type/encryption/size mismatch")
        extraction.mkdir(parents=True, exist_ok=False)
        for info in members:
            raw = zipped.read(info)  # ZIP CRC checked by stdlib as well as SHA-256 below.
            if _sha(raw) != expected[info.filename]["sha256"]:
                raise ValueError(f"ZIP member SHA-256 mismatch: {info.filename}")
            path = extraction / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
    files = _tree(extraction)
    if inventory(files) != expected:
        raise ValueError("Extracted file inventory/hash/size mismatch")
    result = validate_bundle(files)
    result.update(extracted_hashes_checked=len(files), zip_crc_checked=True,
                  total_uncompressed_bytes=sum(len(raw) for raw in files.values()))
    return result


def write_release(bundle_dir, output_dir, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).resolve()
    bundle = _under_outputs(bundle_dir, workspace)
    output = _under_outputs(output_dir, workspace)
    if output.exists():
        raise FileExistsError(f"Refusing existing release directory: {output}")
    if output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Source and release paths must be separate")
    files = _tree(bundle)
    source_checks = validate_bundle(files)
    expected = inventory(files)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / ZIP_NAME
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, raw in sorted(files.items()):
            zipped.writestr(name, raw)
    verified = verify_archive(archive, expected, output / "extracted")
    if inventory(_tree(bundle)) != expected:
        raise ValueError("Source bundle changed during packaging")
    def write_json(name, value):
        raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        with (output / name).open("xb") as stream:
            stream.write(raw)
        return _sha(raw)
    receipt_hash = write_json("verification.json", dict(source=source_checks, extracted=verified,
        source_unchanged=True, browser_qa="NOT_RUN", detector_run=False, inference_run=False,
        noise_audit_repeated=False))
    zip_hash = _sha(archive.read_bytes())
    result = dict(release_version=RELEASE_VERSION, packaging_status="PASSED_OFFLINE_CHECKS",
        validation_status="NOT_VALIDATED", synthetic_only=True, official_eligibility=False,
        assessment_enabled=False, g57=BLOCKED, g6="IN_PROGRESS", browser_qa="NOT_RUN",
        source=dict(path=bundle.relative_to(workspace).as_posix(),
                    manifest_sha256=expected["manifest.json"]["sha256"], unchanged=True),
        file_count=len(files), total_uncompressed_bytes=sum(len(raw) for raw in files.values()),
        files=expected, zip=dict(path=ZIP_NAME, sha256=zip_hash, size_bytes=archive.stat().st_size,
                                file_count=len(files), start_file="START_HERE.html"),
        verification=dict(path="verification.json", sha256=receipt_hash),
        trial_extraction="extracted", manifest_scope="Every source file including manifest.json; ZIP contains only source bytes")
    write_json("release-manifest.json", result)
    sums = [f"{zip_hash}  {ZIP_NAME}"]
    sums.extend(f"{info['sha256']}  extracted/{name}" for name, info in expected.items())
    with (output / "SHA256SUMS.txt").open("xb") as stream:
        stream.write(("\n".join(sums) + "\n").encode("utf-8"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = write_release(args.bundle, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
