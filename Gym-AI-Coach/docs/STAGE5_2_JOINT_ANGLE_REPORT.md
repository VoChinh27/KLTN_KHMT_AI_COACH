# Giai đoạn 5.2 — Primitive hình học góc khớp

Ngày đo: 2026-10-08, Asia/Saigon. **GĐ5.2 COMPLETE**; toàn **GĐ5 IN_PROGRESS**.
Implementation, unit tests và full suite đã chạy thành công. Đây là mathematical
primitive validation bằng synthetic input; chưa feature calculation hoặc pose/
biomechanical accuracy/form validation. Không dùng code/kết quả archive.

## Phạm vi và API thực tế

Thêm `src/gym_coach/geometry.py`, ngoài namespace features để import/call không
phụ thuộc registry I/O, video/pose/quality/preprocessing hoặc extractor.

```python
from gym_coach.geometry import Point2D, joint_angle

angle_degrees = joint_angle((1, 0), (0, 0), (0, 1))  # 90.0
```

- `Point2D = tuple[int | float, int | float] | list[int | float]`; runtime mỗi
  điểm phải có đúng hai coordinates. Không nhận pose/joint schema hoặc tự lấy x/y.
- `joint_angle(a: Point2D, b: Point2D, c: Point2D) -> float`: góc trong tại B,
  đơn vị **degrees**, miền `[0,180]`, không làm tròn.
- Chấp nhận int/float hữu hạn trong finite float range; bool/null/string/complex,
  NaN/±Inf, point sai kiểu/kích thước bị từ chối. Integer giữ nguyên đến subtraction
  để large translated integers không bị ép float thành điểm trùng giả.
- Không rotate/flip/normalize input, không áp dụng image-up/y-down. Tọa độ caller
  đã có sẵn; angle chỉ là phép toán Euclidean giữa hai vector 2D đó.
- Không sửa input, raw pose, QualityReport hoặc GĐ4 output; không đọc dataset.

Định nghĩa toán học, với `u=A−B`, `v=C−B`, `d=||u||·||v||`:

```text
cos(theta) = dot(u,v) / d
sin(theta) = abs(u_x*v_y - u_y*v_x) / d
theta_deg  = degrees(atan2(sin(theta), clamp(cos(theta), -1, 1)))
```

Code tính bằng các direction đã chia **hệ số dương riêng** để giữ components
bounded, tránh overflow/underflow khi nhân độ dài. Finite float subtraction bị
overflow dùng half-coordinates nội bộ; integer differences vượt float range
được halve trước division. Đây là numerical rescaling giữ hướng, không thay đổi
dữ liệu caller hoặc normalization metadata. Dùng `math.hypot`, `math.fsum` và
`math.atan2`; denominator phải finite và >0 trước division/cosine clamp.
atan2 giữ góc nhỏ gần 0°/180° tốt hơn acos khi cosine làm tròn tới endpoint.

Chỉ vector **zero chính xác** bị từ chối; không đặt epsilon hoặc near-zero
measurement threshold. A=B hoặc C=B không có angle; A=C khác B hợp lệ `0°`.
All-input validation chạy trước geometry, vì thế malformed C vẫn là input error
khi A=B. Primitive không tạo FeatureObservation hoặc unavailable feature result.

## Structured error policy

Mọi failure dùng `GymError.flag`, severity `error`, stage `features`, source tier
`S3` engineering. Không phải technical_error của người tập. Flag có primitive,
detailed reason và point/coordinate/vector context khi có; không measured value,
NaN/Inf hoặc video/joint identity giả trong payload. JSON roundtrip dùng schema
StructuredFlag hiện có; location trống vì API không có input identity đó.

| Flag code | Category | Reason chi tiết |
|---|---|---|
| invalid_geometry_input | input_error | point_type, point_dimension, coordinate_type, coordinate_not_finite, coordinate_not_representable |
| degenerate_geometry | measurement_error | zero_length_vector, vector BA hoặc BC |
| geometry_numerical_error | measurement_error | invalid_denominator, non_finite_ratio, invalid_angle_result |

Numeric guards là backstop; invalid denominator đã được fault-injection test,
không trình bày nó như failure quan sát từ video. Không catch-all che lỗi lập trình.
Common FlagCode chỉ thêm ba codes; không sửa error/flag semantics của GĐ0–5.1.

## Versions, dependencies và môi trường

- CPython **3.11.9**, Windows x64 (`Windows-10-10.0.26200-SP0`), pytest **9.1.1**.
- Package **ai-coach-gym 0.5.1**, compatible addition; offline editable reinstall
  thành công. Không thêm dependency, model hoặc framework; pyproject không đổi.
- Arithmetic chỉ thư viện chuẩn math/typing. Structured errors dùng shared
  contracts/flags với Pydantic **2.13.5** đã có; không có standalone error schema mới.
- Venv full suite còn MediaPipe **0.10.14**, NumPy **2.4.6**, SciPy **1.17.1**,
  OpenCV contrib **5.0.0.93**, protobuf **4.25.9**, setuptools **84.0.0**.
  Primitive không import các backend này. Legacy opencv-python 5.0.0.93 vẫn có.
- `pip check`: exit 0, `No broken requirements found.` Chưa clean-environment
  verification hoặc transitive lock; không tuyên bố compatibility môi trường sạch.
- Foundation/config/pose/quality/preprocessing/feature schemas không đổi.
  Feature/registry schema **0.1.0**, registry **gym-feature-registry-v1**, hash vẫn
  `bf0917f6b6ac24d9fd8f3316f4b662a9dcce618bea496bb2e0114643d8051d2a`.
  Registry v1 metadata chưa feature-specific calculator; standalone primitive
  không thay calculation_status=not_implemented hoặc value=null của GĐ5.1.

Workspace **không có Git repository**, không tạo commit/diff giả. Source tree
SHA-256 (src/gym_coach, configs, tests, pyproject; không pycache/docs/outputs):
`d231f9c71d976a781a5dedb96d9935e5ac48f0c9e996a443c9d6e950825e6e30`.
Full per-file hashes/environment/commands:
[stage5_2_verification.json](artifacts/stage5_2_verification.json).

## Commands và test evidence

PowerShell tại project root, interpreter `.venv` hiện hữu:

```powershell
& .venv/Scripts/python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
& .venv/Scripts/python.exe -m pip check
& .venv/Scripts/python.exe -m pytest -q tests/test_geometry.py --junitxml=outputs/stage5_2/verification-a5b0f36ed83c4854b1a9ef6e45404ea2/new-tests.xml
& .venv/Scripts/python.exe -m pytest -q tests/test_geometry.py --junitxml=outputs/stage5_2/verification-a5b0f36ed83c4854b1a9ef6e45404ea2/new-tests-final.xml
& .venv/Scripts/python.exe -m pytest -q --junitxml=outputs/stage5_2/verification-a5b0f36ed83c4854b1a9ef6e45404ea2/full-suite.xml
```

| Run | Passed | Failed / errors / skipped | Warnings | Terminal time |
|---|---:|---|---:|---:|
| Initial primitive tests | 88 | 0 / 0 / 0 | 0 | 0.77 s |
| Final primitive tests (integer safety + denominator guard bổ sung) | **94** | **0 / 0 / 0** | **0** | **1.00 s** |
| Final full suite GĐ0–5.2 | **328** | **0 / 0 / 0** | **4** | **36.58 s** |

Tất cả commands exit 0; JUnit counts được parse đối chiếu. 4 warnings là
`UserWarning: SymbolDatabase.GetPrototype() is deprecated` từ
google/protobuf/symbol_database.py:55, trong hai pose integrations GĐ2 và hai
fresh preprocessing integrations GĐ4; không suppress warning hoặc cài model mới.
Chúng kiểm tra pipeline cũ trên hai development videos, không chạy feature angle
extractor. Không có real-data angle/accuracy experiment trong phạm vi GĐ5.2.

Test coverage: 90/180/0/45/60/135°, endpoint symmetry, translation, positive
uniform scale (`0.125`, `3`, `1e-300`, `1e300`), zero BA/BC/both, coincident A=C,
invalid input ở cả A/B/C, subnormal `5e-324`, huge coordinates/subtraction,
near-collinear vectors, large integer translation/differences, input immutability,
structured flag JSON/category/stage/S3/reason và denominator backstop. Fresh
isolated subprocess chặn NumPy/SciPy/OpenCV/MediaPipe/Torch **và tất cả domain
pipeline/feature-registry imports**, kiểm tra import/call/success/error JSON.

Tolerance synthetic **absolute 1e-10 degrees, relative=0**, không UI rounding.
Exact endpoint/zero cases cũng được assert trực tiếp; tiny nonzero angle có
positive-output check. Tolerance là assertion arithmetic, không pose accuracy
bound, form threshold hoặc engineering gate cho người tập.

[Final unit JUnit](../outputs/stage5_2/verification-a5b0f36ed83c4854b1a9ef6e45404ea2/new-tests-final.xml),
[full-suite JUnit](../outputs/stage5_2/verification-a5b0f36ed83c4854b1a9ef6e45404ea2/full-suite.xml).
Mỗi run dùng artifact path mới, initial XML được giữ. Before hash manifest
167 files xác nhận 160 unchanged; chỉ 7 existing source/docs files được cập nhật
đúng phạm vi. Toàn bộ old outputs, configs/tests/source GĐ0–5.1 (ngoài additive
common flags và package version), historical GĐ4/GĐ5.1 reports giữ nguyên.
Không mở/modify archive, dataset hoặc xóa file.

## Files và đối chiếu DoD

Mới: geometry.py, tests/test_geometry.py, report này,
docs/artifacts/stage5_2_verification.json; verification output directory mới.
Cập nhật: flags.py, _version.py, README, IMPLEMENTATION_PLAN, PROJECT_STATE,
TASKS, SCHEMA_CONVENTIONS. Không sửa registry/policy/FeatureObservation schema.

| Tiêu chí GĐ5.2 | Bằng chứng | Kết quả |
|---|---|---|
| Pure typed three-point angle, degree/range contract | Point2D/joint_angle, known-angle tests | PASS |
| Input malformed/missing/NaN/±Inf rejected có context | 54 parametrized input cases + validation precedence | PASS |
| Zero vectors và coincident endpoints phân biệt | BA/BC/both errors; A=C valid 0° | PASS |
| Finite denominator trước cosine clamp; numeric stability | bounded arithmetic, subnormal/huge/near-parallel/int/guard tests | PASS |
| Symmetry, translation, positive scale invariant | Synthetic tests với tolerance công bố | PASS |
| Structured category/stage/S3/reason, không giả measurement | Error tests + flag JSON roundtrip | PASS |
| Public API không kéo backend/registry/pipeline | Fresh isolated blocked-import subprocess | PASS |
| GĐ0–5.1 behavior/policies/raw outputs preserved | 328 full-suite pass + before hash audit | PASS |
| Docs/environment/commands/results/limitations cập nhật | Report này, evidence và 5 current docs | PASS |
| Không numeric feature/extractor/sequence hoặc downstream | Standalone primitive; feature schema/registry/policy unchanged | PASS |

## Giới hạn và bước nhỏ nhất tiếp theo

- Floating-point rounding vẫn tồn tại; translation/scale invariance chỉ khi phép
  biến đổi giữ hữu hạn và không làm các float endpoints trùng do mất precision.
  Không arbitrary-precision angle guarantee hoặc khôi phục độ chính xác đã mất
  trước khi caller truyền tọa độ. Int ngoài finite float range bị input error.
- Primitive không biết isotropic units, provenance, visibility, viewpoint hoặc
  scale fallback. Tọa độ raw x/y normalize theo hai kích thước ảnh khác nhau có
  thể làm méo angle; caller chịu trách nhiệm hệ tọa độ và policy. Không ngầm sửa.
- Synthetic math pass không xác nhận pose accuracy, độ chính xác feature theo
  camera hoặc biomechanical joint angle; MediaPipe 2D/perspective/occlusion và
  expert ground truth limitations vẫn nguyên. UCF101 chỉ nhãn action.
- Chưa exercise-specific feature calculation, numeric FeatureObservation,
  FeatureSequence, registry adapter, rep/phase, rules/score/feedback. Measurement
  validation statuses vẫn unvalidated/blocked, assessment bị khóa.

Bước nhỏ nhất **GĐ5.3**: thiết kế numeric observation schema có version mới và
adapter cho **một** feature `joint_angle_abc`, kiểm tra readiness/provenance/
quality/viewpoint trước gọi primitive, chuyển measurement failures thành
unavailable với reason và identity thật. Sau synthetic gate mới mở rộng/integrate
theo DoD GĐ5.3. Chưa bắt đầu bước đó và không đánh dấu toàn GĐ5 COMPLETE.
