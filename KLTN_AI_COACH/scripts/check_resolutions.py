import glob
import os
import cv2

video_paths = sorted(glob.glob("data/raw_videos/*.mp4"))
print(f"Kiểm tra {len(video_paths)} video...\n")

video_groups = {}
for path in video_paths:
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS), 2)  # Làm tròn 2 chữ số thập phân
    cap.release()

    # Nhóm theo cả (Độ phân giải, FPS)
    key = (width, height, fps)
    video_groups.setdefault(key, []).append(path)
    print(f"{path}  ->  {width}x{height}  (fps={fps})")

print("\n=== TÓM TẮT ===")
for (w, h, fps), paths in video_groups.items():
    print(f"Độ phân giải {w}x{h} | FPS {fps}: {len(paths)} video")

# Kiểm tra sự đồng nhất
unique_resolutions = { (w, h) for (w, h, fps) in video_groups.keys() }
unique_fps = { fps for (w, h, fps) in video_groups.keys() }

if len(unique_resolutions) > 1:
    print("\n⚠️ CẢNH BÁO: Video KHÔNG cùng độ phân giải — cần resize hoặc xử lý riêng!")
else:
    print("\n✓ Tất cả video CÙNG 1 độ phân giải.")

if len(unique_fps) > 1:
    print("⚠️ CẢNH BÁO: Video KHÔNG cùng FPS — có thể ảnh hưởng đến tính toán theo thời gian/tốc độ!")
else:
    print("✓ Tất cả video CÙNG FPS.")

# ==========================================
# ĐOẠN MỚI BỔ SUNG: LIỆT KÊ ĐƯỜNG DẪN TUYỆT ĐỐI CỦA 9 VIDEO
# ==========================================
print("\n=== DANH SÁCH ĐƯỜNG DẪN TUYỆT ĐỐI (TỐI ĐA 9 VIDEO) ===")
abs_paths = [os.path.abspath(p) for p in video_paths[:9]]

for i, abs_p in enumerate(abs_paths, 1):
    print(f"{i}. {abs_p}")

print("\nList dưới dạng Python list:")
print(abs_paths)