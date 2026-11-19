"""Xuat noi dung TOAN BO file .py va .yaml cua du an ra totalcode.txt.

Quet de quy tu thu muc goc, bo qua cac thu muc rac (.venv, __pycache__,
.git, backup, output, ...) va chinh file output.

Chay tai thu muc goc du an:
    python export_totalcode.py
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "totalcode.txt"

# Cac duoi file se xuat
INCLUDE_EXTENSIONS = {".py", ".yaml", ".yml"}

# Cac thu muc se BO QUA hoan toan (so sanh theo ten thu muc, khong phan biet
# hoa/thuong). Them/bot tuy nhu cau.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "backup",          # thu muc backup
    "output",          # thu muc ket qua xuat
    "data",            # du lieu video/pose cache (khong phai code)
    "test_posemodal",  # neu khong muon xuat cac thu muc test nay
    "test_posemodal_trainee",
    "New folder",
}

# Cac file cu the se bo qua (ten file, khong tinh duong dan)
EXCLUDE_FILES = {
    "export_totalcode.py",   # chinh script nay
    "totalcode.txt",         # file ket qua
}


def should_skip_dir(dir_name: str) -> bool:
    return dir_name.lower() in {d.lower() for d in EXCLUDE_DIRS}


def collect_source_files() -> list[Path]:
    """Quet de quy, tra ve danh sach file .py/.yaml/.yml da sap xep."""
    results: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        # Bo qua neu nam trong bat ky thu muc bi loai
        if any(should_skip_dir(part) for part in path.relative_to(PROJECT_ROOT).parts[:-1]):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix.lower() in INCLUDE_EXTENSIONS:
            results.append(path)
    # Sap xep theo duong dan tuong doi cho on dinh
    results.sort(key=lambda p: str(p.relative_to(PROJECT_ROOT)).lower())
    return results


def write_section(output_file, source_path: Path) -> bool:
    """Ghi mot file ma nguon vao bao cao; tra ve False neu file bi thieu."""
    relative_path = source_path.relative_to(PROJECT_ROOT).as_posix()
    absolute_path = source_path.resolve()
    separator = "=" * 100

    output_file.write(f"{separator}\n")
    output_file.write(f"TEN FILE: {source_path.name}\n")
    output_file.write(f"DUONG DAN TUONG DOI: {relative_path}\n")
    output_file.write(f"DUONG DAN TUYET DOI: {absolute_path}\n")
    output_file.write(f"{separator}\n")
    output_file.write("NOI DUNG CODE:\n\n")

    if not source_path.is_file():
        output_file.write("[FILE KHONG TON TAI]\n\n")
        return False

    try:
        content = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Mot so file yaml co the dung encoding khac -> thu latin-1
        content = source_path.read_text(encoding="latin-1")

    output_file.write(content)
    if not content.endswith("\n"):
        output_file.write("\n")
    output_file.write("\n\n")
    return True


def main() -> None:
    source_files = collect_source_files()
    print(f"Tim thay {len(source_files)} file .py/.yaml/.yml se xuat.")

    written = 0
    missing = []

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("BAO CAO TONG HOP MA NGUON DU AN\n")
        output_file.write(f"THU MUC GOC: {PROJECT_ROOT}\n")
        output_file.write(f"SO FILE DU KIEN: {len(source_files)}\n\n")

        for source_path in source_files:
            if write_section(output_file, source_path):
                written += 1
            else:
                missing.append(str(source_path.relative_to(PROJECT_ROOT)))

        output_file.write("=" * 100 + "\n")
        output_file.write(f"KET QUA: Da xuat {written}/{len(source_files)} file.\n")
        if missing:
            output_file.write("FILE CON THIEU:\n")
            output_file.write("\n".join(f"- {path}" for path in missing) + "\n")

    print(f"Da tao: {OUTPUT_PATH}")
    print(f"Da xuat: {written}/{len(source_files)} file")
    if missing:
        print("Can kiem tra cac file thieu:")
        for relative_path in missing:
            print(f"- {relative_path}")


if __name__ == "__main__":
    main()