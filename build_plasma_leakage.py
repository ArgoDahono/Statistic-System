import subprocess
import shutil
import sys
import re

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build"
CMAKE_LISTS = PROJECT_ROOT / "CMakeLists.txt"
REQUIRED_FILES = ["CMakeLists.txt", "plasma_leakage.cpp", "plasma_leakage.hpp", "bindings.cpp"]

def check_environment():
    if shutil.which("cmake") is None:
        raise EnvironmentError("CMake tidak ditemukan. Pastikan CMake sudah terpasang dan tersedia di PATH.")

def verify_sources():
    missing = [name for name in REQUIRED_FILES if not (PROJECT_ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"File sumber hilang: {', '.join(missing)}")

def ensure_cmake_lists():
    text = CMAKE_LISTS.read_text(encoding="utf-8")
    new_text = text

    # Pastikan CMakeLists menggunakan file sumber yang ada
    new_text = re.sub(
        r"set\(CORE_SOURCES[\s\S]*?\)\n",
        "set(CORE_SOURCES\n    plasma_leakage.cpp\n)\n\n",
        new_text,
        flags=re.MULTILINE,
    )

    new_text = re.sub(
        r"set\(BINDING_SOURCES[\s\S]*?\)\n",
        "set(BINDING_SOURCES\n    bindings.cpp\n)\n\n",
        new_text,
        flags=re.MULTILINE,
    )

    lines = new_text.splitlines()
    seen = False
    filtered = []
    for line in lines:
        if line.strip() == "find_package(pybind11 REQUIRED)":
            if seen:
                continue
            seen = True
        filtered.append(line)
    new_text = "\n".join(filtered) + "\n"

    if new_text != text:
        CMAKE_LISTS.write_text(new_text, encoding="utf-8")
        print("CMakeLists.txt diperbarui agar menggunakan plasma_leakage.cpp dan bindings.cpp.")

def run_command(command, cwd):
    print(f"Menjalankan: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)

def configure_build():
    run_command(["cmake", "-S", str(PROJECT_ROOT), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"], PROJECT_ROOT)

def build_target(config="Release"):
    run_command(["cmake", "--build", str(BUILD_DIR), "--config", config], PROJECT_ROOT)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build project Plasma Leakage menggunakan CMake.")
    parser.add_argument("--clean", action="store_true", help="Hapus folder build sebelum membangun ulang.")
    parser.add_argument("--config", default="Release", help="Konfigurasi build CMake (default: Release).")
    args = parser.parse_args()

    try:
        check_environment()
        verify_sources()
        ensure_cmake_lists()

        if args.clean:
            print(f"Menghapus folder build: {BUILD_DIR}")
            shutil.rmtree(BUILD_DIR, ignore_errors=True)

        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        configure_build()
        build_target(config=args.config)
        print("Build selesai. Hasil akan berada di direktori build.")
    except Exception as exc:
        print(f"Build gagal: {exc}", file=sys.stderr)
        sys.exit(1)