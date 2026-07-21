"""Manual build entry point for the optional SRI-style suffix tree."""

from pathlib import Path

from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:  # pragma: no cover - build-time dependency
    raise RuntimeError("Install pybind11 before building this extension.") from exc


ROOT = Path(__file__).parent

setup(
    name="sglang-sri-suffix-tree",
    version="0.1.0",
    ext_modules=[
        Extension(
            "_sglang_sri_suffix_tree",
            [str(ROOT / "suffix_tree.cpp"), str(ROOT / "pybind.cpp")],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
)
