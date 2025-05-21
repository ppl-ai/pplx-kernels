import os
import subprocess
from pathlib import Path

from setuptools import Command, Extension, find_packages, setup
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext


def _get_torch_cmake_prefix_path() -> str:
    import torch

    return torch.utils.cmake_prefix_path


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exn:
            raise RuntimeError(
                f"CMake must be installed to build the following extensions: {', '.join(e.name for e in self.extensions)}"
            ) from exn
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: Extension) -> None:
        build_lib = Path(self.build_lib)
        build_lib.mkdir(parents=True, exist_ok=True)

        root_dir = Path(__file__).parent
        build_dir = root_dir / "build-cmake"
        build_dir.mkdir(parents=True, exist_ok=True)
        source_dir = root_dir / "csrc"

        subprocess.check_call(
            [
                "cmake",
                "-B",
                str(build_dir),
                "-S",
                str(source_dir),
                "-G",
                "Ninja",
                "-DCMAKE_PREFIX_PATH=" + _get_torch_cmake_prefix_path(),
                "-DTORCH_CUDA_ARCH_LIST=" + os.environ["TORCH_CUDA_ARCH_LIST"],
                "-WITH_TESTS=OFF",
            ]
        )
        subprocess.check_call(["ninja"], cwd=str(build_dir))


class CustomBuild(_build):
    def run(self) -> None:
        self.run_command("build_ext")
        super().run()


class CleanCommand(Command):
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        subprocess.run(
            [
                "rm",
                "-rf",
                "build",
                "build-cmake",
                "src/pplx_kernels.egg-info",
                "src/pplx_kernels/libpplx_kernels.so",
            ]
        )


extensions = [
    Extension(
        "pplx-kernels",
        sources=[],
    ),
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "pplx_kernels": ["libpplx_kernels.so", "py.typed"],
    },
    cmdclass={
        "build_ext": CMakeBuild,
        "build": CustomBuild,
        "clean": CleanCommand,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    zip_safe=False,
    ext_modules=extensions,
    include_package_data=True,
)
