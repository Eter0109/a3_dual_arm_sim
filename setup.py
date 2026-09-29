"""Bundle runtime resources without duplicating them in the source tree."""

from pathlib import Path
from shutil import copytree

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithResources(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / "a3_dual_arm_sim" / "_resources"
        for directory in ("assets", "configs"):
            copytree(source / directory, destination / directory, dirs_exist_ok=True)


setup(cmdclass={"build_py": BuildWithResources})
