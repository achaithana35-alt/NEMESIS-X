"""Setup script for Real-Time Video Anomaly Detection."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="video-anomaly-detection",
    version="1.0.0",
    author="Pari2003",
    author_email="maitraiyeebhatt@gmail.com",
    description="Real-time multi-object tracking and video anomaly detection using YOLOv8, ByteTrack, and VideoMAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pari2003/Real-Time-Video-Anomaly-Detection",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "vad-detect=src.pipeline:main",
            "vad-benchmark=optimization.benchmark:main",
            "vad-evaluate=evaluation.evaluate_ucf_crime:main",
        ],
    },
)
