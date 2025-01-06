from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def get_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the README for the long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="timelapse-bot",
    version="1.0.0",
    description="A Discord bot for managing multiple cameras with motion detection and timelapse capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/timelapse-bot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "timelapse-bot=timelapse.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Video",
        "Topic :: Communications :: Chat",
        "Framework :: AsyncIO",
        "Framework :: Discord",
    ],
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/yourusername/timelapse-bot/issues",
        "Source": "https://github.com/yourusername/timelapse-bot",
    },
)