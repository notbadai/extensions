from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="notbadai_tools_chat",
    version="0.1.4",
    author="NotBadAI Team",
    author_email="contact@notbad.ai",
    description="An intelligent programming assistant powered by AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/notbadai/extensions",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "notbadai_ide",
        "labml",
        "requests",
        "openai",
    ],
include_package_data=True,
    package_data={"notbadai_tools_chat": ["*.md", "**/*.md"]},
)
