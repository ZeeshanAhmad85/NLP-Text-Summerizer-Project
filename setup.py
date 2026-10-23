import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "1.0.0"

REPO_NAME = "NLP-Text-Summerizer-Project"
AUTHOR_USER_NAME = "ZeeshanAhmad85"
SRC_REPO = "TextSummarizer"
AUTHOR_EMAIL = "zeeshan6842895@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A modular Python package for NLP abstractive text summarization using Transformer architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        "Source Code": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.28.0",
        "datasets>=2.10.0",
        "torch>=1.13.0",
        "rouge_score>=0.1.2",
        "evaluate>=0.4.0",
        "pandas>=1.5.0",
        "PyYAML>=6.0",
        "python-box>=6.0.2",
        "ensure>=1.0.2",
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
        "pydantic>=1.10.0",
        "tqdm>=4.64.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ]
)
