from setuptools import setup, find_packages

setup(
    name="ftnirsml",  
    version="0.0.1",  
    author="Daniel Woodrich, Michael Zakariaie",  
    author_email="daniel.woodrich@noaa.gov",  
    description="FTnirs ML codebase", 
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown", 
    url="https://github.com/noaa-afsc/ml-app-codebase",  
    packages=find_packages(),  # Automatically find all packages within the directory
    install_requires=[
    "pandas",
    "numpy",
    "tensorflow > 2",
    "keras_tuner",
    "seaborn",
    "pyCompare",
    "scikit-learn",
    "matplotlib",
    "scipy",
    "PyWavelets",
    "shap"
    ],
    classifiers=[  # Classifiers that describe the project
        "Programming Language :: Python :: 3",
        "License :: MIT License'",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  
    include_package_data=True,  # Include files from MANIFEST.in
    zip_safe=False,  
)
