from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="ftnirsml",  
    version="0.0.1",  
    author="Michael Zakariaie, Daniel Woodrich",  
    author_email="michael.zakariaie@noaa.gov",  
    description="FTnirs ML codebase without web interface", 
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown", 
    url="https://github.com/noaa-afsc/ml-app-codebase",  
    packages=find_packages(),  # Automatically find all packages within the directory
    install_requires=parse_requirements('requirements.txt'), 
    extras_require={  # Optional dependencies for additional functionality
        "dev": ["pytest>=5.4.1", "flake8>=3.8.3"],
    },
    classifiers=[  # Classifiers that describe the project
        "Programming Language :: Python :: 3",
        "should decide this... example 'License :: OSI Approved :: MIT License'",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  
    include_package_data=True,  # Include files from MANIFEST.in
    zip_safe=False,  
)
