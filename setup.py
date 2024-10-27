# Responsible for creating my ML applocation as a package
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """List of python modules requirements for the project.

    Args:
        file_path (str): requirements.txt path.

    Returns:
        List[str]: A List of python modules required for the project.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace("\n", "") for requirement in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements
    

setup(
    name="mlproject",
    version='0.0.1',
    author='Divine',
    author_email='divinenwadigo06@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)