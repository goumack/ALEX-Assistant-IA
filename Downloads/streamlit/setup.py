from setuptools import setup, find_packages

setup(
    name='agriculture_ai_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'joblib'
    ],
    author='Accel-2024',
    description='Projet AI pour la prédiction agricole',
    python_requires='>=3.6',
)
