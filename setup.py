import setuptools

setuptools.setup(
        name="deeptest",
        packages=setuptools.find_packages(exclude=['models', 'experiments']),
        version='0.0.1',
        python_requires='>=3.5',
        description='Code for the paper "Two-sample Testing Using Deep Learning"',
        )
