# How to upload a release to PyPi

## Generate distribution

Follow the instructions from [https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives](https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives)

Modify the project name (in pyproject.toml) and the version in python/version.py when testing against the test instance of PtyPy. Be aware that every upload has to be a unique combination of project name and version and this CANNOT be reversed.

## Upload distribution

Follow instructions from [https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives)

Make sure to have the access token saved in .pypirc

## Testing the installation from test PyPi

For installing the uploaded distribution, use

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple <project name>==<version

```