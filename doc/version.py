
# THIS FILE IS GENERATED FROM ptypy/setup.py
short_version='0.2.0'
version='0.2.0'
release=False

if not release:
    version += '.dev'
    import subprocess
    try:
        git_commit = subprocess.Popen(["git","log","-1","--pretty=oneline","--abbrev-commit"],stdout=subprocess.PIPE).communicate()[0].split()[0]
    except:
        pass
    else:
        version += git_commit.strip()

