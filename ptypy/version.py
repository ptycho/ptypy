
short_version = '0.7.1'
version = '0.7.1'
release = True

if not release:
    version += '.dev'
    import subprocess
    try:
        git_commit = subprocess.Popen(["git","log","-1","--pretty=oneline","--abbrev-commit"],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL).communicate()[0].split()[0]
    except:
        pass
    else:
        version += git_commit.strip().decode()

