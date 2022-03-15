import sys
import subprocess

def pip_install(package):
    print(f"installing (pip) {package} ... ", end="")
    out = subprocess.run(['pip3', 'install', '-q', package])
    if out.returncode:
        print(f"failed")
    else:
        print(f"success")

def apt_install(package):
    print(f"installing (apt) {package} ... ", end="")
    out = subprocess.run(['apt-get', 'install', '-y', '-q', package])
    if out.returncode:
        print(f"failed")
    else:
        print(f"success") 

def ampl_install(package):
    print(f"installing (ampl) {package} ... ", end="")    
    out = subprocess.run(['wget', '-N', '-q', f'https://ampl.com/dl/open/{package}/{package}-linux64.zip'])  
    if out.returncode:
        print(f"failed to download")
        return
    out = subprocess.run(['unzip', '-o', '-q', f'{package}-linux64'])
    if out.returncode:
        print(f"failed to unzip")
    else:
        print(f"success")  

if "google.colab" in sys.modules:

    # install from pipy
    pip_install('pyomo')
    pip_install('gurobipy')
    pip_install('cplex')
    pip_install('xpress')

    # install debian packages
    apt_install('glpk-utils')
    apt_install('coinor-cbc')

    # install ampl binaries
    ampl_install('ipopt')
    ampl_install('bonmin')
    ampl_install('couenne')
    ampl_install('gecode')
    ampl_install('jacop')
    