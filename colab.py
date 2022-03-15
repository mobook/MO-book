import sys
import subprocess

def pip_install(package):
    print(f"installing {package} ... ", end="")
    out = subprocess.run(['pip3', 'install', '-q', package])
    if out.returncode:
        print(f"failed")
    else:
        print(f"success")

if "google.colab" in sys.modules:
    pip_install('pyomo')
    pip_install('gurobipy')
    pip_install('cplex')
    pip_install('xpress')
    
        
        
#import shutil
#import sys
#import os.path
#import subprocess
#import google

#if "google.colab" in sys.modules:
#    subprocess.run(["pip", "install", "-q", "pyomo"])
 #   subprocess.run(["pip", "install", "-q", "gurobipy"])
        #!pip install -q gurobipy
        #!pip install -q cplex
        #!pip install -q xpress
        #!apt-get install -y -qq coinor-cbc
        #!wget -N -q "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"
        #!unzip -o -q ipopt-linux64
