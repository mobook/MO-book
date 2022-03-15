import shutil
import sys
import os.path
import subprocess

if not shutil.which("pyomo"):
    if "google.colab" in sys.modules:
        subprocess.run(["pip", "install", "-q", "pyomo"])
        #!pip install -q gurobipy
        #!pip install -q cplex
        #!pip install -q xpress
        #!apt-get install -y -qq coinor-cbc
        #!wget -N -q "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"
        #!unzip -o -q ipopt-linux64
