import shutil
import sys
import os.path

if not shutil.which("pyomo"):
    if "google.colab" in sys.modules:
        !pip install -q pyomo
        !pip install -q gurobipy
        !pip install -q cplex
        !pip install -q xpress
        !apt-get install -y -qq coinor-cbc
        !wget -N -q "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"
        !unzip -o -q ipopt-linux64
