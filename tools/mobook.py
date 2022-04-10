import sys
import os
import subprocess
import matplotlib as mpl


def mplstyle():
    mpl.rcParams['font.family'] = 'STIXgeneral'
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['pdf.fonttype'] = 42

# default installations for Google Colab
def on_colab():
    return "google.colab" in sys.modules

def pip_install(package):
    print(f"installing {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
def ampl_install(package):
    print(f"installing {package}")
    url = f"https://ampl.cvom/dl/open/{package}/{package}-linux64.zip"
    subprocess.check_call([sys.executable, "-m", "curl", "-sO", url])
    subprocess.check_call([sys.executable, "-m", "unzip", "-o", "-p", f"{package}-linux64.zip"])
    
def install_glpk():
    if on_colab():
        print("installing glpk")
        os.system("apt-get install -y -qq glpk-utils")
        
def install_cbc():
    if on_colab():
        print("installing cbc")
        os.system("apt-get install -y -qq coinor-cbc")
        
def install_gurobi():
    if on_colab():
        pip_install("gurobipy")
        
def install_cplex():
    if on_colab():
        pip_install("cplex")
        

if on_colab():
    pip_install("pyomo")

    