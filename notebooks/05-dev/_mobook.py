# EDIT NOTE: This notebook is overwritten from github. 
#            Iimmediately push this file to github after editing.

"""
Collection of helper functions for use with MO-book notebooks.

Functions:

    mobook.svg(): set svg as default matplotlib format for the notebook
    mobook.on_colab(): return True if running on Google Colab
    
    mobook.setup_pyomo()
    mobook.setup_solvers()
    
    mobook.setup_glpk()
    mobook.setup_cbc()

"""


import sys
import os
import subprocess
import matplotlib as mpl
from IPython.display import set_matplotlib_formats
import matplotlib_inline.backend_inline

def svg():
    """reset matplotlib defaults to use SVG"""
    # see: https://www.alanshawn.com/tech/2021/03/27/matplotlib-latex-style.html
    
    # set backend to svg plotting
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
    
    #set_matplotlib_formats('svg', 'pdf')
    
    # embed fonts in svg files
    mpl.rcParams['svg.fonttype'] = 'path'
    mpl.rcParams['pdf.fonttype'] = 42    
    mpl.rcParams['font.family'] = 'STIXgeneral'
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.titlesize'] = 18
    
# default installations for Google Colab
def on_colab():
    return "google.colab" in sys.modules

def pip_install(package):
    print(f"installing {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
def ampl_install(package):
    print(f"installing {package}")
    url = f"https://ampl.com/dl/open/{package}/{package}-linux64.zip"
    os.system("curl -sO " + url)
    os.system("unzip -o -p  " + f"{package}-linux64.zip") 
    
def setup_glpk():
    if on_colab():
        print("installing glpk")
        os.system("apt-get install -y -qq glpk-utils")
        
def setup_cbc():
    if on_colab():
        print("installing cbc")
        os.system("apt-get install -y -qq coinor-cbc")
        
def setup_gurobi():
    if on_colab():
        pip_install("gurobipy")
        
def setup_ipopt():
    if on_colab():
        ampl_install("ipopt")
        
def setup_cplex():
    if on_colab():
        pip_install("cplex")
        
def setup_solvers():
    setup_glpk()
    setup_cbc()
    setup_ipopt()
        
def setup_pyomo():
    if on_colab():
        pip_install("pyomo")
        
        


    
