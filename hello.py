import subprocess
import sys

def pip_install(package):
    subprocess.run(["pip", "-qq", "install", package])
    
if __name__ == "__main__":
    pip_install("pyomo")
    pip_install("gurobipy")
    pip_install("cplex")
    pip_install("xpress")
    subprocess(["wget", "-N", "-q", "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"])
    subprocess(["unzip", "-o", "-q", "ipopt-linux64"])
    subprocess(["apt-get", "install", "-y", "-q", "coinor-cbc"])
    
    
