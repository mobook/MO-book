import subprocess
import sys

def pip_install(package):
    subprocess.run(["pip", "-qq", "install", package])
    
if __name__ == "__main__":
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    print("starting install")
    for module in sys.modules:
        print(module)
    if IN_COLAB:
        print("google colab detected")
        pip_install("pyomo")
        pip_install("gurobipy")
        pip_install("cplex")
        pip_install("xpress")
        subprocess.run(["apt-get", "install", "-y", "-q", "coinor-cbc"])
        subprocess.run(["wget", "-N", "-q", "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"])
        subprocess.run(["unzip", "-o", "-q", "ipopt-linux64"])
