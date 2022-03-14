import subprocess
import sys

def install(package):
    subprocess.run("pip", "install", package)
    
    
if __name__ == "__main__":
    install("pyomo")
