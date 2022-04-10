import sys
import subprocess

def on_colab():
    return "google.colab" in sys.modules

def mplstyle():
    pass

def pip_install(package):
    print(f"installing {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
if on_colab():
    pip_install("pyomo")
    
