import sys
import subprocess
import matplotlib as mpl


def mplstyle():
    mpl.rcParams['font.family'] = 'STIXgeneral'
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['pdf.fonttype'] = 42

def pip_install(package):
    print(f"installing {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
def apt_install(package):
    print(f"installing {package} via apt")
    subprocess.check_call([sys.executable, "-m", "apt-get", "install", "-y", "-qq", package])
   
# default installations for Google Colab
def on_colab():
    return "google.colab" in sys.modules

if on_colab():
    pip_install("pyomo")
    !apt-get install -y -qq glpk-utils
    
