import shutil
import sys
import os.path
import os
import requests
import urllib

import subprocess

def _check_available(executable_name):
    """Return True if executable_name is found in the search path."""
    return (shutil.which(executable_name) is not None) or os.path.isfile(executable_name)

def package_available(package_name):
    """Return True if package_name is installed."""
    if package_name == "glpk":
        return _check_available("glpsol")        
    else:
        return _check_available(package_name)

def on_colab(): 
    """Return True if running on Google Colab."""
    return "google.colab" in sys.modules

def command_with_output(command):
    print(subprocess.getoutput(command))
    
def install_pyomo():
    if package_available("pyomo"):
        print("Pyomo found! No need to install.")
    else:
        print("Installing pyomo via pip...")
        os.system("pip install -q idaes_pse")
        assert package_available("pyomo"), "Pyomo was not successfully installed."

    command_with_output("pyomo --version")

def install_idaes():
    if package_available("idaes"):
        print("IDAES found! No need to install.")
    else:
        print("Installing idaes via pip...")
        os.system("pip install -q idaes_pse")
        assert package_available("idaes"), "Ideas installation was not successful."
        
    command_with_output("idaes --version")

def install_ipopt():
    if package_available("ipopt"):
        print("Ipopt found! No need to install.")
    else:
        if on_colab():
            # Install idaes solvers
            print("Running idaes get-extensions to install Ipopt and k_aug...")
            os.system("idaes get-extensions")

            # Add symbolic link for idaes solvers
            os.system("ln -s /root/.idaes/bin/ipopt ipopt")
            os.system("ln -s /root/.idaes/bin/k_aug k_aug")          
            
        # Check again if Ipopt is available
        if not package_available("ipopt"):
            if on_colab():
                print("Installing ipopt via zip file...")
                os.system('wget -N -q "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"')
                os.system('!unzip -o -q ipopt-linux64')
            # Otherwise, try conda
            else:
                try:
                    print("Installing Ipopt via conda...")
                    os.system('conda install -c conda-forge ipopt')
                except:
                    pass

        assert package_available("ipopt"), "ipopt installation was not successful."           
        
    command_with_output("./ipopt --version")
    if package_available("k_aug"):
        command_with_output("./k_aug --version")

def install_glpk():
    if package_available("glpk"):
        print("GLPK found! No need to install.")
    else:
        if on_colab():
            print("Installing glpk via apt-get...")
            os.system('apt-get install -y -qq glpk-utils')
        else:
            print("Installing glpk via conda...")
            os.system('conda install -c conda-forge glpk')
        assert package_available("glpk"), "glpk is not available"   
        
    command_with_output("glpsol --version")

def install_cbc():
    if package_available("cbc"):
        print("CBC found! No need to install.")
    else:
        if on_colab():
            print("Installing cbc via zip file...")
            os.system('wget -N -q "https://ampl.com/dl/open/cbc/cbc-linux64.zip"')
            os.system('unzip -o -q cbc-linux64')
        else:
            print("Installing cbc via apt-get...")
            os.system('apt-get install -y -qq coinor-cbc')
        assert package_available("cbc"), "cbc installation was not successful."
    
    command_with_output("./cbc -v")
        
def install_bonmin():
    if package_available("bonmin"):
        print("bonmin found! No need to install.")
    else:
        if on_colab():
            print("Installing bonmin via zip file...")
            os.system('wget -N -q "https://ampl.com/dl/open/bonmin/bonmin-linux64.zip"')
            os.system('unzip -o -q bonmin-linux64')
        assert package_available("bonmin"), "bonmin is not available"
        
    command_with_output("./bonmin -v")

def install_couenne():
    if package_available("couenne"):
        print("bonmin found! No need to install.")
    else:
        if on_colab():
            print("Installing couenne via via zip file...")
            os.system('wget -N -q "https://ampl.com/dl/open/couenne/couenne-linux64.zip"')
            os.system('unzip -o -q couenne-linux64')
        assert package_available("couenne"), "couenne is not available"
    
    command_with_output("./couenne -v")

def install_gecode():
    if package_available("gecode"):
        print("gecode found! No need to install.")
    else:
        if on_colab():
            print("Installing gecode via via zip file...")
            os.system('wget -N -q "https://ampl.com/dl/open/gecode/gecode-linux64.zip"')
            os.system('unzip -o -q gecode-linux64')
        assert package_available("gecode"), "gecode is not available"
    
    command_with_output("./gecode -v")
