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
    r = subprocess.getoutput(command)
    print(r)
    
def install_pyomo():
    if not package_available("pyomo"):
        print("Installing pyomo via pip...")
        os.system("pip install -q idaes_pse")
        assert package_available("pyomo"), "pyomo was not successfully installed."
        print("pyomo was successfully installed")
    else:
        print("Pyomo found! No need to install.")
    command_with_output("pyomo --version")

def install_idaes():
    if not package_available("idaes"):
        print("Installing idaes via pip...")
        os.system("pip install -q idaes_pse")
        assert package_available("idaes"), "idaes was not successfully installed."
        print("idaes was successfully installed")
    else:
        print("IDAES found! No need to install.")
    command_with_output("idaes --version")

def install_ipopt():
    if not package_available("ipopt"):
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
                print("Installing Ipopt via zip file...")
                os.system('wget -N -q "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"')
                os.system('!unzip -o -q ipopt-linux64')
            # Otherwise, try conda
            else:
                try:
                    print("Installing Ipopt via conda...")
                    os.system('conda install -c conda-forge ipopt')
                except:
                    pass

        assert package_available("ipopt"), "Ipopt is not available"       
        print("ipopt was successfully installed")  

    else:
        print("Ipopt found! No need to install.")
        
    # Verify Ipopt is now available
    command_with_output("./ipopt --version")
    if package_available("k_aug"):
        print("k_aug was successfully installed")
        command_with_output("./k_aug --version")

def install_glpk():
    if not package_available("glpk") and on_colab():
        print("Installing glpk via apt-get...")
        os.system('apt-get install -y -qq glpk-utils')
        
        # Does not work on Colab. Feb-03-2021.
        #print("Installing glpk via conda...")
        #os.system('conda install -c conda-forge glpk')
        
    # Verify package is now available
    assert package_available("glpk"), "glpk is not available"

def install_cbc():
    if not package_available("cbc") and on_colab():
        #print("Installing cbc via apt-get...")
        #os.system('apt-get install -y -qq coinor-cbc')
        print("Installing cbc via zip file...")
        os.system('wget -N -q "https://ampl.com/dl/open/cbc/cbc-linux64.zip"')
        os.system('unzip -o -q cbc-linux64')
        
    # Verify package is now available
    assert package_available("cbc"), "cbc is not available"
    
    # command_with_output("./cbc -v")
        
def install_bonmin():
    if not package_available("bonmin") and on_colab():
        print("Installing bonmin via zip file...")
        os.system('wget -N -q "https://ampl.com/dl/open/bonmin/bonmin-linux64.zip"')
        os.system('unzip -o -q bonmin-linux64')
    
    # Verify package is now available
    assert package_available("bonmin"), "bonmin is not available"
    command_with_output("./bonmin -v")

def install_couenne():
    if not package_available("couenne") and on_colab():
        print("Installing couenne via via zip file...")
        os.system('wget -N -q "https://ampl.com/dl/open/couenne/couenne-linux64.zip"')
        os.system('unzip -o -q couenne-linux64')
        
    # Verify package is now available
    assert package_available("couenne"), "couenne is not available"
    command_with_output("./couenne -v")

def install_gecode():
    if not package_available("gecode") and on_colab():
        print("Installing gecode via via zip file...")
        os.system('wget -N -q "https://ampl.com/dl/open/gecode/gecode-linux64.zip"')
        os.system('unzip -o -q gecode-linux64')
    
    # Verify package is now available
    assert package_available("gecode"), "gecode is not available"
    command_with_output("./gecode -v")
    
def _download(relative_file_names):

    # GitHub pages url
    # url = "https://ndcbe.github.io/CBE60499/"
    url = "https://raw.githubusercontent.com/ndcbe/CBE60499/main/docs/"

    # loop over all files to download
    for file_path in relative_file_names:
        print("Checking for",file_path)
        # split each file_path into a folder and filename
        stem, filename = os.path.split(file_path)
    
        # check if the folder name is not empty
        if stem:
            # add "." for Colab. Doing it here because we do not want it in the URL
            # stem = '.' + stem
            
            # check if the folder exists
            if not os.path.exists(stem):
                print("\tCreating folder",stem)
                # if the folder does not exist, create it
                os.mkdir(stem)
        # if the file does not exist, create it by downloading from GitHub pages
        if not os.path.isfile(file_path):
            file_url = urllib.parse.urljoin(url,
                    urllib.request.pathname2url(file_path))
            print("\tDownloading",file_url)
            with open(file_path, 'wb') as f:
                f.write(requests.get(file_url).content)
        else:
            print("\tFile found!")

def download_data(filenames):
    for i in range(len(filenames)):
        filenames[i] = "./data/"+filenames[i]
    
    _download(filenames)

def download_figures(filenames):
    for i in range(len(filenames)):
        filenames[i] = "./figures/"+filenames[i]
    
    _download(filenames)
