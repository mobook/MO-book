import sys

def on_colab():
    return "google.colab" in sys.modules

def mplstyle():
    pass
    
if on_colab():
    print("Hi from Colab")


