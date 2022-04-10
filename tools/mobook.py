import sys

print("OS :", sys.platform)

def on_colab():
    return "google.colab" in sys.modules


