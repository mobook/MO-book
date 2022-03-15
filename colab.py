import sys
import subprocess

def pip_install(package, solver=None):
    print(f"{package} ...", end="")
    print(f"installing ...", end="")
    out = subprocess.run(['pip3', 'install', '-q', package])
    if out.returncode:
        print(f"failed")
        return
    if solver is not None:
        test_solver(solver)

def apt_install(package, solver=None):
    print(f"installing {package} ... ", end="")
    out = subprocess.run(['apt-get', 'install', '-y', '-q', package])
    if out.returncode:
        print(f"failed")
        return
    if solver is not None:
        test_solver(solver)

def ampl_install(package, solver=None):
    print(f"installing {package} ... ", end="")    
    out = subprocess.run(['wget', '-N', '-q', f'https://ampl.com/dl/open/{package}/{package}-linux64.zip'])  
    if out.returncode:
        print(f"failed to download")
        return
    out = subprocess.run(['unzip', '-o', '-q', f'{package}-linux64'])
    if out.returncode:
        print(f"failed to unzip")
        return
    if solver is not None:
        test_solver(solver) 
        
def test_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.y = pyo.Var(bounds=(0, 10))

    @m.Objective(sense=pyo.maximize)
    def obj(m):
        return m.x + m.y

    @m.Constraint()
    def con(m):
        return m.x + 2*m.y <= 5

    return m

def test_solver(solver):
    m = test_model()
    print(f"testing {solver} ...", end="")
    try:
        pyo.SolverFactory(solver).solve(m)
        print('success')
    except:
        return
        
if "google.colab" in sys.modules:

    # install Pyomo
    pip_install('pyomo')
    try:
        import pyomo.environ as pyo
        print()
    except:
        print("pyomo not found")
        quit()
    
    # install solvers from PiPy
    pip_install('gurobipy', 'gurobi_direct')
    #pip_install('cplex', 'cplex')
    pip_install('xpress', 'xpress')

    # install debian packages
    apt_install('glpk-utils', 'glpk')
    apt_install('coinor-cbc', 'cbc')

    # install ampl binaries
    ampl_install('ipopt', 'ipopt')
    ampl_install('bonmin', 'bonmin')
    ampl_install('couenne', 'couenne')
    #ampl_install('gecode', 'gecode')
    #ampl_install('jacop', 'jacop')

import pyomo.environ as pyo
    
def test_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.y = pyo.Var(bounds=(0, 10))

    @m.Objective(sense=pyo.maximize)
    def obj(m):
        return m.x + m.y

    @m.Constraint()
    def con(m):
        return m.x + 2*m.y <= 5

    return m
