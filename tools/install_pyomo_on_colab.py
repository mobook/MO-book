import sys
import subprocess
import urllib.request

class PyomoOnGoogleColab():

    packages = {
        'gurobi_direct': {'package': 'gurobipy', 'method': 'pip'},
#        'cplex': {'package': 'cplex', 'method': 'pip'},
        'xpress': {'package': 'xpress', 'method': 'pip'},

        'glpk': {'package': 'glpk-utils', 'method': 'apt'},
        'cbc': {'package': 'coinor-cbc', 'method': 'apt'},

        'ipopt': {'package': 'ipopt', 'method': 'ampl'},
        'bonmin': {'package': 'bonmin', 'method': 'ampl'},
        'couenne': {'package': 'couenne', 'method': 'ampl'},
#        'gecode': {'package': 'gecode', 'method': 'ampl'},
#        'jacop': {'package': 'jacop', 'method': 'ampl'},
    }

    def __init__(self, solvers='all'):
        if solvers == 'all':
            self.solvers = self.packages.keys()
        else:
            self.solvers = solvers
        return

    def cmd(self, cmd_str):
        print(f" ... {cmd_str.split()[0]}", end="")
        out = subprocess.run(cmd_str.split())
        if out.returncode:
            print("failed", end="")
        else:
            print(" ok", end="")
        return out.returncode

    def test_solver(self, solver):
        print(f" ... test ", end="")
        import pyomo.environ as pyo
        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.c = pyo.Constraint(expr=model.x >= 0)
        model.obj = pyo.Objective(expr=model.x)
        try:
            pyo.SolverFactory(solver).solve(model)
            print(" ok", end="")
        except:
            print(" failed", end="")
        return

    def run(self):
        print("pyomo", end="")
        self.cmd(f'pip3 install -q pyomo')
        print()
        for solver in self.solvers:
            print(f"{solver}", end="")
            package = self.packages[solver]['package']
            method = self.packages[solver]['method']
            if method == 'pip':
                self.cmd(f"pip3 install -q {package}")
            elif method == 'apt':
                self.cmd(f"apt-get install -y -q {package}")
            elif method == 'ampl':
                self.cmd(f'wget -N -q https://ampl.com/dl/open/{package}/{package}-linux64.zip')
                self.cmd(f'unzip -o -q {package}-linux64')
            self.test_solver(solver)
            print()
        return

if __name__ == "__main__":
    solvers = sys.argv[1:]
    installer = PyomoOnGoogleColab(solvers)
    installer.run()
