import asyncio
import nest_asyncio

async def run(cmd: str):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE, 
        stderr=asyncio.subprocess.PIPE
        )
    stdout, stderr = await proc.communicate()
    #print(f"{cmd} exited with {proc.returncode}")
    #if stdout:
    #    print(stdout.decode())
    if stderr:
        print(stderr.decode())
    return proc.returncode

async def pip(pkg:str, solver:str):
    if not await run(f'pip3 install -q {pkg}'):
        test_solver(solver)
    else:
        print(f". {solver} not installed")

async def apt(pkg:str, solver:str):
    if not await run(f'apt-get install -y -q {pkg}'):
        test_solver(solver)
    else:
        print(f". {solver} not installed")

async def ampl(pkg:str, solver:str):
    await run(f'wget -N -q https://ampl.com/dl/open/{pkg}/{pkg}-linux64.zip')
    if not await run(f'unzip -o -q {pkg}-linux64'):
        test_solver(solver)
    else:
        print(f". {solver} failed download")

def test_solver(solver):
        import pyomo.environ as pyo
        model = pyo.ConcreteModel()
        model.x = pyo.Var()
        model.c = pyo.Constraint(expr=model.x >= 0)
        model.obj = pyo.Objective(expr=model.x)
        try:
            pyo.SolverFactory(solver).solve(model)
            print(f". {solver}")
        except:
            print(f". {solver} test failed")

async def install_pyomo():
    returncode = await run("pip3 install -q pyomo")
    if returncode:
        print("pyomo failed to install")
    else:
        print("pyomo installed")
        print("installing solvers")
        await apt("glpk-utils", "glpk"),
        await asyncio.gather(
            pip("gurobipy", "gurobi_direct"),
            pip("xpress", "xpress"),
            #pip("cplex", "cplex"),
            apt("coinor-cbc", "cbc"),
            ampl("ipopt", "ipopt"),
            ampl("bonmin", "bonmin"),
            ampl("couenne", "couenne"),
            #ampl("gecode", "gecode"),
            #ampl("jacop", "jacop")
        )
    print("installation complete")

nest_asyncio.apply()
asyncio.run(install_pyomo())
