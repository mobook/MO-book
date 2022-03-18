import asyncio
import nest_asyncio

def lp_test(solver):
    import pyomo.environ as pyo
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.c = pyo.Constraint(expr=model.x >= 0)
    model.obj = pyo.Objective(expr=model.x)
    try:
        pyo.SolverFactory(solver).solve(model)
    except:
        print(f"{solver} failed test")
    
def ip_test(solver):
    import pyomo.environ as pyo
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Integers)
    model.c = pyo.Constraint(expr=model.x >= 0)
    model.obj = pyo.Objective(expr=model.x)
    try:
        pyo.SolverFactory(solver).solve(model)
    except:
        print(f"{solver} failed test")

async def run(cmd: str):
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE, 
        stderr=asyncio.subprocess.PIPE
        )
    stdout, stderr = await proc.communicate()
    if stderr:
        print(stderr.decode())
    return proc.returncode

async def pip_install(pkg:str, solver:str, test=lp_test):
    if await run(f'pip3 install -q {pkg}'):
        print(f"{solver} not installed ") 
        return
    test(solver)
    print(f".. {solver}")
    return

async def apt_install(pkg:str, solver:str, test=lp_test):
    if await run(f'apt-get install -y -q {pkg}'):
        print(f"{solver} not installed . ", end="")
        return
    test(solver)
    print(f".. {solver}")
    return

async def ampl_install(pkg:str, solver:str, test=lp_test):
    if await run(f'wget -N -q https://ampl.com/dl/open/{pkg}/{pkg}-linux64.zip'):
        print(f"{pkg} failed to download . ", end="")
        return
    if await run(f'unzip -o -q {pkg}-linux64'):
        print(f"{pkg} failed to unzip . ", end="")
        return  
    test(solver) 
    print(f".. {solver}")
    return
        
async def install_pyomo():
    print("installing pyomo . ", end="")
    if await run("pip3 install -q pyomo"):
        print("pyomo failed to install")
        return
    print("pyomo installed")
    print("installing and testing solvers ...")
    await apt_install("glpk-utils", "glpk"),
    await asyncio.gather(
        pip_install("gurobipy", "gurobi_direct"),
        apt_install("coinor-cbc", "cbc"),
        ampl_install("ipopt", "ipopt"),
        ampl_install("bonmin", "bonmin"),
        ampl_install("couenne", "couenne"),
        ampl_install("gecode", "gecode", ip_test),
            
        #pip_install("xpress", "xpress"),
        #pip_install("cplex", "cplex"),
        #ampl_install("jacop", "jacop")
        )
    print("installation and testing complete")

nest_asyncio.apply()
asyncio.run(install_pyomo())
