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
    returncode = await run(f'pip3 install -q {pkg}')
    if not returncode:
        print(f"{solver} installed")
    else:
        print(f"{solver} not installed")

async def apt(pkg:str, solver:str):
    returncode = await run(f'apt-get install -y -q {pkg}')
    if not returncode:
        print(f"{solver} installed")
    else:
        print(f"{solver} not installed")

async def ampl(pkg:str, solver:str):
    await run(f'wget -N -q https://ampl.com/dl/open/{pkg}/{pkg}-linux64.zip')
    returncode = await run(f'unzip -o -q {pkg}-linux64')
    if not returncode:
        print(f"{solver} installed")
    else:
        print(f"{solver} not installed")

async def install_pyomo():
    await pip("pyomo", "pyomo")
    await apt("glpk-utils", "glpk"),
    await asyncio.gather(
        pip("gurobipy", "gurobi_direct"),
        pip("xpress", "xpress"),
        pip("cplex", "cplex"),
        apt("coinor-cbc", "cbc"),
        ampl("ipopt", "ipopt"),
        ampl("bonmin", "bonmin"),
        ampl("couenne", "couenne"),
        ampl("gecode", "gecode"),
        ampl("jacop", "jacop")
    )
    print("--- Installation complete")

nest_asyncio.apply()
asyncio.run(install_pyomo())
