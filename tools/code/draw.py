import sys
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import namedtuple
import pandas as pd
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
from anytree import Node
from anytree.exporter import DotExporter
import dot2tex
from pathlib import Path

_output_path = '/work in progress/MO book/results'

def SetOutputPath( path ):
    global _output_path
    _output_path = path
    if not _output_path.endswith('/'):
        _output_path += '/'
    Path(_output_path).mkdir(parents=True, exist_ok=True)

def Interpret( model ):
    collect = lambda what : list( model.component_objects(what, active=True) )
    variables   = collect(pyo.Var)
    constraints = collect(pyo.Constraint)
    objectives  = collect(pyo.Objective)

    names       = [f.name for f in constraints + objectives]
    expressions = [f.body for f in constraints] + [f.expr for f in objectives]

    repn = [ generate_standard_repn(e) for e in expressions ]
    coefficients = { name : { v.name : c for v,c in zip(repn.linear_vars,repn.linear_coefs) } for name,repn in zip(names,repn) }

    Result = namedtuple(  'Canonical'
                        , [ 'lower_variable'
                           , 'upper_variable'
                           , 'lower_constraint'
                           , 'upper_constraint'
                           , 'formulas'
                           , 'coefficients'
                           , 'objective'
                          ]
                       )

    return Result(  { v.name : v.bounds[0] for v in variables }
                  , { v.name : v.bounds[1] for v in variables }
                  , { c.name : c.lb for c in constraints }
                  , { c.name : c.ub for c in constraints }
                  , { n : str(f) for n,f in zip(names,expressions) }
                  , coefficients
                  , { o.name : 'min' if o.is_minimizing() else 'max' for o in objectives } 
                 )
    
def GetCanonicalMatrices( rep ):
    variables   = list(rep.lower_variable.keys())
    constraints = list(rep.lower_constraint.keys())
    objective   = list(rep.objective.keys())[0]
    n = len(variables)
    A = []
    b = []
    expressions = []
    for j,v in enumerate(variables):
        if rep.lower_variable[v] is not None:
            line = np.zeros(n)
            line[j] = -1
            A.append(line)
            b.append(-rep.lower_variable[v])
            expressions.append(v+' \geq '+str(rep.lower_variable[v]) )
        if rep.upper_variable[v] is not None:
            line = np.zeros(n)
            line[j] = 1
            A.append(line)
            b.append(rep.upper_variable[v])
            expressions.append(v+' \leq '+str(rep.upper_variable[v]) )
    for c in constraints:
        line = np.array([rep.coefficients[c].get(v,0) for v in variables])
        if rep.lower_constraint[c] is not None:
            A.append(-1*line)
            b.append(-rep.lower_constraint[c])
            expressions.append(rep.formulas[c]+' \geq '+str(rep.lower_constraint[c]) )
        if rep.upper_constraint[c] is not None:
            A.append(line)
            b.append(rep.upper_constraint[c])
            expressions.append(rep.formulas[c]+' \leq '+str(rep.upper_constraint[c]) )
    factor = 1 if rep.objective[objective] == 'max' else -1
    return np.vstack(A),np.vstack(b),factor*np.array([rep.coefficients[objective].get(v,0) for v in variables]),expressions
 
def GetBasicFeasibleSolutions( A, b ):
    m,n   = A.shape
    basis = []
    for i,j in itertools.combinations(range(len(b)),n):
        try:
            basis.append( np.linalg.solve(A[[i,j],:],b[[i,j]]).T )
        except:
            continue   
    import sys
    basis = np.vstack(basis).T
    return sys.float_info.epsilon*10 + basis[:,np.all( np.dot(A, basis) <= b+sys.float_info.epsilon*1e3, axis = 0 )]
 
# TODO: the blues gradinet for isolines!
# https://stackoverflow.com/questions/35394564/is-there-a-context-manager-for-temporarily-changing-matplotlib-settings
def Draw( model, file_name=None, trajectories=dict(), isolines=True, integer=False, xlim=None, ylim=None, title=None ):
    rep       = Interpret(model)
    variables = list(rep.lower_variable.keys())
    n         = len(variables)
    assert(n==2)

    plt.grid()
    plt.xlabel(r'$'+variables[0].replace('x','x_')+'$')
    plt.ylabel(r'$'+variables[1].replace('x','x_')+'$')

    A,b,c,expressions = GetCanonicalMatrices( rep )
    basis = GetBasicFeasibleSolutions( A, b )

    x = n*[[]]
    if ( xlim is None ) or ( ylim is None ):
        for i in range(n):
            min_i = min(0,min(basis[i]))
            max_i = max(0,max(basis[i]))
            delta = max_i - min_i
            x[i]  = np.linspace( min_i-delta/8, max_i+delta/8, 1000 )
    else:
        x[0] = np.linspace( xlim[0], xlim[1], 1000 )
        x[1] = np.linspace( ylim[0], ylim[1], 1000 )

    m,_ = A.shape
    for j in range(m):
        label = expressions[j].replace('*','').replace('x','x_')
        label = r'$'+label+'$'
        row = A[j,:]
        if np.count_nonzero(row) == n:
            X = (b[j]-row[0]*x[0])/row[1]
            plt.plot(x[0], X, label = label, zorder=3, alpha=1)
        else:
            assert( np.count_nonzero(row) == 1 )
            if row[0] == 0:
                plt.plot(x[0], b[j]/row[1]*np.ones_like(x[1]), label = label, zorder=2, alpha=1)
            else:
                assert( row[1] == 0 )
                plt.plot(b[j]/row[0]*np.ones_like(x[0]), x[1], label = label, zorder=2, alpha=1)

    if basis.size > 0:
        opt = basis[:,np.argmax(np.dot(c,basis))]
        plt.plot( opt[0], opt[1], 'o', label = r'$'+str(tuple(opt.round(1)))+'$', color='gray', zorder=10 )

    obj = c if list(rep.objective.values())[0] == 'max' else -1*c
    if isolines:
        for value in sorted(np.dot(obj,basis)):
            label = rep.formulas[list(rep.objective.keys())[0]].replace('x','x_').replace('*','')
            label = r'$'+label+' = '+str(round(value,1))+'$'
            if obj[0] == 0 and obj[1] != 0:
                plt.plot(x[0], value/obj[1]*np.ones_like(x[1]), '--', label = label, zorder=5, alpha=1)
            elif obj[0] != 0 and obj[1] == 0:
                plt.plot(value/obj[0]*np.ones_like(x[0]), x[1], '--', label = label, zorder=5, alpha=1)
            elif obj[0] == 0 and obj[1] == 0:
                assert( opt == 0 )
                plt.plot(np.zeros_like(x[0]), x[1], '--', label = label, zorder=5, alpha=1)
            else:
                assert( all( c != 0 ) )
                plt.plot(x[0], (value-obj[0]*x[0])/obj[1], '--', label = label, zorder=5, alpha=1)

    plt.plot( basis[0], basis[1], 'o', color='gray', fillstyle='none', zorder=11 )

    x[0],x[1] = np.meshgrid(x[0],x[1])
    borders = [ (A[j,0]*x[0]+A[j,1]*x[1] <= b[j]).astype(int) for j in range(m) ]
    image = borders[0]
    for i in range(1,len(borders)):
        image *= borders[i]
        
    plt.imshow( image
               , extent=(x[0].min(),x[0].max(),x[1].min(),x[1].max())
               , origin="lower"
               , cmap="Greys"
               , alpha = 0.2
               , zorder = 0)
    
    for label,points in trajectories.items():
        plt.plot(*zip(*points),'o-',label=label,linewidth=5,zorder=9)

    if xlim is None:
        plt.xlim( x[0].min(),x[0].max() )
    else:
        plt.xlim( xlim )

    if ylim is None:
        plt.ylim( x[1].min(),x[1].max() )
    else:
        plt.ylim( ylim )

    if integer:
        import itertools  
        points = list(itertools.product( range(int(x[0].min()),int(x[0].max())+1), range(int(x[1].min()),int(x[1].max())+1) ) )
        feasible   = [ p for p in points if ( np.dot(A,p) <= b.T + sys.float_info.epsilon*10 ).all() ]
        infeasible = [ p for p in points if ( np.dot(A,p) > b.T + sys.float_info.epsilon*10 ).any() ]
        if infeasible:
            plt.plot( *zip(*infeasible), 'ro', zorder=8)
        if feasible:
            plt.plot( *zip(*feasible), 'bo', zorder=8)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect('equal', adjustable='box')
    
    if title:
        plt.title( title )
    
    plt.draw()

    if file_name is not None:
        plt.savefig( _output_path+file_name, bbox_inches='tight', pad_inches=0 )

    plt.show()      

    return pd.DataFrame( np.vstack( [ basis, np.dot(obj,basis) ] ).T.round(1), columns=variables + ['value'] ).sort_values( by=variables )

def BB( m, solver='cbc', draw_integer=True, xlim=(-.5,5.5), ylim=(-.5,7.5) ):
    x      = [ m.x1, m.x2 ]
    xi     = [ x._name.replace('x','x_') for x in x ]
    solver = pyo.SolverFactory(solver)

    lb  = -np.inf
    sol = None

    idx = 0

    root = None

    def _BB( m, parent, innequality ):
        nonlocal x, xi, solver, lb, sol, idx, root, xlim, ylim
        idx = idx+1

        node = Node( 'Node {}'.format(idx), parent ) 
        node.idx         = idx
        node.innequality = innequality
        node.x           = None
        node.lb          = None
        node.ub          = None
        node.termination = None
        
        if parent is None:
            root = node

        result = solver.solve(m)
        Draw(m,integer=draw_integer,isolines=False, xlim=xlim, ylim=ylim, file_name='{}_{}.pdf'.format(m.name, idx), title='Node {}'.format(idx) )
        if (result.solver.status == pyo.SolverStatus.ok) and \
           (result.solver.termination_condition == pyo.TerminationCondition.optimal):
            node.x = (pyo.value(x[0]),pyo.value(x[1]))
            ub = pyo.value( m.obj )
            if ub > lb:
                s = np.array( [ pyo.value(x) for x in x ] )
                print(s)
                fs = np.floor(s)
                cs = np.ceil(s)
                j = np.argmax( np.absolute( s - ( fs + (cs-fs)/2 ) ) )
                if fs[j] == cs[j]:
                    if ub > lb:
                        lb = ub
                        sol = s
                    node.termination = 'feasible'
                    node.lb          = lb
                    node.ub          = ub
                else:
                    node.lb          = lb
                    node.ub          = ub
                    xlb,xub = x[j].bounds
                    x[j].setub( fs[j] )
                    _BB(m, node, (xi[j],r'\leq',fs[j]) )
                    x[j].setub( xub )
                    x[j].setlb( cs[j] )
                    _BB(m, node, (xi[j],r'\geq',cs[j]) )
                    x[j].setlb( xlb )
            else:
                node.termination = 'fathomed'
                node.lb          = lb
                node.ub          = ub
        else:
            node.termination = 'infeasible'
            node.lb          = lb
            node.ub          = -np.inf

    _BB( m, root, None )
    return sol, root

nodecolors = { 'infeasible' : 'red', 'feasible' : 'green', 'fathomed' : 'magenta' }

def TeXnumber( x ):
    if np.isinf(x):
        return '-'
    return '{:.1f}'.format(x)

node_labels = { 
#    'infeasible' : lambda node : 'label="{{ {} | {{ {} | {} }} | {} }}" shape=record color={}'.format(node.name,TeXnumber(node.lb),TeXnumber(node.ub),node.termination,nodecolors.get(node.termination,'blue')),
    'infeasible' : lambda node : 'label="{{ {} | ({}, {}) | {{ {}|{} }} | {} }}" shape=record color={}'.format(node.name,'-','-',TeXnumber(node.lb),TeXnumber(node.ub),node.termination,nodecolors.get(node.termination,'blue')),
    'all'        : lambda node : 'label="{{ {} | ({:.1f}, {:.1f}) | {{ {}|{} }} | {} }}" shape=record color={}'.format(node.name,*node.x,TeXnumber(node.lb),TeXnumber(node.ub),node.termination,nodecolors.get(node.termination,'blue')),
    'default'    : lambda node : 'label="{{ {} | ({:.1f}, {:.1f}) | {{ {}|{} }} }}" shape=record color={}'.format(node.name,*node.x,TeXnumber(node.lb),TeXnumber(node.ub),nodecolors.get(node.termination,'blue'))
    }

def nodeattrfunc(node):
    if node.termination:
        if node.x:
            return node_labels['all'](node)        
        else:
            return node_labels['infeasible'](node)
    return node_labels['default'](node)

def edgeattrfunc(node, child):
    label = '{} {} {:.0f}'.format(*child.innequality)
    return 'label="dummy" texlbl="${}$"'.format(label)

def edgetypefunc(node, child):
    return '--'
 
def Dotter( root ):
    return DotExporter(root,nodeattrfunc=nodeattrfunc,edgeattrfunc=edgeattrfunc)

def DrawBB( root, file_name ):
    Dotter(root).to_picture(_output_path+file_name)
    
def ToTikz( root, tex_file_name, dot_file_name=None, fig_only=True ):
    dot = '\n'.join(list(Dotter(root)))
    if dot_file_name:
        with open(_output_path+dot_file_name,'w') as f:
            f.write(dot)
    tex = dot2tex.dot2tex(dot,crop=True,figonly=fig_only,tikzedgelabels=False)
#    tex = '\n'.join( [line for line in tex.splitlines() if not '-#' in line] )
    tex = tex.replace('\n-#0000','black')
    tex = tex.replace('-#0000','black')
    with open(_output_path+tex_file_name,'w') as f:
        f.write(tex)