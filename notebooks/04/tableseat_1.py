def TableSeat( members, capacity, k, domain=pyo.NonNegativeReals ):
    m   = pyo.ConcreteModel("Dina's seat plan")
    m.F = pyo.Set( initialize=range( len(members) ) )
    m.T = pyo.Set( initialize=range( len(capacity) ) )
    m.M = pyo.Param( m.F, initialize=members )
    m.C = pyo.Param( m.T, initialize=capacity )
    m.x = pyo.Var( m.F, m.T, bounds=(0,k), domain=domain )
    
    @m.Objective( sense=pyo.maximize )
    def goal(m):
        return 0

    @m.Constraint( m.T )    
    def capacity( m, t ):
        return pyo.quicksum( m.x[f,t] for f in m.F  ) <= m.C[t]
    
    @m.Constraint( m.F )
    def seat( m, f ):
        return pyo.quicksum( m.x[f,t] for t in m.T ) == m.M[f]
        
    return m
