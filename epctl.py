import sympy as sp

def sat_agent(agent, phi, until=False, next=False):
    if next:
        custom_hitting_function = lambda t_a, t_b, s : h_next(agent,t_a,s)
    elif (not until):
        custom_hitting_function = lambda t_a, t_b, s : h_cond(agent,t_a,t_b,s)
    else:
        custom_hitting_function = lambda t_a, t_b, s : h_cond_bounded(agent,t_a,t_b,s,until)

    temp_before = set()

    for disjunct in phi["psi"]["before"]:
        temp = set(states)
        for conjunct in disjunct:
            if conjunct[0] != "-":
                temp = temp.intersection([i for i in states if conjunct in label[agent][i]])
            else:
                temp = temp.intersection([i for i in states if conjunct[1:] not in label[agent][i]])
        temp_before = temp_before.union(temp)

    temp_after = set()

    for disjunct in phi["psi"]["after"]:
        temp = set(states)
        for conjunct in disjunct:
            if conjunct[0] != "-":
                temp = temp.intersection([i for i in states if conjunct in label[agent][i]])
            else:
                temp = temp.intersection([i for i in states if conjunct[1:] not in label[agent][i]])
        temp_after = temp_after.union(temp)

    return set([ s for s in states if phi["probability"]( custom_hitting_function(temp_after,temp_before,s) ) ])

def h_cond(agent,A,B,initial_state):
    syms = sp.symbols("x0:"+str(n_states))
    variables = sp.Matrix(syms[0:n_states])
    setOfEquations = sp.Matrix(transition[agent]) * variables

    for i in A:
        setOfEquations.row_del(i)
        setOfEquations = setOfEquations.row_insert(i, sp.Matrix([1]))
        
    for i in (states-A-B):
        setOfEquations.row_del(i)
        setOfEquations = setOfEquations.row_insert(i, sp.Matrix([0]))

    eq = sp.Eq(setOfEquations,variables)
    ans = (sp.solve(eq,syms,dict=True))[0].get(syms[initial_state])

    if ans:
        return ans.subs((x,0) for x in syms)
    else:
        return 0
    
def h_cond_bounded(agent,A,B,initial_state,t):
    modified_transition_matrix = sp.Matrix(transition[agent])
    
    for i in states-B:
        modified_transition_matrix.row_del(i)
        modified_transition_matrix = modified_transition_matrix.row_insert(i, sp.zeros(1,n_states))
        modified_transition_matrix[i,i] = 1
    
    modified_transition_matrix = modified_transition_matrix**t

    sum = 0
    for i in A:
        sum = sum + modified_transition_matrix[initial_state,i]

    return sum

def h_next(agent,A,initial_state):
    transition_matrix = sp.Matrix(transition[agent])

    sum = 0
    for i in A:
        sum = sum + transition_matrix[initial_state,i]

    return sum
    
def h(agent,A,initial_state):
    syms = sp.symbols("x0:"+str(n_states))
    variables = sp.Matrix(syms[0:n_states])
    setOfEquations = sp.Matrix(transition[agent]) * variables

    for i in A:
        setOfEquations.row_del(i)
        setOfEquations = setOfEquations.row_insert(i, sp.Matrix([1]))

    eq = sp.Eq(setOfEquations,variables)
    ans = (sp.solve(eq,syms,dict=True))[0].get(syms[initial_state])

    if ans:
        return ans.subs((x,0) for x in syms)
    else:
        return 0

def jaccard(a,b):
    if ( set(a).union(set(b)) == set()):
        return 0
    else:
        return 1 - (len(set(a).union(set(b))) - len(set(a).intersection(set(b))))/len(set(a).union(set(b)))



def sat(alpha):
    out = []
    Phij = {}
    Phii = {}
    for s in states:
        Phij[s] = []
        Phii[s] = []

        for s1 in sat_agent("j",alpha["phi"]):
            if h("j",[s1],s) == 1:
                Phij[s].append(s1)

        for s1 in sat_agent("i",alpha["phi"]):
            if h("i",[s1],s) == 1:
                Phii[s].append(s1)

    for s in states:
        ji = jaccard(Phij[s],Phii[s])
        print("State "+str(s)+" has jaccard index = "+str(ji))
        if alpha["accuracy"](ji):
            out.append(s)

    return set(out)

if __name__=="__main__":
    global states
    global agents
    global transition
    global probability
    global ap
    global label
    global n_states
    
    # Representation of the models "j" and "i" of the example in
    # "Modelling and checking explanation reliability of surrogate models"
    # submitted to the Special Issue on Logic, Rationality and Interaction (LORI)

    n_states = 16
    states = set(range(n_states))
    agents = {"j","i"}

    transition = {}
    transition["j"] = [[0.25, 0.25, 0.25, 0.25, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 1   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0.25, 0   , 0   , 0.25, 0.25, 0.25, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 ]]

    transition["i"] = [[0.25, 0.25, 0.25, 0.25, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0.5 , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0.25, 0   , 0   , 0.25, 0.25, 0.25, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.25, 0.25, 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   , 0   , 0   , 0.5 , 0   , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 , 0   ],
                      [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.5 , 0.5 ]]

    label = {}

    label["j"] = {}
    label["j"][0] = {}
    label["j"][1] = {}
    label["j"][2] = {}
    label["j"][3] = {}
    label["j"][4] = {}
    label["j"][5] = {}
    label["j"][6] = {}
    label["j"][7] = {}
    label["j"][8] = {}
    label["j"][9] = {"p"}
    label["j"][10] = {"p"}
    label["j"][11] = {}
    label["j"][12] = {}
    label["j"][13] = {}
    label["j"][14] = {"p"}
    label["j"][15] = {"p"}

    label["i"] = {}
    label["i"][0] = {}
    label["i"][1] = {}
    label["i"][2] = {}
    label["i"][3] = {}
    label["i"][4] = {}
    label["i"][5] = {}
    label["i"][6] = {}
    label["i"][7] = {}
    label["i"][8] = {}
    label["i"][9] = {}
    label["i"][10] = {"p"}
    label["i"][11] = {"p"}
    label["i"][12] = {}
    label["i"][13] = {}
    label["i"][14] = {"p"}
    label["i"][15] = {"p"}

    # Partitions
    tilde = {} 
    tilde["u"] = {}
    tilde["u"][4] = [0,1,2,3,4,6,7,10,11,12,13]

    beta = {}
    beta["phi"] = {}
    beta["phi"]["probability"] = lambda x: x == 1
    beta["phi"]["psi"] = {}
    beta["phi"]["psi"]["before"] = [[ ]]
    beta["phi"]["psi"]["after"] = [[ "p" ]]
    beta["accuracy"] = lambda x: x >= 0.75

    print(sat_agent("j",beta["phi"]))
    print("\n")
    print(sat_agent("i",beta["phi"]))

    # Formula
    # A^{e}_{>= 0.75} P_{=1} (\top \cup p)
    alpha = {}
    alpha["phi"] = {}
    alpha["phi"]["probability"] = lambda x: x == 1
    alpha["phi"]["psi"] = {}
    alpha["phi"]["psi"]["before"] = [[ ]] # [[ ]] stands for \top
    alpha["phi"]["psi"]["after"] = [[ "p" ]]
    alpha["accuracy"] = lambda x: x >= 0.75
    
    # TODO: Extend alpha to most general form:
    #
    # alpha := *A^{Gamma}_{nabla h} phi
    # phi := p |
    #        phi1 and phi2 |      // sat(phi1) intersect sat(phi2)
    #        neg phi |            // complement of sat(phi)
    #        *P_{nabla p} psi
    #
    # psi := *next phi |
    #        *phi until phi |
    #        *phi until_{<= t} phi
    #
    # * = done

    s = sat(beta)
    
    print("*****")
    print("The following states satisfy the accuracy requirement: " + str(s))
    