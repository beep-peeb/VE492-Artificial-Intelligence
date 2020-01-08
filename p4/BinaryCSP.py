from collections import deque
import util

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
        Dequeue the earliest enqueued item still in the queue. This
        operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

"""
    Base class for unary constraints
    Implement isSatisfied in subclass to use
"""
class UnaryConstraint:
    def __init__(self, var):
        self.var = var

    def isSatisfied(self, value):
        util.raiseNotDefined()

    def affects(self, var):
        return var == self.var


""" 
    Implementation of UnaryConstraint
    Satisfied if value does not match passed in paramater
"""
class BadValueConstraint(UnaryConstraint):
    def __init__(self, var, badValue):
        self.var = var
        self.badValue = badValue

    def isSatisfied(self, value):
        return not value == self.badValue

    def __repr__(self):
        return 'BadValueConstraint (%s) {badValue: %s}' % (str(self.var), str(self.badValue))


""" 
    Implementation of UnaryConstraint
    Satisfied if value matches passed in paramater
"""
class GoodValueConstraint(UnaryConstraint):
    def __init__(self, var, goodValue):
        self.var = var
        self.goodValue = goodValue

    def isSatisfied(self, value):
        return value == self.goodValue

    def __repr__(self):
        return 'GoodValueConstraint (%s) {goodValue: %s}' % (str(self.var), str(self.goodValue))


"""
    Base class for binary constraints
    Implement isSatisfied in subclass to use
"""
class BinaryConstraint:
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def isSatisfied(self, value1, value2):
        util.raiseNotDefined()

    def affects(self, var):
        return var == self.var1 or var == self.var2

    def otherVariable(self, var):
        if var == self.var1:
            return self.var2
        return self.var1


"""
    Implementation of BinaryConstraint
    Satisfied if both values assigned are different
"""
class NotEqualConstraint(BinaryConstraint):
    def isSatisfied(self, value1, value2):
        if value1 == value2:
            return False
        return True

    def __repr__(self):
        return 'BadValueConstraint (%s, %s)' % (str(self.var1), str(self.var2))


class ConstraintSatisfactionProblem:
    """
    Structure of a constraint satisfaction problem.
    Variables and domains should be lists of equal length that have the same order.
    varDomains is a dictionary mapping variables to possible domains.

    Args:
        variables (list<string>): a list of variable names
        domains (list<set<value>>): a list of sets of domains for each variable
        binaryConstraints (list<BinaryConstraint>): a list of binary constraints to satisfy
        unaryConstraints (list<BinaryConstraint>): a list of unary constraints to satisfy
    """
    def __init__(self, variables, domains, binaryConstraints = [], unaryConstraints = []):
        self.varDomains = {}
        for i in range(len(variables)):
            self.varDomains[variables[i]] = domains[i]
        self.binaryConstraints = binaryConstraints
        self.unaryConstraints = unaryConstraints

    def __repr__(self):
        return '---Variable Domains\n%s---Binary Constraints\n%s---Unary Constraints\n%s' % ( \
            ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
            ''.join([str(e) + '\n' for e in self.binaryConstraints]), \
            ''.join([str(e) + '\n' for e in self.binaryConstraints]))


class Assignment:
    """
    Representation of a partial assignment.
    Has the same varDomains dictionary stucture as ConstraintSatisfactionProblem.
    Keeps a second dictionary from variables to assigned values, with None being no assignment.

    Args:
        csp (ConstraintSatisfactionProblem): the problem definition for this assignment
    """
    def __init__(self, csp):
        self.varDomains = {}
        for var in csp.varDomains:
            self.varDomains[var] = set(csp.varDomains[var])
        self.assignedValues = { var: None for var in self.varDomains }

    """
    Determines whether this variable has been assigned.

    Args:
        var (string): the variable to be checked if assigned
    Returns:
        boolean
        True if var is assigned, False otherwise
    """
    def isAssigned(self, var):
        return self.assignedValues[var] != None

    """
    Determines whether this problem has all variables assigned.

    Returns:
        boolean
        True if assignment is complete, False otherwise
    """
    def isComplete(self):
        for var in self.assignedValues:
            if not self.isAssigned(var):
                return False
        return True

    """
    Gets the solution in the form of a dictionary.

    Returns:
        dictionary<string, value>
        A map from variables to their assigned values. None if not complete.
    """
    def extractSolution(self):
        if not self.isComplete():
            return None
        return self.assignedValues

    def __repr__(self):
        return '---Variable Domains\n%s---Assigned Values\n%s' % ( \
            ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
            ''.join([str(e) + ':' + str(self.assignedValues[e]) + '\n' for e in self.assignedValues]))



####################################################################################################


"""
    Checks if a value assigned to a variable is consistent with all binary constraints in a problem.
    Do not assign value to var. Only check if this value would be consistent or not.
    If the other variable for a constraint is not assigned, then the new value is consistent with the constraint.

    Args:
        assignment (Assignment): the partial assignment
        csp (ConstraintSatisfactionProblem): the problem definition
        var (string): the variable that would be assigned
        value (value): the value that would be assigned to the variable
    Returns:
        boolean
        True if the value would be consistent with all currently assigned values, False otherwise
"""
def consistent(assignment, csp, var, value):
    """Question 1"""
    #Your code here
    constraints = [c for c in csp.binaryConstraints if c.affects(var)]
    for c in constraints:
        neighbour = c.otherVariable(var)
        if neighbour in assignment.assignedValues.keys():
            if not c.isSatisfied(value,assignment.assignedValues[neighbour]):
                return False
    return True

"""
    Recursive backtracking algorithm.
    A new assignment should not be created. The assignment passed in should have its domains updated with inferences.
    In the case that a recursive call returns failure or a variable assignment is incorrect, the inferences made along
    the way should be reversed. See maintainArcConsistency and forwardChecking for the format of inferences.

    Examples of the functions to be passed in:
    orderValuesMethod: orderValues, leastConstrainingValuesHeuristic
    selectVariableMethod: chooseFirstVariable, minimumRemainingValuesHeuristic
    inferenceMethod: noInferences, maintainArcConsistency, forwardChecking

    Args:
        assignment (Assignment): a partial assignment to expand upon
        csp (ConstraintSatisfactionProblem): the problem definition
        orderValuesMethod (function<assignment, csp, variable> returns list<value>): a function to decide the next value to try
        selectVariableMethod (function<assignment, csp> returns variable): a function to decide which variable to assign next
        inferenceMethod (function<assignment, csp, variable, value> returns set<variable, value>): a function to specify what type of inferences to use
    Returns:
        Assignment
        A completed and consistent assignment. None if no solution exists.
"""
def recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod):
    """Question 1"""
##    print(orderValuesMethod)
##    print(selectVariableMethod)
##    print(inferenceMethod)
    if assignment.isComplete():
        return assignment
    var = selectVariableMethod(assignment, csp)
##    print("choosing var:", var)
##    print("domain:",assignment.varDomains)
##    print("order of value:",orderValuesMethod(assignment,csp,var))
    if not var:
        print("testing here")
    for value in orderValuesMethod(assignment,csp,var):
        if consistent(assignment,csp,var,value):
            inferences = inferenceMethod(assignment, csp, var, value)

            if inferences != None:
##                print("assign:", var, value)
                assignment.assignedValues[var] = value
                result = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
                if result != None:
                    return result

                assignment.assignedValues[var] = None
                for i in inferences:
                    assignment.varDomains[i[0]].add(i[1])

    return None


"""
    Uses unary constraints to eleminate values from an assignment.

    Args:
        assignment (Assignment): a partial assignment to expand upon
        csp (ConstraintSatisfactionProblem): the problem definition
    Returns:
        Assignment
        An assignment with domains restricted by unary constraints. None if no solution exists.
"""
def eliminateUnaryConstraints(assignment, csp):
    domains = assignment.varDomains
    for var in domains:
        for constraint in (c for c in csp.unaryConstraints if c.affects(var)):
            for value in (v for v in list(domains[var]) if not constraint.isSatisfied(v)):
                domains[var].remove(value)
                if len(domains[var]) == 0:
                    # Failure due to invalid assignment
                    return None
    return assignment


"""
    Trivial method for choosing the next variable to assign.
    Uses no heuristics.
"""
def chooseFirstVariable(assignment, csp):
    for var in csp.varDomains:
        if not assignment.isAssigned(var):
            return var


"""
    Selects the next variable to try to give a value to in an assignment.
    Uses minimum remaining values heuristic to pick a variable. Use degree heuristic for breaking ties.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
    Returns:
        the next variable to assign
"""
def minimumRemainingValuesHeuristic(assignment, csp):
    nextVar = None
    domains = assignment.varDomains
    """Question 2"""
    unassigned = []
    for each in domains.keys():
        if domains[each] and not assignment.isAssigned(each):
            unassigned.append((len(domains[each]),each))
##    print("unassigned:", unassigned)
    unassigned.sort()
    if len(unassigned) == 0:
        print("error!")
    minnum = unassigned[0][0]

    compare = []
    for each in unassigned:
        if each[0] == minnum:
            compare.append(each)
    if len(compare)<2:
        nextVar = compare[0][1]
##        if nextVar == None:
##            print("None test1")
        return nextVar
    
    maximum = -float("inf")
    for each in compare:
        var = each[1]
        var_num = 0
        constraints = [c for c in csp.binaryConstraints if c.affects(var)]
        for c in constraints:
            if not assignment.isAssigned(c.otherVariable(var)):
                var_num += 1
        if maximum < var_num:
            maximum = var_num
            nextVar = var
##    if nextVar == None:
##        print("None test1")
    return nextVar


"""
    Trivial method for ordering values to assign.
    Uses no heuristics.
"""
def orderValues(assignment, csp, var):
    return list(assignment.varDomains[var])


"""
    Creates an ordered list of the remaining values left for a given variable.
    Values should be attempted in the order returned.
    The least constraining value should be at the front of the list.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable to be assigned the values
    Returns:
        list<values>
        a list of the possible values ordered by the least constraining value heuristic
"""
def leastConstrainingValuesHeuristic(assignment, csp, var):
    """Question 3"""
    choices = list(assignment.varDomains[var])
    constraints = [c for c in csp.binaryConstraints if c.affects(var)]
    affected = [c.otherVariable(var) for c in csp.binaryConstraints if c.affects(var)]
##    print('********')
##    print(var)
##    print(choices)
##    print(constraints)
##    print(affected)
##    print('********')
    changed_num = [0 for c in choices]
    index = 0
    for c in choices:
        for a in affected:
            a_choices = assignment.varDomains[a]
            for a_c in a_choices:
##                print(csp.binaryConstraints)
##                print("var:\n")
##                print(var)
##                print("affected:\n")
##                print(affected)
                constraint = [cons for cons in constraints if cons.affects(a)]
                constraint = constraint[0]
##                print("constraint:\n")
##                print(constraint)
                if not constraint.isSatisfied(c,a_c):
                    changed_num[index]+=1
        index += 1
    pair = []
##    print(choices)
    for i in range(len(choices)):
        pair.append((changed_num[i], choices[i]))
    pair.sort() 
    result =[]
##    print("pair:",pair)
    for i in range(len(pair)):
        result.append(pair[i][1])
##    print("LCV:",result)
    return result


"""
    Trivial method for making no inferences.
"""
def noInferences(assignment, csp, var, value):
    return set([])


"""
    Implements the forward checking algorithm.
    Each inference should take the form of (variable, value) where the value is being removed from the
    domain of variable. This format is important so that the inferences can be reversed if they
    result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
    inferences made should be reversed before ending the fuction.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable that has just been assigned a value
        value (string): the value that has just been assigned
    Returns:
        set<tuple<variable, value>>
        the inferences made in this call or None if inconsistent assignment
"""
def forwardChecking(assignment, csp, var, value):
    inferences = set([])
##    print("trying to forward:",var,value)
    """Question 4"""
    if not consistent(assignment, csp, var, value):
        return None
    constraints = [c for c in csp.binaryConstraints if c.affects(var)]
    for i in range(len(constraints)):
        constraint = constraints[i]
        neighbour = constraint.otherVariable(var)
        if assignment.isAssigned(neighbour):
            continue
        removed = 0
        for neighbour_val in assignment.varDomains[neighbour]:
            if not constraint.isSatisfied(value, neighbour_val):
                inferences.add((neighbour,neighbour_val))
                removed += 1
            if removed == len(assignment.varDomains[neighbour]):
                return None
    for each in inferences:
        assignment.varDomains[each[0]].remove(each[1])
##    print("Domain:",assignment.varDomains[var])

    return inferences

"""
    Helper funciton to maintainArcConsistency and AC3.
    Remove values from var2 domain if constraint cannot be satisfied.
    Each inference should take the form of (variable, value) where the value is being removed from the
    domain of variable. This format is important so that the inferences can be reversed if they
    result in a conflicting partial assignment. If the algorithm reveals an inconsistency, any
    inferences made should be reversed before ending the fuction.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var1 (string): the variable with consistent values
        var2 (string): the variable that should have inconsistent values removed
        constraint (BinaryConstraint): the constraint connecting var1 and var2
    Returns:
        set<tuple<variable, value>>
        the inferences made in this call or None if inconsistent assignment
"""
def revise(assignment, csp, var1, var2, constraint):
    inferences = set([])
##    print("------------revising:",var1,var2,constraint)
    """Question 5"""
    revised = False
    constraints = [c for c in csp.binaryConstraints if (c.affects(var1) and c.affects(var2))]
    constraint = constraints[0]
    for x in assignment.varDomains[var1]:
        exist_proper_y = False
        for y in assignment.varDomains[var2]:
            if constraint.isSatisfied(x,y):
                exist_proper_y = True
                break
        if not exist_proper_y:
            inferences.add((var1,x))
            revised = True
    revise_num = len(inferences)
    domain_num = len(assignment.varDomains[var1])
    if revise_num == domain_num:
        return None
##    print("---finish revising:", inferences)
##    if len(inferences)==len(assignment.varDomains[var1]):
##        return None
##    if not revised:
##        return None
##    print("domain:",assignment.varDomains[var1])
    for each in inferences:
        assignment.varDomains[each[0]].remove(each[1])
##        print("********revise from domain:",each[0],each[1])
    return inferences


"""
    Implements the maintaining arc consistency algorithm.
    Inferences take the form of (variable, value) where the value is being removed from the
    domain of variable. This format is important so that the inferences can be reversed if they
    result in a conflicting partial assignment. If the algorithm reveals an inconsistency, and
    inferences made should be reversed before ending the fuction.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable that has just been assigned a value
        value (string): the value that has just been assigned
    Returns:
        set<<variable, value>>
        the inferences made in this call or None if inconsistent assignment
"""
def maintainArcConsistency(assignment, csp, var, value):
    inferences = set([])
    domains = assignment.varDomains
    """Hint: implement revise first and use it as a helper function"""
    """Question 5"""
    queue = Queue()
    constraints = [c for c in csp.binaryConstraints if c.affects(var)]
    affected = [c.otherVariable(var) for c in constraints]
    for each in constraints:
        if not assignment.isAssigned(each.otherVariable(var)):
            queue.push((var, each.otherVariable(var)))
    while not queue.isEmpty():
        (Xi,Xj) = queue.pop()
        temp_list = [c for c in csp.binaryConstraints if (c.affects(Xi) and c.affects(Xj))]
        c_Xi_Xj = temp_list[0]
        if not (c_Xi_Xj.affects(Xi) and c_Xi_Xj.affects(Xj)):
            print("asdkjlasjdlkj")
        revise_set = revise(assignment,csp,Xj,Xi,c_Xi_Xj)
        if revise_set == None:
            for each in inferences:
                assignment.varDomains[each[0]].add(each[1])
            return None
        if len(revise_set):
            revise_constraints = [c for c in csp.binaryConstraints if c.affects(Xj)]
            for each in revise_set:
                inferences.add(each)
            for each in revise_constraints:
                queue.push((Xj, each.otherVariable(Xj)))
    return inferences


"""
    AC3 algorithm for constraint propogation. Used as a preprocessing step to reduce the problem
    before running recursive backtracking.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
    Returns:
        Assignment
        the updated assignment after inferences are made or None if an inconsistent assignment
"""
def AC3(assignment, csp):
    inferences = set([])
    """Hint: implement revise first and use it as a helper function"""
    """Question 6"""
    queue = Queue()
    constraints = list(csp.binaryConstraints)
    for var in csp.varDomains.keys():
        arcs = [c for c in constraints if c.affects(var)]
        for each in arcs:
            queue.push((var, each.otherVariable(var)))
    while not queue.isEmpty():
        (Xi,Xj) = queue.pop()
        temp_list = [c for c in csp.binaryConstraints if (c.affects(Xi) and c.affects(Xj))]
        c_Xi_Xj = temp_list[0]
        revise_set = revise(assignment,csp,Xj,Xi,c_Xi_Xj)
        if revise_set == None:
            for each in inferences:
                assignment.varDomains[each[0]].add(each[1])
            return None
        if len(revise_set):
            revise_constraints = [c for c in csp.binaryConstraints if c.affects(Xj)]
##            for each in revise_set:
##                inferences.add(each)
            for each in revise_constraints:
                queue.push((Xj, each.otherVariable(Xj)))            
    return assignment

"""
    Solves a binary constraint satisfaction problem.

    Args:
        csp (ConstraintSatisfactionProblem): a CSP to be solved
        orderValuesMethod (function): a function to decide the next value to try
        selectVariableMethod (function): a function to decide which variable to assign next
        inferenceMethod (function): a function to specify what type of inferences to use
        useAC3 (boolean): specifies whether to use the AC3 preprocessing step or not
    Returns:
        dictionary<string, value>
        A map from variables to their assigned values. None if no solution exists.
"""
def solve(csp, orderValuesMethod=leastConstrainingValuesHeuristic, selectVariableMethod=minimumRemainingValuesHeuristic, inferenceMethod=forwardChecking, useAC3=True):
    assignment = Assignment(csp)

    assignment = eliminateUnaryConstraints(assignment, csp)
    if assignment == None:
        return assignment
    
    if useAC3:
        assignment = AC3(assignment, csp)
        if assignment == None:
            return assignment
##    print(assignment.varDomains)
##    print("assigned:")
##    print(assignment.assignedValues)
    assignment = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
    if assignment == None:
        return assignment
##    print("place3")
##    print(assignment)
    return assignment.extractSolution()

