"""
Quantum Sparse Interations in Compact Encoding (QuSpICE)

Compiles quantum circuits to implement Hamiltonian oracles for general field theoretic interactions in compact encoding.

Copyright 2022 William M. Kirby

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import math
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import itertools as it
from copy import deepcopy
from functools import reduce


"""
`FockState(input_state, dim, momentum_cutoffs, occ_cutoff)`: class; provides a template for the form of a Fock state in the compact encoding.

Input:
- `dim` is the spatial dimension;
- `momentum_cutoffs` is a list of pairs specifying single-mode momentum cutoffs (upper and lower bounds) for each dimension;
- `occ_cutoff` is the cutoff on the occupation of any single mode (which may be specified naturally by the theory, as in the light-front formulation);
- `input_state` should be a list whose elements represent the modes present in the state.
Each element of `input_state` should have the form:
	[ <occupation>, <quantum numbers>, <momenta> ],

where...
- <occupation> is at least 1 (and no more than 1 if the mode is fermionic or antifermionic);
- <quantum numbers> is a list of the intrinsic quantum numbers, the first element of which should be 0 or 1 indicating fermion or antifermion,
    and the second element of which should be 0 or 1, indicating particle or antiparticle;
- <momenta> is a list of the momenta.

All modes must be distinct.
"""

class FockState:
    
    _dim = 0 # spatial dimensionality
    _cutoffs = [] # momentum cutoffs
    _fock_state = [] # state
    _p = [] # momentum
    _modes = 0 # number of distinct occupied modes
    _occ = 0 # total occupation
    _occ_cutoff = 0 # cutoff on allowed occupation of a single-mode (may be determined naturally, as in front-form)
    
    _q_state = [] # state encoded in qubits (list of modes, which are lists of bitstrings)
    _q_count = 0 # total number of qubits
    _q_per_mode = 0 # number of qubits to encode a mode
    _q_per_n = [] # number of qubits to encode each momentum
    _q_per_qn = 0 # number of qubits to encode quantum numbers
    _q_per_occ = 0 # number of qubits to encode an occupation
    
    def __init__(self,input_state,dim,momentum_cutoffs,occ_cutoff):
        
        # Check well-formed input
        assert(occ_cutoff >= 0)
        assert(len(momentum_cutoffs) == dim)
        assert(all(len(i) == 2 for i in momentum_cutoffs))
        assert(len(input_state) >= 0)
        assert(all(len(i) == 3 for i in input_state)) # elements of input_state have length 3
        assert(all(i[0] > 0 and i[0] <= occ_cutoff for i in input_state)) # occupation numbers are positive and bounded by occ_cutoff
        assert(all((i[0] == 1 or i[1][0] == 0) for i in input_state)) # occupation numbers of fermions and antifermions are 1
        assert(all(len(i[2]) == dim for i in input_state)) # dimensionality of momenta
        assert(all(all(i[2][j] >= momentum_cutoffs[j][0] and i[2][j] <= momentum_cutoffs[j][1] for j in range(dim)) for i in input_state)) # momenta fall within cutoffs
        assert(all(all((input_state[i][1:] != input_state[j][1:]) or (i == j) for j in range(len(input_state))) for i in range(len(input_state)))) # no copies of the same mode
        
        # Initialize
        self._fock_state = input_state
        self._dim = dim
        self._cutoffs = momentum_cutoffs
        self._occ_cutoff = occ_cutoff
        
        self._p = list(sum([i[0]*i[2][j] for i in input_state]) for j in range(dim)) # total momenta
        self._modes = len(input_state) # number of distinct occupied modes
        self._occ = sum([i[0] for i in input_state]) # total occupation
        
        # Order Fock state by intrinsic quantum numbers, then by momentum
        self._fock_state.sort(key = lambda x: x[2])
        self._fock_state.sort(key = lambda x: x[1])
        
        # Qubit numbers
        self._q_per_n = list(int(np.ceil(math.log(self._cutoffs[j][1]-self._cutoffs[j][0]+1,2))) for j in range(dim))
        self._q_per_qn = 2 # will be larger if more quantum numbers are specified than just boson/fermion and particle/antiparticle
        self._q_per_occ = int(np.ceil(math.log(self._occ_cutoff+1,2)))
        self._q_per_mode = sum(i for i in self._q_per_n) + self._q_per_qn + self._q_per_occ
        self._q_count = self._modes * self._q_per_mode
    
    
    def __str__(self):
        the_string = ''
        for m in self._fock_state:
            if m[1][1] == 1:
                the_string += 'anti'
            if m[1][0] == 0:
                the_string += 'boson '
            if m[1][0] == 1:
                the_string += 'fermion '
            the_string += str(m[2]) + ', occ = ' + str(m[0]) + '\n'
        return the_string


"""
`Interaction(f, g, qn_in, qn_out, dimensions, momentum_cutoffs)`: class.

Represents an interaction as specified by incoming and outgoing particles, agnostic to momenta.

Input:
- `f` is the total number of particles in the interaction.
- `g` is the total number of outgoing particles in the interaction (the number of creation operators).
- `qn_in` and `qn_out` are lists of the intrinsic quantum numbers of the (`f-g`) incoming and (`g`) outgoing particles, respectively.
    Each element of `qn_in` or `qn_out` should itself be a dict of the form:
        { <name of quantum number>:<value>, <name of another quantum number>:<another value>,... }
- `dimensions` is the number of spatial dimensions.
- `momentum_cutoffs` is a list of the momentum cutoffs (specified as pairs giving the upper and lower bounds) for each dimension.
"""

class Interaction:
    
    _f = 0 # number of external lines
    _g = 0 # number of outgoing particles
    _qn_in = [] # incoming quantum numbers
    _qn_out = [] # outgoing quantum numbers
    _qn_in_names = [] # names of incoming quantum numbers
    _qn_out_names = [] #names of outgoing quantum numbers
    _dim = 0 # number of spatial dimensions
    _cutoffs = [] # momentum cutoffs
    
    def __init__(self,f,g,qn_in,qn_out,dimensions,momentum_cutoffs):
        self._f = f
        self._g = g
        self._qn_in = [list(i.values()) for i in qn_in]
        self._qn_out = [list(i.values()) for i in qn_out]
        self._qn_in_names = [list(i.keys()) for i in qn_in]
        self._qn_out_names = [list(i.keys()) for i in qn_out]
        self._dim = dimensions
        self._cutoffs = momentum_cutoffs
        
        # order incoming and outgoing particles
        self._qn_in.sort()
        self._qn_out.sort()


"""
`set_index(v,x,*indices)`: for value v and nested list x, sets the value of the entry in x indexed by indices to v.
"""
def set_index(v,x,*indices):
    assert(isinstance(indices,tuple))
    assert(len(indices) >= 1)
    assert(isinstance(x,list))
    if len(indices) == 1:
        x[indices[0]] = v
    else:
        y = x[indices[0]]
        for i in range(1,len(indices)-1):
            y = y[indices[i]]
        y[indices[-1]] = v


"""
`iszero(x)`: returns True if x is 0 or a list (possibly nested) containing only zeroes.
"""
def iszero(x):
    if isinstance(x,int):
        return x==0
    elif isinstance(x,list):
        return all(iszero(y) for y in x)
    else:
        raise ValueError('x must be an int or a list (possibly nested) of ints.')


"""
`makezero(x)`: if x is an integer, returns 0. If x is a list (possibly nested) of integers, returns the list in which all integers in x have been replaced by 0.
"""
def makezero(x):
    if isinstance(x,int):
        return 0
    elif isinstance(x,list):
        return [makezero(y) for y in x]
    else:
        raise ValueError('x must be an int or a list (possibly nested) of ints.')


"""
`QubitRegister(name,value)`: class.

Represents a single qubit or set of qubits, as either an int or a list.
If an int, should assume that it is represented as a binary number in a set of qubits.

Input:
- `name`, a string defining the name of the register.
- `value`, an integer or list (could contain sublists) specifying the initial value(s) of the register.

Methods:
- `__str__()`: returns a string representation of the register. For example, `QubitRegister('x').get(1,2,3).__str__()` returns `'x[1][2][3]'`.
- `get(*indices)`: returns the subregister specified by `indices` as a `QubitRegister`.
- `update(*indices,new_value)`: sets the value of the subregister specified by `indices` to `new_value`.
- `value()`: returns the value of the register.
"""

class QubitRegister:
    
    _name = ''
    _value = []
    
    def __init__(self,name,value):
        
        assert(isinstance(name,str))
        assert(isinstance(value,int) or isinstance(value,list))
        
        self._name = name
        self._value = value
        
        
    def __str__(self):
        return self._name
        
        
    def get(self,*indices):
        if indices:
            assert (isinstance(indices[-1],int) or (isinstance(indices[-1],list) and len(indices[-1])==2)), ValueError('final index must be an int or a list of length 2 specifying a slice: indices = '+str(indices))
            
            try:
                if isinstance(indices[-1],int):
                    sub_name = self._name + reduce(lambda x, y: x + y,['['+str(i)+']' for i in indices])
                    sub_value = self._value[indices[0]]
                    for i in indices[1:]:
                        sub_value = sub_value[i]
                    return QubitRegister(sub_name,sub_value)
            
                elif indices[:-1]:
                    sub_name = self._name + reduce(lambda x, y: x + y,['['+str(i)+']' for i in indices[:-1]]) + '['+str(indices[-1][0])+':'+str(indices[-1][1])+']'
                    sub_value = self._value[indices[0]]
                    for i in indices[1:-1]:
                        sub_value = sub_value[i]
                    sub_value = sub_value[indices[-1][0]:indices[-1][1]]
                    return QubitRegister(sub_name,sub_value)
            
                else:
                    sub_name = self._name + '['+str(indices[-1][0])+':'+str(indices[-1][1])+']'
                    sub_value = self._value[indices[-1][0]:indices[-1][1]]
                    return QubitRegister(sub_name,sub_value)
            except:
                print(indices,self._value)
                assert 1==0
                
        else:
            return self
        
        
    def update(self,*indices,new_value):
        assert (isinstance(new_value,int) or isinstance(new_value,list)), ValueError('new_value must be an int or a list: new_value = '+str(new_value))
        if indices:
            set_index(new_value,self._value,*indices)
        else:
            self._value = new_value
    
    
    def value(self):
        return self._value


"""
`QuantumOperation(action, action_reg, [value, controls])`: class.

Represents a quantum operation implementing `action` on some QubitRegister `action_reg`, possibly involving some other `value`, and possibly controlled.
Provides methods for returning an abstract representation of the operation, and for returning the value of `action_reg` after the operation is implemented.

Input:
- `action_reg`, the register that the operation acts on. Should have the form `x.get(*i)`, where `x` is some `QubitRegister`, and `*i` is a tuple of indices.
- `action`, the operation to be performed on `action_reg`. Valid actions are:
    - `'bitflip'`
    - `'binary add'`
    - `'binary subtract'`
    - `'pairwise CNOT'`
    - `'swap'`
- `value`, an optional integer, float, or QubitRegister whose value is used in the operation on `action_reg`. If a register, should have the form `x.get(*i)`.
- `controls`, an optional list of controls on the operation: list elements should have the form
    [ <control 1>, <control condition>, <control 2> ],
    where <control 1> and <control 2> are integers, floats, or registers in the form `x.get(*i)`.
    Currently available control conditions are '==', '!=', '>', '<', '>=', and '<='.

Methods:
- `__str__()`: returns a string giving an abstract representation of the operation.
- `simulate()': returns the final value of `action_reg`.
"""

class QuantumOperation:
    
    _action = ''
    _action_reg = ''
    _value = None
    _controls = []
    
    def __init__(self,action,action_reg,value=None,controls=None):
        
        assert(
            action == 'bitflip' or
            action == 'binary add' or
            action == 'binary subtract' or
            action == 'pairwise CNOT' or
            action == 'swap' or
            action == 'compute matrix element'
        )
        
        self._action = action
        self._action_reg = action_reg
        self._value = value
        self._controls = controls
        
    def __str__(self):
        
        if self._action == 'bitflip':
            temp_str = 'bitflip ' + self._action_reg.__str__()
            
        if self._action == 'binary add':
            temp_str = 'binary add ' + str(self._value) + ' to ' + self._action_reg.__str__()
            
        if self._action == 'binary subtract':
            temp_str = 'binary subtract ' + str(self._value) + ' from ' + self._action_reg.__str__()
            
        if self._action == 'pairwise CNOT':
            temp_str = 'pairwise CNOT from ' + str(self._value) + ' on ' + self._action_reg.__str__()
            
        if self._action == 'swap':
            temp_str = 'swap ' + self._action_reg.__str__() + ' and ' + str(self._value)
        
        if self._action == 'compute matrix element':
            temp_str = 'compute matrix element in register ' + self._action_reg.__str__() + ', using parameters ' + str(self._value)
        
        if self._controls:
            temp_str += ', controlled on '
            for c in self._controls:
                temp_str += c[0].__str__() + ' ' + c[1] + ' ' + c[2].__str__() + ', '
            temp_str = temp_str[:-2]
            
        return temp_str

    
    # Returns the value of self._action_reg after the operation has been executed
    def simulate(self):
        
        if isinstance(self._action_reg,int) or isinstance(self._action_reg,list):
            print(self._action_reg)
            assert(1==0)
        
        if self._controls:
            
            go = True
            
            # check controls
            for c in self._controls:
                
                if go:
                    # cast condition arguments as either ints or lists
                    if isinstance(c[0],int) or isinstance(c[0],list):
                        c0 = c[0]
                    else:
                        c0 = c[0].value()
                
                    if isinstance(c[2],int) or isinstance(c[2],list):
                        c2 = c[2]
                    else:
                        c2 = c[2].value()
                        
                    # check control conditions
                    if not ((c[1] == '==' and c0 == c2) or (c[1] == '!=' and c0 != c2) or (c[1] == '>=' and c0 >= c2) or (c[1] == '<=' and c0 <= c2) or (c[1] == '>' and c0 > c2) or (c[1] == '<' and c0 < c2)):
                        go = False
                        
             
            # execute operation if all controls are satisfied
            if go:
                if self._action == 'bitflip':
                    assert(self._action_reg.value() == 0 or self._action_reg.value() == 1)
                    return int(np.mod(self._action_reg.value() + 1, 2))
            
                if self._action == 'binary add':
                    if isinstance(self._value,int):
                        try:
                            return int(self._action_reg.value() + self._value)
                        except:
                            print(self._action_reg._name,self._action_reg.value())
                            print(self._value)
                            assert(1==0)
                    else:
                        return int(self._action_reg.value() + self._value.value())
        
                if self._action == 'binary subtract':
                    if isinstance(self._value,int):
                        return int(self._action_reg.value() - self._value)
                    else:
                        return int(self._action_reg.value() - self._value.value())
                
                # Currently only works if the target register is either 0 (fiducial state), identical to the value register, or a list (possibly nested) containing only zeroes.
                if self._action == 'pairwise CNOT':
                    
                    if isinstance(self._value, int) or isinstance(self._value, list):
                        val = self._value
                    else:
                        val = self._value.value()
                
                    if iszero(self._action_reg.value()):
                        return val
                    elif iszero(val):
                        return self._action_reg.value()
                    elif self._action_reg.value() == val:
                        return makezero(val)
                    else:
                        raise ValueError('Target register must either be 0, a list (possibly nested) of 0s, or identical to value register.')
                        
                if self._action == 'swap':
                    assert isinstance(self._value,QubitRegister), ValueError('Value register must be a QubitRegister')
                    return self._value.value(), self._action_reg.value()
                
                if self._action == 'compute matrix element':
                    assert isinstance(self._value,tuple), ValueError('Value register must be pair of QubitRegisters')
                    assert len(self._value) == 2, ValueError('Value register must be pair of QubitRegisters')
                    assert isinstance(self._value[0],QubitRegister), ValueError('Value register must be pair of QubitRegisters')
                    assert isinstance(self._value[1],QubitRegister), ValueError('Value register must be pair of QubitRegisters')
                    return [self._value[0].value(), self._value[1].value()]
                
            else:
                if self._action == 'swap':
                    return self._action_reg.value(), self._value.value()
                else:
                    return self._action_reg.value()
                
        else: # no controls
            if self._action == 'bitflip':
                assert(self._action_reg.value() == 0 or self._action_reg.value() == 1)
                return int(np.mod(self._action_reg.value() + 1, 2))
            
            if self._action == 'binary add':
                if isinstance(self._value,int):
                    try:
                        return int(self._action_reg.value() + self._value)
                    except:
                        print(self._action_reg._name,self._action_reg.value())
                        print(self._value)
                        assert(1==0)
                else:
                    return int(self._action_reg.value() + self._value.value())
        
            if self._action == 'binary subtract':
                if isinstance(self._value,int):
                    return int(self._action_reg.value() - self._value)
                else:
                    return int(self._action_reg.value() - self._value.value())
                
            # Currently only works if the target register is either 0 (fiducial state), identical to the value register, or a list containing only zeroes.
            if self._action == 'pairwise CNOT':
                
                if isinstance(self._value, int) or isinstance(self._value, list):
                    val = self._value
                else:
                    val = self._value.value()
                
                if iszero(self._action_reg.value()):
                    return val
                elif iszero(val):
                    return self._action_reg.value()
                elif self._action_reg.value() == val:
                    return makezero(val)
                else:
                    raise ValueError('Target register must either be 0, a list (possibly nested) of 0s, or identical to value register.')
                        
            if self._action == 'swap':
                assert isinstance(self._value,QubitRegister), ValueError('Value register must be a QubitRegister')
                return self._value.value(), self._action_reg.value()
            
            if self._action == 'compute matrix element':
                assert isinstance(self._value,tuple), ValueError('Value register must be pair of QubitRegisters')
                assert len(self._value) == 2, ValueError('Value register must be pair of QubitRegisters')
                assert isinstance(self._value[0],QubitRegister), ValueError('Value register must be pair of QubitRegisters')
                assert isinstance(self._value[1],QubitRegister), ValueError('Value register must be pair of QubitRegisters')
                return [self._value[0].value(), self._value[1].value()]


"""
`list_diagrams(interaction,mode_registers)`: function.
	
Input:
- `interaction`, an `Interaction`;
- `mode_registers`, the number of mode registers per encoded Fock state.
	
Output: a list of the possible interaction diagrams acting on a state encoded in `mode_registers` mode registers, before assigning outgoing momenta.
Each diagram in the output is specified as a list of indices indicating the mode registers in the input state from which particles are to be removed.
The quantum numbers of the outgoing particles are specified by the interaction, up to assigning outgoing momenta (which will happen later).
"""

def list_diagrams(interaction,mode_registers):
    
    f = interaction._f
    g = interaction._g
    qn_in = interaction._qn_in
    
    diagrams = [list(d) for d in it.combinations_with_replacement(range(mode_registers), f-g)]
    
    return diagrams


"""
`outgoing_momentum_assignments(interaction)`: function.
	
Input:
- `interaction`, an `Interaction`.
	
Output: a dict whose values are assignments of momenta to the outgoing particles in `interaction`. The keys have the form

	(<index>,<incoming total momenta>),

where <index> is the index of the desired outgoing state, and <incoming total momenta> is a tuple containing the total transferred momenta for each dimension (from the incoming state).

This output dict represents a classical lookup table that will be encoded as a quantum controlled-operation: it has polynomial size in the momentum cutoffs.
"""

def outgoing_momentum_assignments(interaction):
    
    h = interaction._f-interaction._g # number of incoming particles
    g = interaction._g # number of outgoing particles
    ranges = interaction._cutoffs # allowed momentum ranges for each dimension
    
    ranges = [[r[0],r[1]+1] for r in ranges] # change upper bounds from inclusive to exclusive
    
    assignments_by_totals = {}
    
    if h > 0:
        for a in it.product(it.product(*[range(*r) for r in ranges]),repeat=g):
            totals = [sum(a[i][j] for i in range(g)) for j in range(interaction._dim)]
            if all((totals[i] >= h*ranges[i][0] and totals[i] <= h*(ranges[i][1]-1)) for i in range(interaction._dim)):
                totals = tuple(totals)
                if totals in assignments_by_totals.keys():
                    assignments_by_totals[totals] = assignments_by_totals[totals] + [a]
                else:
                    assignments_by_totals[totals] = [a]

        # Remove all assignments that have bad orderings for particles of the same type:
        for t in assignments_by_totals.keys():
            for a in assignments_by_totals[t]:
                for i,j in it.combinations(range(len(a)),2):
                    if interaction._qn_out[i] == interaction._qn_out[j] and a[i] > a[j]:
                        if a in assignments_by_totals[t]:
                            assignments_by_totals[t].remove(a)
    
        quantities_by_totals = []
        assignments_by_index = {}
    
        for t in assignments_by_totals.keys():
            quantities_by_totals.append(len(assignments_by_totals[t]))
            for i in range(len(assignments_by_totals[t])):
                key = tuple([i,t])
                assignments_by_index[key] = [list(a) for a in assignments_by_totals[t][i]]
    
        if quantities_by_totals:
            return assignments_by_index,max(quantities_by_totals)
        else:
            return assignments_by_index,0
    
    else: # if zero incoming particles
        # print('mark2')
        # print(len([a for a in it.product(it.product(*[range(*r) for r in ranges]),repeat=g-1)]))
        assignments = []
        for a in it.product(it.product(*[range(*r) for r in ranges]),repeat=g-1):
            totals = tuple([-sum(a[i][j] for i in range(g-1)) for j in range(interaction._dim)])
            if all((totals[i] >= ranges[i][0] and totals[i] <= (ranges[i][1]-1)) for i in range(interaction._dim)):
                b = a + (totals,)
                if all([(interaction._qn_out[i] != interaction._qn_out[j] or b[i] <= b[j]) for i,j in it.combinations(range(g),2)]):
                    assignments.append(b)
            # # total transferred momentum must be zero...
            # minus_totals = [-sum(a[i][j] for i in range(g-1)) for j in range(interaction._dim)]
            # totals = [0 for i in range(interaction._dim)]
            # if all((minus_totals[i] >= ranges[i][0] and minus_totals[i] <= (ranges[i][1]-1)) for i in range(interaction._dim)):
            #     totals = tuple(totals)
            #     if totals in assignments_by_totals.keys():
            #         assignments_by_totals[totals] = assignments_by_totals[totals] + [a + (minus_totals,)]
            #     else:
            #         assignments_by_totals[totals] = [a + (minus_totals,)]
        
        # print('mark3')

        # # Remove all assignments that have bad orderings for particles of the same type:
        # for a in assignments:
        #     for i,j in it.combinations(range(len(a)),2):
        #         if interaction._qn_out[i] == interaction._qn_out[j] and a[i] > a[j]:
        #             if a in assignments:
        #                 assignments.remove(a)

        # print('mark4')
    
        quantities_by_totals = []
        assignments_by_index = {}

        t = [0 for j in range(interaction._dim)]
        t = tuple(t)

        quantities_by_totals.append(len(assignments))
        for i in range(len(assignments)):
            key = tuple([i,t])
            assignments_by_index[key] = [list(a) for a in assignments[i]]
    
        if quantities_by_totals:
            return assignments_by_index,max(quantities_by_totals)
        else:
            return assignments_by_index,0


"""
`enumerator_circuit(interaction,mode_registers,[input_state],[ind])`: function.

Input:
- `interaction`, an `Interaction`;
- `mode_registers`, the number of mode registers per encoded Fock state;
- `input_state` (optional), a `FockState` representing the incoming state;
- `ind` (optional), the index of the outgoing state to be obtained.

`input_state` and `ind` must be either both specified or both not specified.

Output: if `input_state` and `ind` are specified, returns `circuit, output_state, matrix_element, flag`,
    where `circuit` is a list of `QuantumOperation`s representing the quantum circuit,
    `output_state` is a `QubitRegister` encoding the `output_state` indexed by `ind` (if `flag` is zero),
    `matrix_element` is a `QubitRegister` encoding the list of incoming and outgoing particles of the interaction, as well as the sign (from fermion and antifermion anticommutation),
    and `flag` is a `QubitRegister` whose value is zero if `ind` indexes a valid interaction to be applied to `input_state`, and strictly positive otherwise.
    
    If `flag` is nonzero, `output_state` is returned in its initial fiducial state (which is all zeroes), and `incoming`, `outgoing` are as well.
    
    Entries in `incoming` and `outgoing` have the form `[occ, qn, mom]`, where `qn` and `mom` are the quantum numbers and momentum of the particle, respectively,
    and `occ` is either the occupation of the mode in `input_state` before removing the particle (for `incoming`), or the occupation of the mode in `output_state` after adding the particle (for `outgoing`).

If `input_state` and `ind` are not specified, then returns only `circuit`.
"""

def enumerator_circuit(interaction,mode_registers,input_state=None,ind=None):
    
    gatecount = 0
    
    # set/check dimensions and momentum cutoffs
    dim = interaction._dim
    cutoffs = interaction._cutoffs

    qn_in = interaction._qn_in
    qn_out = interaction._qn_out

    # check number of intrinsic quantum numbers
    if qn_in:
        qn_dim = len(qn_in[0])
    else:
        qn_dim = len(qn_out[0])
    
    if input_state:
        assert ind != None
        assert input_state._dim == dim
        assert input_state._cutoffs == cutoffs
    
    circuit = []
    
    # Variables representing quantum registers:
    if input_state:
        state1_init = input_state._fock_state
        state1_init += [[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]] for i in range(mode_registers - len(state1_init))]
        i0_init = ind
    else:
        state1_init = [[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]] for i in range(mode_registers)]
        i0_init = 0
        
    # inputs
    state1 = QubitRegister('state1',state1_init)
    i0 = QubitRegister('i0',i0_init)
    
    # ancillas
    i1 = QubitRegister('i1',0)
    i2 = QubitRegister('i2',0)
    delta = QubitRegister('delta',[0 for i in range(interaction._f-interaction._g)]) # locations of incoming particles in state1
    delta_out = QubitRegister('delta_out',[0 for i in range(interaction._g)]) # locations of outgoing particles in state2
    Q = QubitRegister('Q',[0 for j in range(dim)]) # 
    incoming = QubitRegister('incoming',[[0,interaction._qn_in[i],[0 for j in range(dim)]] for i in range(interaction._f-interaction._g)])
    outgoing_momenta = QubitRegister('outgoing_momenta',[[0 for j in range(dim)] for i in range(interaction._g)])
    outgoing = QubitRegister('outgoing',[[0,interaction._qn_out[i],[0 for j in range(dim)]] for i in range(interaction._g)])
    num_removed = QubitRegister('num_removed',[0 for j in range(mode_registers)])
    num_added = QubitRegister('num_added',[0 for i in range(mode_registers)])
    emptied = QubitRegister('emptied',[-1 for j in range(interaction._f-interaction._g)])
    emptied_rectified = QubitRegister('emptied_rectified',[0 for j in range(interaction._f-interaction._g)])
    added = QubitRegister('added',0)
    matched1 = QubitRegister('matched1',[0 for i in range(mode_registers)])
    matched2 = QubitRegister('matched2',[0 for i in range(mode_registers)])
    flag = QubitRegister('flag',0)
    
    # outputs
    matrix_element = QubitRegister('matrix_element',[0,0,0])
    state2 = QubitRegister('state2',[[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]] for i in range(mode_registers)])
    
    # classical preprocessing:
    diagrams = list_diagrams(interaction,mode_registers)
    num_diagrams = len(diagrams)
#     print(num_diagrams,'\n')
    assignments, num_assignments = outgoing_momentum_assignments(interaction)
#     print('num_diagrams',num_diagrams)
#     print('num_assignments',num_assignments,'\n')
    # print('assignments',assignments,'\n')
    
    if input_state:
        assert ind < num_diagrams*num_assignments, ValueError('ind is greater than maximum sparsity')
    
    # quantum operations:
    
    # compute sub-indices i1 and i2: i1 = i0 mod num_diagrams, and i2 = floor(i0/num_diagrams).
    op = QuantumOperation(
        'binary add',
        i1.get(),
        value = i0.get()
    )
    circuit.append(op)
    i1.update(new_value = op.simulate())
    
    for i in range(num_assignments): # i will be the value of i2...
        op = QuantumOperation(
            'binary subtract',
            i1.get(),
            value = i*num_diagrams,
            controls = [
                [i0.get(),'>=',i*num_diagrams],
                [i0.get(),'<',(i+1)*num_diagrams]
            ]
        )
        circuit.append(op)
        i1.update(new_value = op.simulate())
    
        op = QuantumOperation(
            'binary add',
            i2.get(),
            value = i,
            controls = [
                [i0.get(),'>=',i*num_diagrams],
                [i0.get(),'<',(i+1)*num_diagrams]
            ]
        )
        circuit.append(op)
        i2.update(new_value = op.simulate())
    
    # compute the entries in delta, list of indices of modes from which the incoming particles will be removed
    for i in range(num_diagrams): # will be the value of i1
            
        op = QuantumOperation(
            'pairwise CNOT',
            delta.get(),
            value = diagrams[i],
            controls = [
                [i1.get(),'==',i]
            ]
        )
        circuit.append(op)
        delta.update(new_value = op.simulate())

    # copy state1 to state2
    for i in range(mode_registers):
        
        op = QuantumOperation(
            'pairwise CNOT',
            state2.get(i),
            value = state1.get(i)
        )
        circuit.append(op)
             
    state2 = QubitRegister('state2',deepcopy(state1_init))

    # print(state2,state2.value(),'\n')
    
#     print(delta,delta.value(),'\n')
    
    # Remove particles indexed by entries in delta and compute Q, the total momentum transferred:
    for i in range(interaction._f-interaction._g):
        
        for j in range(mode_registers): # will be equal to the ith entry in delta
            
            # Controlled on the (delta[i])th mode register's quantum numbers not matching the ith incoming line in interaction,
            # add 1 to the flag register, since in this case delta cannot be applied to state2.
            op = QuantumOperation(
                'binary add',
                flag.get(),
                value = 1,
                controls = [
                    [delta.get(i),'==',j],
                    [state2.get(j,1),'!=',qn_in[i]]
                ]
            )
            circuit.append(op)
            flag.update(new_value = op.simulate())

            # Update parity:
            if qn_in[i][:2] == [1,0]:
                op = QuantumOperation(
                    'bitflip',
                    matrix_element.get(2),
                    controls = [
                        [delta.get(i),'>',j],
                        [state2.get(j,1),'==',[1,0]]
                    ]
                )
                circuit.append(op)
                matrix_element.update(2,new_value = op.simulate())
            
            if qn_in[i][:2] == [1,1]:
                op = QuantumOperation(
                    'bitflip',
                    matrix_element.get(2),
                    controls = [
                        [delta.get(i),'>',j],
                        [state2.get(j,1),'==',[1,1]]
                    ]
                )
                circuit.append(op)
                matrix_element.update(2,new_value = op.simulate())
            
            # Add momentum of current mode to Q and to the current entry in incoming:
            for l in range(dim): # the components of momentum
                # add the lth component of this mode register's momentum to the lth component of the total momentum transferred.
                op = QuantumOperation(
                    'binary add',
                    Q.get(l),
                    value = state2.get(j,2,l),
                    controls = [
                        [delta.get(i),'==',j]
                    ]
                )
                circuit.append(op)
                Q.update(l,new_value = op.simulate())
                
                # add the lth component of this mode register's momentum to the lth component of the current entry in incoming.
                op = QuantumOperation(
                    'binary add',
                    incoming.get(i,2,l),
                    value = state2.get(j,2,l),
                    controls = [
                        [delta.get(i),'==',j]
                    ]
                )
                circuit.append(op)
                incoming.update(i,2,l,new_value = op.simulate())
                
            # Record the current mode's occupation in the current entry in incoming:
            op = QuantumOperation(
                'binary add',
                incoming.get(i,0),
                value = state2.get(j,0),
                controls = [
                    [delta.get(i),'==',j]
                ]
            )
            circuit.append(op)
            incoming.update(i,0,new_value = op.simulate())
            
            # subtract 1 from the occupation of the (delta[i])th mode in state2
            op = QuantumOperation(
                'binary subtract',
                state2.get(j,0),
                value = 1,
                controls = [
                    [delta.get(i),'==',j]
                ]
            )
            circuit.append(op)
            state2.update(j,0,new_value = op.simulate())
            
            # keep track of the fact that we just did that
            op = QuantumOperation(
                'binary add',
                num_removed.get(j),
                value = 1,
                controls = [
                    [delta.get(i),'==',j]
                ]
            )
            circuit.append(op)
            num_removed.update(j,new_value = op.simulate())
            
            # If this mode is now empty, set its intrinsic quantum numbers to 0
            op = QuantumOperation(
                'pairwise CNOT',
                state2.get(j,1),
                value = state1.get(j,1),
                controls = [
                    [delta.get(i),'==',j],
                    [state2.get(j,0),'==',0]
                ]
            )
            circuit.append(op)
            state2.update(j,1,new_value = op.simulate())
            
            # If this mode is now empty, set its momenta to 0
            op = QuantumOperation(
                'pairwise CNOT',
                state2.get(j,2),
                value = state1.get(j,2),
                controls = [
                    [delta.get(i),'==',j],
                    [state2.get(j,0),'==',0]
                ]
            )
            circuit.append(op)
            state2.update(j,2,new_value = op.simulate())
            
            # If this mode now has negative occupation, add 1 to the flag register
            op = QuantumOperation(
                'binary add',
                flag.get(),
                value = 1,
                controls = [
                    [delta.get(i),'==',j],
                    [state2.get(j,0),'<',0]
                ]
            )
            circuit.append(op)
            flag.update(new_value = op.simulate())
            
    # compute sub-indices i1 and i2: i1 = i0 mod num_diagrams, and i2 = floor(i0/num_diagrams).
    gatecount += 1
    gatecount += 2*num_assignments
    
    # compute the entries in delta, list of indices of modes from which the incoming particles will be removed
    gatecount += num_diagrams

    # copy state1 to state2
    gatecount += mode_registers
    
    # remove particles indexed by entries in delta and compute Q, the total momentum transferred
    gatecount += (interaction._f - interaction._g)*mode_registers*(1 + 2*dim + 6)

    # update parity
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers):
            if qn_in[i][:2] == [1,0] or qn_in[i][:2] == [1,1]:
                gatecount += 1
            
#     print('1.',len(circuit),gatecount)
    
    # compute outgoing, the full list of outgoing particles (final occupations will be computed later)
    gatecount += len(assignments.keys())*interaction._g
#     print('assignments',assignments,'\n')
    for i in assignments.keys(): # i[0] will be the value of i2, i[1] will be the value of Q
        assignment = assignments[i]
        for k in range(interaction._g): # indexes the outgoing particle whose momentum is to be input
            op = QuantumOperation(
                'pairwise CNOT',
                outgoing.get(k,2),
                value = assignment[k],
                controls = [
                    [flag.get(),'==',0],
                    [i2.get(),'==',i[0]],
                    [Q.get(),'==',list(i[1])]
                ]
            )
            circuit.append(op)
            outgoing.update(k,2,new_value = op.simulate())

    # print(outgoing,outgoing.value(),'\n')
                    
#     for i in range(num_assignments): # i will be the value of i2
        
#         for j in it.product(*[[m for m in range((interaction._f-interaction._g)*cutoffs[l][0],(interaction._f-interaction._g)*(cutoffs[l][1]+1))] for l in range(dim)]): # j will be the value of Q
            
#             # [i,j] takes O(num_assignments) values. For FrontForm in 1D, this is O(K^2).
            
#             if tuple([i,j]) in list(assignments.keys()):
#                 print(tuple([i,j]))
#                 assignment = assignments[tuple([i,j])]
#                 for k in range(interaction._g): # indexes the outgoing particle whose momentum is to be input
#                     gatecount += 1
                    
#                     op = QuantumOperation(
#                         'pairwise CNOT',
#                         outgoing.get(k,2),
#                         value = assignment[k],
#                         controls = [
#                             [flag.get(),'==',0],
#                             [i2.get(),'==',i],
#                             [Q.get(),'==',list(j)]
#                         ]
#                     )
#                     circuit.append(op)
#                     outgoing.update(k,2,new_value = op.simulate())
    
    # If outgoing is still empty, it means that this action cannot be applied to state1:
    if interaction._g > 0:
        gatecount += 1
        op = QuantumOperation(
            'binary add',
            flag.get(),
            value = 1,
            controls = [
                [outgoing.get(0),'==',[0,interaction._qn_out[0],[0 for j in range(dim)]]]
            ]
        )
        circuit.append(op)
        flag.update(new_value = op.simulate())
                    
    # Make sure the outgoing particles don't duplicate any fermions:
    gatecount += interaction._g*(interaction._g-1)/2
    gatecount += interaction._g*mode_registers
    for i in range(interaction._g):
        for j in range(i):
            # If two new fermions are identical, add 1 to flag register:
            op = QuantumOperation(
                'binary add',
                flag.get(),
                value = 1,
                controls = [
                    [outgoing.get(i,1,0),'==',1], # particle i is fermionic
                    [outgoing.get(j,1,0),'==',1], # particle j is fermionic
                    [outgoing.get(i,1),'==',outgoing.get(j,1)], # quantum numbers match
                    [outgoing.get(i,2),'==',outgoing.get(j,2)] # momenta match
                ]
            )
            circuit.append(op)
            flag.update(new_value = op.simulate())

    for i in range(interaction._g):
        for j in range(mode_registers):
            # If a new fermion matches an existing mode in state2, add 1 to flag register:
            op = QuantumOperation(
                'binary add',
                flag.get(),
                value = 1,
                controls = [
                    [outgoing.get(i,1,0),'==',1], # mode is fermionic
                    [outgoing.get(i,[1,3]),'==',state2.get(j,[1,3])] # quantum numbers and momenta match mode in state2
                ]
            )
            circuit.append(op)
            flag.update(new_value = op.simulate())
            
    # Undo the removals from state2 if flag is now nonzero
    gatecount += (interaction._f-interaction._g)*mode_registers*4
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers):
            if qn_in[i][:2] == [1,0] or qn_in[i][:2] == [1,1]:
                gatecount += 1

    for i in range(interaction._f-interaction._g-1,-1,-1):
        
        for j in range(mode_registers-1,-1,-1):
            
            op = QuantumOperation(
                'pairwise CNOT',
                state2.get(j,1),
                value = state1.get(j,1),
                controls = [
                    [delta.get(i),'==',j],
                    [state2.get(j,0),'==',0],
                    [flag.get(),'!=',0]
                ]
            )
            circuit.append(op)
            state2.update(j,1,new_value = op.simulate())
            
            op = QuantumOperation(
                'pairwise CNOT',
                state2.get(j,2),
                value = state1.get(j,2),
                controls = [
                    [delta.get(i),'==',j],
                    [state2.get(j,0),'==',0],
                    [flag.get(),'!=',0]
                ]
            )
            circuit.append(op)
            state2.update(j,2,new_value = op.simulate())
            
            op = QuantumOperation(
                'binary subtract',
                num_removed.get(j),
                value = 1,
                controls = [
                    [delta.get(i),'==',j],
                    [flag.get(),'!=',0]
                ]
            )
            circuit.append(op)
            num_removed.update(j,new_value = op.simulate())
            
            op = QuantumOperation(
                'binary add',
                state2.get(j,0),
                value = 1,
                controls = [
                    [delta.get(i),'==',j],
                    [flag.get(),'!=',0]
                ]
            )
            circuit.append(op)
            state2.update(j,0,new_value = op.simulate())

            if qn_in[i][:2] == [1,0]:
                op = QuantumOperation(
                    'bitflip',
                    matrix_element.get(2),
                    controls = [
                        [delta.get(i),'>',j],
                        [state2.get(j,1),'==',[1,0]],
                        [flag.get(),'!=',0]
                    ]
                )
                circuit.append(op)
                matrix_element.update(2,new_value = op.simulate())
            
            if qn_in[i][:2] == [1,1]:
                op = QuantumOperation(
                    'bitflip',
                    matrix_element.get(2),
                    controls = [
                        [delta.get(i),'>',j],
                        [state2.get(j,1),'==',[1,1]],
                        [flag.get(),'!=',0]
                    ]
                )
                circuit.append(op)
                matrix_element.update(2,new_value = op.simulate())
    
#     print('2.',len(circuit),gatecount)
#     print('parity',matrix_element.get(2).value(),'\n')
        
    # Compute emptied, a list of the mode registers in state2 that were emptied by the removals.
    gatecount += (interaction._f-interaction._g)*mode_registers
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers): # will be equal to the ith entry in delta
            # If this mode is now empty, put its index in the corresponding entry in emptied (the extra +1 is because each entry in emptied is initially -1).
            op = QuantumOperation(
                'binary add',
                emptied.get(i),
                value = j+1,
                controls = [
                    [delta.get(i),'==',j],
                    [state1.get(j,0),'==',num_removed.get(j)],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            emptied.update(i,new_value = op.simulate())

    # Copy emptied to emptied_rectified,
    # then from each entry in emptied_rectified, subtract 1 for each nonnegative previous entry less than the current entry.
    # This is to account for the modes corresponding to the previous entries being removed.
    gatecount += interaction._f-interaction._g
    gatecount += (interaction._f-interaction._g)*(interaction._f-interaction._g-1)/2
    for i in range(interaction._f-interaction._g):
        
        op = QuantumOperation(
            'pairwise CNOT',
            emptied_rectified.get(i),
            value = emptied.get(i),
            controls = [
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        emptied_rectified.update(i,new_value = op.simulate())
        
        for j in range(i):
            
            op = QuantumOperation(
                'binary subtract',
                emptied_rectified.get(i),
                value = 1,
                controls = [
                    [emptied.get(i),'>',emptied.get(j)],
                    [emptied.get(j),'>=',0],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            emptied_rectified.update(i,new_value = op.simulate())
    
    # Order state2:
    gatecount += (interaction._f-interaction._g)*(mode_registers-1)
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers-1):
            
            op = QuantumOperation(
                'swap',
                state2.get(j),
                value = state2.get(j+1),
                controls = [
                    [emptied_rectified.get(i),'>=',0],
                    [emptied_rectified.get(i),'<=',j],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            new_value1, new_value2 = op.simulate()
            state2.update(j,new_value = new_value1)
            state2.update(j+1,new_value = new_value2)
            
    # Uncompute emptied_rectified:
    gatecount += (interaction._f-interaction._g)*(interaction._f-interaction._g-1)/2
    gatecount += interaction._f-interaction._g
    for i in range(interaction._f-interaction._g-1,-1,-1):
        
        for j in range(i):
            
            op = QuantumOperation(
                'binary add',
                emptied_rectified.get(i),
                value = 1,
                controls = [
                    [emptied.get(i),'>',emptied.get(j)],
                    [emptied.get(j),'>=',0],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            emptied_rectified.update(i,new_value = op.simulate())
            
        op = QuantumOperation(
            'pairwise CNOT',
            emptied_rectified.get(i),
            value = emptied.get(i),
            controls = [
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        emptied_rectified.update(i,new_value = op.simulate())
        
    # Uncompute emptied:
    gatecount += (interaction._f-interaction._g)*mode_registers
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers): # will be equal to the ith entry in delta
            op = QuantumOperation(
                'binary subtract',
                emptied.get(i),
                value = j+1,
                controls = [
                    [delta.get(i),'==',j],
                    [state1.get(j,0),'==',num_removed.get(j)],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            emptied.update(i,new_value = op.simulate())

    
    # print(state2,state2.value(),'\n')

            
#     print('3.',len(circuit),gatecount)
    
    # Add outgoing particles to state2:        
    for i in range(interaction._g): # i is the index of the new particle to be added.

        # Insert the new particle if it matches a mode already in state2:
        for j in range(mode_registers):
            
            # If the new particle matches an existing bosonic mode in state2, add it and mark it as added:
            op = QuantumOperation(
                'binary add',
                state2.get(j,0),
                value = 1,
                controls = [
                    [outgoing.get(i,1,0),'==',0], # mode is bosonic
                    [outgoing.get(i,[1,3]),'==',state2.get(j,[1,3])], # quantum numbers and momenta match mode in state2
                    [state2.get(j,0),'>',0], # mode in state2 is nonempty
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            state2.update(j,0,new_value = op.simulate())
            
            # Update final occupation number of this entry in outgoing:
            op = QuantumOperation(
                'binary add',
                outgoing.get(i,0),
                value = state2.get(j,0),
                controls = [
                    [outgoing.get(i,1,0),'==',0], # mode is bosonic
                    [outgoing.get(i,[1,3]),'==',state2.get(j,[1,3])], # quantum numbers and momenta match mode in state2
                    [state2.get(j,0),'>',0], # mode in state2 is nonempty
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            outgoing.update(i,0,new_value = op.simulate())
            
            op = QuantumOperation(
                'bitflip',
                added.get(),
                controls = [
                    [outgoing.get(i,1,0),'==',0], # mode is bosonic
                    [outgoing.get(i,[1,3]),'==',state2.get(j,[1,3])], # quantum numbers and momenta match mode in state2
                    [state2.get(j,0),'>',0] # mode in state2 is nonempty
                ]
            )
            circuit.append(op)
            added.update(new_value = op.simulate())
            
        # Insert the new particle if it didn't match any mode in state2:
        for j in range(mode_registers-2,-1,-1): # j will be the index of the mode in state2 that will appear immediately before the new mode.

            # Update parity:
            op = QuantumOperation(
                'bitflip',
                matrix_element.get(2),
                controls = [
                    [flag.get(),'==',0],
                    [state2.get(j,1),'==',[1,0]], # current mode is a fermion
                    [outgoing.get(i,1),'==',[1,0]], # particle to be added is a fermion
                    [outgoing.get(i,2),'>',state2.get(j,2).value()] # current mode precedes particle to be added
                ]
            )
            circuit.append(op)
            matrix_element.update(2,new_value = op.simulate())

            op = QuantumOperation(
                'bitflip',
                matrix_element.get(2),
                controls = [
                    [flag.get(),'==',0],
                    [state2.get(j,1),'==',[1,1]], # current mode is an antifermion
                    [outgoing.get(i,1),'==',[1,1]], # particle to be added is an antifermion
                    [outgoing.get(i,2),'>',state2.get(j,2).value()] # current mode precedes particle to be added
                ]
            )
            circuit.append(op)
            matrix_element.update(2,new_value = op.simulate())
            
            # Case #1: the new mode needs to appear before mode j in state2, so need to swap modes j and j+1.
            # Swap state2[j] and state2[j+1] (which is guaranteed to be unencoded), controlled on the ith particle in outgoing having not already been added, and on it needing to be placed before the jth mode.
            op = QuantumOperation(
                'pairwise CNOT',
                state2.get(j+1),
                value = state2.get(j),
                controls = [
                    [flag.get(),'==',0],
                    [added.get(),'==',0],
                    [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                    [outgoing.get(i,[1,3]),'<',state2.get(j,[1,3])] # compare modes without occupations
                ]
            )
            circuit.append(op)
            state2.update(j+1,new_value=op.simulate())
            
            op = QuantumOperation(
                'pairwise CNOT',
                state2.get(j),
                value = state2.get(j+1),
                controls = [
                    [flag.get(),'==',0],
                    [added.get(),'==',0],
                    [state2.get(j+1),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                    [outgoing.get(i,[1,3]),'<',state2.get(j+1,[1,3])] # compare modes without occupations
                ]
            )
            circuit.append(op)
            state2.update(j,new_value=op.simulate())
            
            # Case #2: the new mode needs to appear after mode j in state2, but before mode j+2 in state2 (mode j+1 in state2 is guaranteed to be unencoded)..
            if j+2 == mode_registers: # If mode j+1 is the last mode in state2...
                # Insert the ith particle in outgoing as a new mode in the (j+1)th location in state2,
                # controlled on it having not already been added, and on the new mode needing to be placed after mode j in state2.
                
                # Set occupation of new mode:
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,0),
                    value = 1,
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])] # compare modes without occupations
                    ]
                )
                circuit.append(op)
                state2.update(j+1,0,new_value=op.simulate())
                
                # Set quantum numbers of new mode:
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,1),
                    value = outgoing.get(i,1),
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])] # compare modes without occupations
                    ]
                )
                circuit.append(op)
                state2.update(j+1,1,new_value=op.simulate())
                
                # Set momenta of new mode:
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,2),
                    value = outgoing.get(i,2),
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])] # compare modes without occupations
                    ]
                )
                circuit.append(op)
                state2.update(j+1,2,new_value=op.simulate())
                
                # Update final occupation number of this entry in outgoing:
                op = QuantumOperation(
                    'binary add',
                    outgoing.get(i,0),
                    value = 1,
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])] # compare modes without occupations
                    ]
                )
                circuit.append(op)
                outgoing.update(i,0,new_value = op.simulate())
    
            else: 
                # If mode j+1 is not the last mode in state2...
                # Insert the ith particle in outgoing as a new mode in the (j+1)th location in state2,
                # controlled on it having not already been added, on the new mode needing to be placed after mode j in state2, and on the new mode needing to be placed before mode j+2 in state2.
                
                # Set occupation of new mode (in the case when mode j+2 is unencoded):
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,0),
                    value = 1,
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])], # compare modes without occupations
                        [state2.get(j+2),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
                    ]
                )
                circuit.append(op)
                state2.update(j+1,0,new_value=op.simulate())
                
                # Set quantum numbers of new mode (in the case when mode j+2 is unencoded):
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,1),
                    value = outgoing.get(i,1),
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])], # compare modes without occupations
                        [state2.get(j+2),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
                    ]
                )
                circuit.append(op)
                state2.update(j+1,1,new_value=op.simulate())
                
                # Set momenta of new mode (in the case when mode j+2 is unencoded):
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,2),
                    value = outgoing.get(i,2),
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])], # compare modes without occupations
                        [state2.get(j+2),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
                    ]
                )
                circuit.append(op)
                state2.update(j+1,2,new_value=op.simulate())
                
                # Update final occupation number of this entry in outgoing:
                op = QuantumOperation(
                    'binary add',
                    outgoing.get(i,0),
                    value = 1,
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])], # compare modes without occupations
                        [state2.get(j+2),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
                    ]
                )
                circuit.append(op)
                outgoing.update(i,0,new_value = op.simulate())
                
                # Set occupation of new mode (in the case when mode j+2 is encoded and should appear after the new mode):
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,0),
                    value = 1,
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])],
                        [state2.get(j+2),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'<',state2.get(j+2,[1,3])]
                    ]
                )
                circuit.append(op)
                state2.update(j+1,0,new_value=op.simulate())
                
                # Set quantum numbers of new mode (in the case when mode j+2 is encoded and should appear after the new mode):
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,1),
                    value = outgoing.get(i,1),
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])],
                        [state2.get(j+2),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'<',state2.get(j+2,[1,3])]
                    ]
                )
                circuit.append(op)
                state2.update(j+1,1,new_value=op.simulate())
                
                # Set momenta of new mode (in the case when mode j+2 is encoded and should appear after the new mode):
                op = QuantumOperation(
                    'pairwise CNOT',
                    state2.get(j+1,2),
                    value = outgoing.get(i,2),
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])],
                        [state2.get(j+2),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'<',state2.get(j+2,[1,3])]
                    ]
                )
                circuit.append(op)
                state2.update(j+1,2,new_value=op.simulate())
                
                # Update final occupation number of this entry in outgoing:
                op = QuantumOperation(
                    'binary add',
                    outgoing.get(i,0),
                    value = 1,
                    controls = [
                        [flag.get(),'==',0],
                        [added.get(),'==',0],
                        [state2.get(j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'>',state2.get(j,[1,3])],
                        [state2.get(j+2),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [outgoing.get(i,[1,3]),'<',state2.get(j+2,[1,3])]
                    ]
                )
                circuit.append(op)
                outgoing.update(i,0,new_value = op.simulate())
                
        # The one remaining case is when the new mode should appear first in state2
        # Set occupation of new mode (in the case when state2 contains no other modes):
        op = QuantumOperation(
            'pairwise CNOT',
            state2.get(0,0),
            value = 1,
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [state2.get(1),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
            ]
        )
        circuit.append(op)
        state2.update(0,0,new_value=op.simulate())
                
        # Set quantum numbers of new mode (in the case when state2 contains no other modes):
        op = QuantumOperation(
            'pairwise CNOT',
            state2.get(0,1),
            value = outgoing.get(i,1),
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [state2.get(1),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
            ]
        )
        circuit.append(op)
        state2.update(0,1,new_value=op.simulate())
                
        # Set momenta of new mode (in the case when state2 contains no other modes):
        op = QuantumOperation(
            'pairwise CNOT',
            state2.get(0,2),
            value = outgoing.get(i,2),
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [state2.get(1),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
            ]
        )
        circuit.append(op)
        state2.update(0,2,new_value=op.simulate())
        
        # Update final occupation number of this entry in outgoing:
        op = QuantumOperation(
            'binary add',
            outgoing.get(i,0),
            value = 1,
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [state2.get(1),'==',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]]
            ]
        )
        circuit.append(op)
        outgoing.update(i,0,new_value = op.simulate())
        
        # Set occupation of new mode (in the case when it should be first in state2 and state2 contains other modes):
        op = QuantumOperation(
            'pairwise CNOT',
            state2.get(0,0),
            value = 1,
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [outgoing.get(i,[1,3]),'<',state2.get(1,[1,3])],
                [state2.get(1,0),'>',0]
            ]
        )
        circuit.append(op)
        state2.update(0,0,new_value=op.simulate())
                
        # Set quantum numbers of new mode (in the case when it should be first in state2 and state2 contains other modes):
        op = QuantumOperation(
            'pairwise CNOT',
            state2.get(0,1),
            value = outgoing.get(i,1),
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [outgoing.get(i,[1,3]),'<',state2.get(1,[1,3])],
                [state2.get(1,0),'>',0]
            ]
        )
        circuit.append(op)
        state2.update(0,1,new_value=op.simulate())
                
        # Set momenta of new mode (in the case when it should be first in state2 and state2 contains other modes):
        op = QuantumOperation(
            'pairwise CNOT',
            state2.get(0,2),
            value = outgoing.get(i,2),
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [outgoing.get(i,[1,3]),'<',state2.get(1,[1,3])],
                [state2.get(1,0),'>',0]
            ]
        )
        circuit.append(op)
        state2.update(0,2,new_value=op.simulate())
        
        # Update final occupation number of this entry in outgoing:
        op = QuantumOperation(
            'binary add',
            outgoing.get(i,0),
            value = 1,
            controls = [
                [flag.get(),'==',0],
                [added.get(),'==',0],
                [outgoing.get(i,[1,3]),'<',state2.get(1,[1,3])],
                [state2.get(1,0),'>',0]
            ]
        )
        circuit.append(op)
        outgoing.update(i,0,new_value = op.simulate())
        
        # Uncompute added:
        for j in range(mode_registers):
            
            op = QuantumOperation(
                'bitflip',
                added.get(),
                controls = [
                    [outgoing.get(i,[1,3]),'==',state2.get(j,[1,3])], # quantum numbers and momenta match mode in state2
                    [state2.get(j,0),'>',1] # mode contains more than one particle: in this case, we must have just added a particle to a preexisting mode.
                ]
            )
            circuit.append(op)
            added.update(new_value = op.simulate())

        # print(i,state2,state2.value(),'\n')
    
    # Calculate matrix element:
    op = QuantumOperation(
        'compute matrix element',
        matrix_element.get(),
        value = ( deepcopy(incoming.get()), deepcopy(outgoing.get()) ),
        controls = [
            [flag.get(),'==',0]
        ]
    )
    circuit.append(op)
    outcomes = op.simulate()
    matrix_element.update(0,new_value = outcomes[0])
    matrix_element.update(1,new_value = outcomes[1])
    
    state2 = deepcopy(state2)
    
    # Add outgoing particles to state2:  
    gatecount += interaction._g*mode_registers*3
    gatecount += interaction._g*(mode_registers-1)*4
    gatecount += interaction._g*4
    gatecount += interaction._g*(mode_registers-2)*8
    gatecount += interaction._g*8
    gatecount += interaction._g*mode_registers
    
    # Calculate matrix element:
    gatecount += 1
    
#     print('4.',len(circuit),gatecount)
            
    ############################################################################
    ### Uncompute i0 and remaining uncomputed ancillas in the flag == 0 case ###
    ############################################################################
    
    # Uncompute i0:
    gatecount += num_assignments + 1
    for i in range(num_assignments): # i will be the value of i2...
        op = QuantumOperation(
            'binary subtract',
            i0.get(),
            value = i*num_diagrams,
            controls = [
                [i2.get(),'==',i],
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        i0.update(new_value = op.simulate())
    
    op = QuantumOperation(
        'binary subtract',
        i0.get(),
        value = i1.get(),
        controls = [
            [flag.get(),'==',0]
        ]
    )
    circuit.append(op)
    i0.update(new_value = op.simulate())
    
    # uncompute i1:
    gatecount += num_diagrams
    for i in range(num_diagrams): # will be the value of i1
            
        op = QuantumOperation(
            'binary subtract',
            i1.get(),
            value = i,
            controls = [
                [delta.get(),'==',diagrams[i]],
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        i1.update(new_value = op.simulate())
        
    # compute outgoing_momenta, the particles in outgoing without their occupation numbers and particle types
    gatecount += interaction._g
    for k in range(interaction._g):
        op = QuantumOperation(
            'pairwise CNOT',
            outgoing_momenta.get(k),
            value = outgoing.get(k,2),
            controls = [
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        outgoing_momenta.update(k,new_value = op.simulate())
        
    # uncompute i2
    gatecount += len(assignments.keys())
    for i in assignments.keys():
        op = QuantumOperation(
            'binary subtract',
            i2.get(),
            value = i[0],
            controls = [
                [outgoing_momenta.get(),'==',assignments[i]],
                [flag.get(),'==',0],
                [Q.get(),'==',list(i[1])]
            ]
        )
        circuit.append(op)
        i2.update(new_value = op.simulate())
                
#     for i in range(num_assignments): # i will be the value of i2
#         for j in it.product(*[[m for m in range((interaction._f-interaction._g)*cutoffs[l][0],(interaction._f-interaction._g)*cutoffs[l][1]+1)] for l in range(dim)]): # j will be the value of Q
#             if tuple([i,j]) in list(assignments.keys()):
#                 gatecount += 1
#                 op = QuantumOperation(
#                     'binary subtract',
#                     i2.get(),
#                     value = i,
#                     controls = [
#                         [outgoing_momenta.get(),'==',assignments[tuple([i,j])]],
#                         [flag.get(),'==',0],
#                         [Q.get(),'==',list(j)]
#                     ]
#                 )
#                 circuit.append(op)
#                 i2.update(new_value = op.simulate())
                
    # uncompute outgoing_momenta
    gatecount += interaction._g
    for k in range(interaction._g):
        op = QuantumOperation(
            'pairwise CNOT',
            outgoing_momenta.get(k),
            value = outgoing.get(k,2),
            controls = [
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        outgoing_momenta.update(k,new_value = op.simulate())

#     print('5.',len(circuit),gatecount)
        
    # uncompute incoming and Q:
    
    for i in range(interaction._f-interaction._g-1,0,-1):
        for j in range(mode_registers): # will be equal to the ith entry in delta
            
            # uncompute occupation records in all but first entry in incoming
            
            op = QuantumOperation(
                'binary subtract',
                incoming.get(i,0),
                value = state1.get(j,0),
                controls = [
                    [delta.get(i),'==',j],
                    [flag.get(),'==',0],
                    [incoming.get(i,[1,3]),'!=',incoming.get(i-1,[1,3])]
                ]
            )
            circuit.append(op)
            incoming.update(i,0,new_value = op.simulate())
            
            op = QuantumOperation(
                'binary add',
                incoming.get(i,0),
                value = 1,
                controls = [
                    [delta.get(i),'==',j],
                    [flag.get(),'==',0],
                    [incoming.get(i,[1,3]),'==',incoming.get(i-1,[1,3])]
                ]
            )
            circuit.append(op)
            incoming.update(i,0,new_value = op.simulate())
            
            op = QuantumOperation(
                'binary subtract',
                incoming.get(i,0),
                value = incoming.get(i-1,0),
                controls = [
                    [delta.get(i),'==',j],
                    [flag.get(),'==',0],
                    [incoming.get(i,[1,3]),'==',incoming.get(i-1,[1,3])]
                ]
            )
            circuit.append(op)
            incoming.update(i,0,new_value = op.simulate())

    gatecount += max((interaction._f-interaction._g-1)*mode_registers*3,0)
    # print('1.',gatecount,len(circuit))
    
    if interaction._f-interaction._g > 0:
        for j in range(mode_registers): # will be equal to the 0th entry in delta
            # uncompute occupation record in first entry in incoming
            op = QuantumOperation(
                'binary subtract',
                incoming.get(0,0),
                value = state1.get(j,0),
                controls = [
                    [delta.get(0),'==',j],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            incoming.update(0,0,new_value = op.simulate())
    
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers): # will be equal to the ith entry in delta

            for l in range(dim): # the components of momentum
                op = QuantumOperation(
                    'binary subtract',
                    Q.get(l),
                    value = state1.get(j,2,l),
                    controls = [
                        [delta.get(i),'==',j]
                    ]
                )
                circuit.append(op)
                Q.update(l,new_value = op.simulate())
                
                op = QuantumOperation(
                    'binary subtract',
                    incoming.get(i,2,l),
                    value = state1.get(j,2,l),
                    controls = [
                        [delta.get(i),'==',j],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                incoming.update(i,2,l,new_value = op.simulate())

    if interaction._f-interaction._g > 0:
        gatecount += mode_registers
    gatecount += (interaction._f-interaction._g)*mode_registers*dim*2
    
    # compute delta_out and num_added, which store the locations and numbers of extra particles in state2, respectively
    for i in range(interaction._g):
        for j in range(mode_registers):
            
            op = QuantumOperation(
                'binary add',
                delta_out.get(i),
                value = j,
                controls = [
                    [outgoing.get(i,[1,3]),'==',state2.get(j,[1,3])],
                    [state2.get(j,0),'>',0],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            delta_out.update(i,new_value = op.simulate())
            
            op = QuantumOperation(
                'binary add',
                num_added.get(j),
                value = 1,
                controls = [
                    [outgoing.get(i,[1,3]),'==',state2.get(j,[1,3])],
                    [state2.get(j,0),'>',0],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            num_added.update(j,new_value = op.simulate())
            
    # uncompute outgoing
    
    for i in range(interaction._g-1):
        for j in range(mode_registers):
            
            # uncompute occupation records in all but last entry in outgoing
            
            op = QuantumOperation(
                'binary subtract',
                outgoing.get(i,0),
                value = state2.get(j,0),
                controls = [
                    [delta_out.get(i),'==',j],
                    [flag.get(),'==',0],
                    [outgoing.get(i,[1,3]),'!=',outgoing.get(i+1,[1,3])]
                ]
            )
            circuit.append(op)
            outgoing.update(i,0,new_value = op.simulate())
            
            op = QuantumOperation(
                'binary add',
                outgoing.get(i,0),
                value = 1,
                controls = [
                    [delta_out.get(i),'==',j],
                    [flag.get(),'==',0],
                    [outgoing.get(i,[1,3]),'==',outgoing.get(i+1,[1,3])]
                ]
            )
            circuit.append(op)
            outgoing.update(i,0,new_value = op.simulate())
            
            op = QuantumOperation(
                'binary subtract',
                outgoing.get(i,0),
                value = outgoing.get(i+1,0),
                controls = [
                    [delta_out.get(i),'==',j],
                    [flag.get(),'==',0],
                    [outgoing.get(i,[1,3]),'==',outgoing.get(i+1,[1,3])]
                ]
            )
            circuit.append(op)
            outgoing.update(i,0,new_value = op.simulate())
    
    if interaction._g > 0:
        for j in range(mode_registers): # will be equal to the last entry in delta_out
            # uncompute occupation record in last entry in outgoing
            op = QuantumOperation(
                'binary subtract',
                outgoing.get(interaction._g-1,0),
                value = state2.get(j,0),
                controls = [
                    [delta_out.get(interaction._g-1),'==',j],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            outgoing.update(interaction._g-1,0,new_value = op.simulate())
        
    for i in range(interaction._g):
        for j in range(mode_registers): # will be equal to the ith entry in delta_out
            for l in range(dim): # the components of momentum
                op = QuantumOperation(
                    'binary subtract',
                    outgoing.get(i,2,l),
                    value = state2.get(j,2,l),
                    controls = [
                        [delta_out.get(i),'==',j],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                outgoing.update(i,2,l,new_value = op.simulate())
    
    # compute delta_out and num_added, which store the locations and numbers of extra particles in state2, respectively
    gatecount += interaction._g*mode_registers*2
            
    # uncompute outgoing
    gatecount += max((interaction._g-1)*mode_registers*3,0)
    if interaction._g > 0:
        gatecount += mode_registers
    gatecount += interaction._g*mode_registers*dim
                
#     print('6.',len(circuit),gatecount)
                
    # At this point, all that remains to be uncomputed is delta, num_removed, delta_out, and num_added.
    # We can first uncompute delta using num_removed, and uncompute delta_out using num_added, since each of these pairs contains redundant information.
    
    # uncompute delta using num_removed:
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers): # will be equal to the ith entry in delta
            
            # If j is between the current value of added and its next value, then j is the ith entry in delta:
            
            # Try subtracting j from the ith entry in delta, if it is big enough to be between added and the next value of added.
            op = QuantumOperation(
                'binary subtract',
                delta.get(i),
                value = j,
                controls = [
                    [added.get(),'<=',i],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            delta.update(i,new_value = op.simulate())
            
            # Count how many particles are removed from the current mode and add the total to added:
            op = QuantumOperation(
                'binary add',
                added.get(),
                value = num_removed.get(j),
                controls = [
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            added.update(new_value = op.simulate())
            
            # If j is too big (i.e., at least as large as the next value of added), undo the change to the ith entry of delta:
            op = QuantumOperation(
                'binary add',
                delta.get(i),
                value = j,
                controls = [
                    [added.get(),'<=',i],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            delta.update(i,new_value = op.simulate())
            
        # Reset added to 0:
        op = QuantumOperation(
            'binary subtract',
            added.get(),
            value = interaction._f-interaction._g,
            controls = [
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        added.update(new_value = op.simulate())

    # print(delta_out,delta_out.value(),'\n')
        
    # uncompute delta_out using num_added:
    for i in range(interaction._g):
        for j in range(mode_registers): # will be equal to the ith entry in delta_out
            
            # If j is between the current value of added and its next value, then j is the ith entry in delta_out:
            
            # Try subtracting j from the ith entry in delta_out, if it is big enough to be between added and the next value of added.
            op = QuantumOperation(
                'binary subtract',
                delta_out.get(i),
                value = j,
                controls = [
                    [added.get(),'<=',i],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            delta_out.update(i,new_value = op.simulate())
            
            # Count how many particles are added from the current mode and add the total to added:
            op = QuantumOperation(
                'binary add',
                added.get(),
                value = num_added.get(j),
                controls = [
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            added.update(new_value = op.simulate())
            
            # If j is too big (i.e., at least as large as the next value of added), undo the change to the ith entry of delta:
            op = QuantumOperation(
                'binary add',
                delta_out.get(i),
                value = j,
                controls = [
                    [added.get(),'<=',i],
                    [flag.get(),'==',0]
                ]
            )
            circuit.append(op)
            delta_out.update(i,new_value = op.simulate())
            
        # Reset added to 0:
        op = QuantumOperation(
            'binary subtract',
            added.get(),
            value = interaction._g,
            controls = [
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        added.update(new_value = op.simulate())

    # print(delta_out,delta_out.value(),'\n')
        
    # Uncompute num_removed and num_added using state1 and state2:
    for i in range(mode_registers): # Iterate over modes in state1.
        
        for j in range(-(interaction._f-interaction._g),interaction._g+1): # Iterate over modes in state2 that could be the same as the ith mode in state1: such modes will be indexed by i+j.
            
            if (i+j >= 0) and (i+j < mode_registers): # Make sure i+j is a valid index for a mode (in state2).
                
                # For brevity, we refer to mode i in state1 as mode1, and mode i+j in state2 as mode2.
                # The following 6 operations are all controlled on both mode1 and mode2 encoding physical modes (i.e., not being in the fiducial initial state [0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]),
                # and on mode1 and mode2 being the same.
                
                # Mark mode1 as matched in state2:
                op = QuantumOperation(
                    'bitflip',
                    matched1.get(i),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                matched1.update(i,new_value = op.simulate())
                
                # Further controlled on the occupation of mode1 being greater than the occupation of mode2,
                # the next two operations subtract from num_removed[i] the difference in occupations between mode1 and mode2.
                op = QuantumOperation(
                    'binary add',
                    num_removed.get(i),
                    value = state2.get(i+j,0),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,0),'>',state2.get(i+j,0)],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                num_removed.update(i,new_value = op.simulate())
                
                op = QuantumOperation(
                    'binary subtract',
                    num_removed.get(i),
                    value = state1.get(i,0),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,0),'>',state2.get(i+j,0)],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                num_removed.update(i,new_value = op.simulate())
                
                # Mark mode2 as matched in state1:
                op = QuantumOperation(
                    'bitflip',
                    matched2.get(i+j),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                matched2.update(i+j,new_value = op.simulate())
                
                # Further controlled on the occupation of mode2 being greater than the occupation of mode1,
                # the next two operations set num_added[i+j] to the difference in occupations between mode1 and mode2.
                op = QuantumOperation(
                    'binary add',
                    num_added.get(i+j),
                    value = state1.get(i,0),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,0),'<',state2.get(i+j,0)],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                num_added.update(i+j,new_value = op.simulate())
                
                op = QuantumOperation(
                    'binary subtract',
                    num_added.get(i+j),
                    value = state2.get(i+j,0),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,0),'<',state2.get(i+j,0)],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                num_added.update(i+j,new_value = op.simulate())
                
    for i in range(mode_registers):
        # Extra particles in modes in state1 that are not matched in state2
        op = QuantumOperation(
            'binary subtract',
            num_removed.get(i),
            value = state1.get(i,0),
            controls = [
                [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                [matched1.get(i),'==',0],
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        num_removed.update(i,new_value = op.simulate())
    
        # Extra particles in modes in state2 that are not matched in state1
        op = QuantumOperation(
            'binary subtract',
            num_added.get(i),
            value = state2.get(i,0),
            controls = [
                [state2.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                [matched2.get(i),'==',0],
                [flag.get(),'==',0]
            ]
        )
        circuit.append(op)
        num_added.update(i,new_value = op.simulate())
        
    # Uncompute matched1 and matched2:
    for i in range(mode_registers):
        for j in range(-(interaction._f-interaction._g),interaction._g+1):
            if (i+j >= 0) and (i+j < mode_registers):
                
                op = QuantumOperation(
                    'bitflip',
                    matched1.get(i),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                matched1.update(i,new_value = op.simulate())
                
                op = QuantumOperation(
                    'bitflip',
                    matched2.get(i+j),
                    controls = [
                        [state1.get(i),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state2.get(i+j),'!=',[0,[0 for i in range(qn_dim)],[0 for i in range(dim)]]],
                        [state1.get(i,1),'==',state2.get(i+j,1)],
                        [state1.get(i,2),'==',state2.get(i+j,2)],
                        [flag.get(),'==',0]
                    ]
                )
                circuit.append(op)
                matched2.update(i+j,new_value = op.simulate())
    
    # uncompute delta using num_removed:
    gatecount += (interaction._f-interaction._g)*(mode_registers*3+1)
        
    # uncompute delta_out using num_added:
    gatecount += interaction._g*(mode_registers*3+1)
        
    # Uncompute num_removed and num_added using state1 and state2:
    for i in range(mode_registers): # Iterate over modes in state1.
        for j in range(-(interaction._f-interaction._g),interaction._g+1): # Iterate over modes in state2 that could be the same as the ith mode in state1: such modes will be indexed by i+j.
            if (i+j >= 0) and (i+j < mode_registers): # Make sure i+j is a valid index for a mode (in state2).
                gatecount += 6
    
    gatecount += mode_registers*2
        
    # Uncompute matched1 and matched2:
    for i in range(mode_registers):
        for j in range(-(interaction._f-interaction._g),interaction._g+1):
            if (i+j >= 0) and (i+j < mode_registers):
                gatecount += 2
                
#     print('final counts',len(circuit),gatecount)
    
    
    assert((flag.value() != 0) or (i0.value() == 0))
    assert((flag.value() != 0) or (i1.value() == 0))
    assert((flag.value() != 0) or (i2.value() == 0))
    assert((flag.value() != 0) or (delta.value() == [0 for i in range(interaction._f-interaction._g)]))
    assert((flag.value() != 0) or (delta_out.value() == [0 for i in range(interaction._g)]))
    assert((flag.value() != 0) or (Q.value() == [0 for j in range(dim)]))
    assert((flag.value() != 0) or (incoming.value() == [[0,interaction._qn_in[i],[0 for j in range(dim)]] for i in range(interaction._f-interaction._g)]))
    assert((flag.value() != 0) or (outgoing.value() == [[0,interaction._qn_out[i],[0 for j in range(dim)]] for i in range(interaction._g)]))
    assert((flag.value() != 0) or (outgoing_momenta.value() == [[0 for j in range(dim)] for i in range(interaction._g)]))
    assert((flag.value() != 0) or (num_removed.value() == [0 for j in range(mode_registers)]))
    assert((flag.value() != 0) or (num_added.value() == [0 for j in range(mode_registers)]))
    assert((flag.value() != 0) or (emptied.value() == [-1 for j in range(interaction._f-interaction._g)]))
    assert((flag.value() != 0) or (emptied_rectified.value() == [0 for j in range(interaction._f-interaction._g)]))
    assert((flag.value() != 0) or (added.value() == 0))
    assert((flag.value() != 0) or (matched1.value() == [0 for i in range(mode_registers)]))
    assert((flag.value() != 0) or (matched2.value() == [0 for i in range(mode_registers)]))
    
    if input_state:
        return circuit, state2, i0, matrix_element
    else:
        return circuit


"""
`enumerator_circuit_gatecount(interaction,mode_registers)`: function.

Input:
- `interaction`, an `Interaction`;
- `mode_registers`, the number of mode registers per encoded Fock state.


Output: the number of log-local gates required to implement enumerator_circuit for these inputs.
"""

def enumerator_circuit_gatecount(interaction,mode_registers):
    
    gatecount = 0
    
    # set dimension and cutoffs
    dim = interaction._dim
    cutoffs = interaction._cutoffs

    # incoming quantum numbers
    qn_in = interaction._qn_in
    
    # classical preprocessing:
    diagrams = list_diagrams(interaction,mode_registers)
    num_diagrams = len(diagrams)
#     print(num_diagrams,'\n')
    assignments, num_assignments = outgoing_momentum_assignments(interaction)
#     print('num_diagrams',num_diagrams)
#     print('num_assignments',num_assignments,'\n')
    
    # quantum operations:
            
    # compute sub-indices i1 and i2: i1 = i0 mod num_diagrams, and i2 = floor(i0/num_diagrams).
    gatecount += 1
    gatecount += 2*num_assignments
    
    # compute the entries in delta, list of indices of modes from which the incoming particles will be removed
    gatecount += num_diagrams

    # copy state1 to state2
    gatecount += mode_registers
    
    # Remove particles indexed by entries in delta and compute Q, the total momentum transferred:
    gatecount += (interaction._f - interaction._g)*mode_registers*(1 + 2*dim + 6)
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers):
            if qn_in[i][:2] == [1,0] or qn_in[i][:2] == [1,1]:
                gatecount += 1
    
    # compute outgoing, the full list of outgoing particles (final occupations will be computed later)
    gatecount += len(assignments.keys())*interaction._g
    
    # If outgoing is still empty, it means that this action cannot be applied to state1:
    if interaction._g > 0:
        gatecount += 1
                    
    # Make sure the outgoing particles don't duplicate any fermions:
    gatecount += interaction._g*(interaction._g-1)/2
    gatecount += interaction._g*mode_registers
            
    # Undo the removals from state2 if flag is now nonzero
    gatecount += (interaction._f-interaction._g)*mode_registers*4
    for i in range(interaction._f-interaction._g):
        for j in range(mode_registers):
            if qn_in[i][:2] == [1,0] or qn_in[i][:2] == [1,1]:
                gatecount += 1
        
    # Compute emptied, a list of the mode registers in state2 that were emptied by the removals.
    gatecount += (interaction._f-interaction._g)*mode_registers

    # Copy emptied to emptied_rectified,
    # then from each entry in emptied_rectified, subtract 1 for each nonnegative previous entry less than the current entry.
    # This is to account for the modes corresponding to the previous entries being removed.
    gatecount += interaction._f-interaction._g
    gatecount += (interaction._f-interaction._g)*(interaction._f-interaction._g-1)/2
    
    # Order state2:
    gatecount += (interaction._f-interaction._g)*(mode_registers-1)
            
    # Uncompute emptied_rectified:
    gatecount += (interaction._f-interaction._g)*(interaction._f-interaction._g-1)/2
    gatecount += interaction._f-interaction._g
        
    # Uncompute emptied:
    gatecount += (interaction._f-interaction._g)*mode_registers
    
    # Add outgoing particles to state2:  
    gatecount += interaction._g*mode_registers*3
    gatecount += interaction._g*(mode_registers-1)*4
    gatecount += interaction._g*4
    gatecount += interaction._g*(mode_registers-2)*8
    gatecount += interaction._g*8
    gatecount += interaction._g*mode_registers
    
    # Calculate matrix element:
    gatecount += 1
            
    ############################################################################
    ### Uncompute i0 and remaining uncomputed ancillas in the flag == 0 case ###
    ############################################################################
    
    # Uncompute i0:
    gatecount += num_assignments + 1
    
    # uncompute i1:
    gatecount += num_diagrams
        
    # compute outgoing_momenta, the particles in outgoing without their occupation numbers and particle types
    gatecount += interaction._g
        
    # uncompute i2
    gatecount += len(assignments.keys())
                
    # uncompute outgoing_momenta
    gatecount += interaction._g
                
    # uncompute incoming and Q:
    gatecount += max((interaction._f-interaction._g-1)*mode_registers*3,0)
    if interaction._f-interaction._g > 0:
        gatecount += mode_registers
    gatecount += (interaction._f-interaction._g)*mode_registers*dim*2

    # print('1.',gatecount)
    
    # compute delta_out and num_added, which store the locations and numbers of extra particles in state2, respectively
    gatecount += interaction._g*mode_registers*2
            
    # uncompute outgoing
    gatecount += max((interaction._g-1)*mode_registers*3,0)
    if interaction._g > 0:
        gatecount += mode_registers
    gatecount += interaction._g*mode_registers*dim
                
    # At this point, all that remains to be uncomputed is delta, num_removed, delta_out, and num_added.
    # We can first uncompute delta using num_removed, and uncompute delta_out using num_added, since each of these pairs contains redundant information.
    
    # uncompute delta using num_removed:
    gatecount += (interaction._f-interaction._g)*(mode_registers*3+1)
        
    # uncompute delta_out using num_added:
    gatecount += interaction._g*(mode_registers*3+1)
        
    # Uncompute num_removed and num_added using state1 and state2:
    for i in range(mode_registers): # Iterate over modes in state1.
        for j in range(-(interaction._f-interaction._g),interaction._g+1): # Iterate over modes in state2 that could be the same as the ith mode in state1: such modes will be indexed by i+j.
            if (i+j >= 0) and (i+j < mode_registers): # Make sure i+j is a valid index for a mode (in state2).
                gatecount += 6
    
    gatecount += mode_registers*2
        
    # Uncompute matched1 and matched2:
    for i in range(mode_registers):
        for j in range(-(interaction._f-interaction._g),interaction._g+1):
            if (i+j >= 0) and (i+j < mode_registers):
                gatecount += 2
    
    return gatecount