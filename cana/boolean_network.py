# -*- coding: utf-8 -*-
"""
Boolean Network
================



"""
#	Copyright (C) 2017 by
#	Rion Brattig Correia <rionbr@gmail.com>
#	Alex Gates <ajgates@indiana.edu>
#	All rights reserved.
#	MIT license.
from collections import defaultdict
try:
    import cStringIO.StringIO
except ImportError:
    from io import StringIO
import os

import numpy as np
import networkx as nx
import random
import itertools
from cana.boolean_node import BooleanNode
import cana.bns as bns
from cana.control import fvs, mds, sc
from cana.utils import *
import warnings
from math import log2,ceil
#
#
class BooleanNetwork:
	"""


	"""
	def __init__(self, name='', Nnodes=0, logic=None, sg=None, stg=None, stg_r=None, _ef=None, attractors=None,
			constants={}, Nconstants=0, keep_constants=False,
			bin2num=binstate_to_statenum, num2bin=statenum_to_binstate,
			verbose=False, *args, **kwargs):

		self.name = name 							# Name of the Network
		self.Nnodes = Nnodes 						# Number of Nodes
		self.logic = logic 							# A dict that contains the network logic {<id>:{'name':<string>,'in':<list-input-node-id>,'out':<list-output-transitions>},..}
		self._sg = sg 								# Structure-Graph (SG)
		self._stg = stg 							# State-Transition-Graph (STG)
		self._stg_r = stg_r 						# State-Transition-Graph Reachability dict (STG-R)
		self._eg = None 							# Effective Graph, computed from the effective connectivity
		self._attractors = attractors 				# Network Attractors
		#
		self.keep_constants = keep_constants		# Keep/Include constants in some of the computations
		self.constants = constants 					# A dict that contains of constant variables in the network
		self.Nconstants = len(constants)			# Number of constant variables
		#
		self.Nstates = 2**Nnodes 					# Number of possible states in the network 2^N
		#
		self.verbose = verbose

		# Intanciate BooleanNodes
		self.nodes = list()
		for i in range(Nnodes):
			name = logic[i]['name']
			k = len(logic[i]['in'])
			inputs = [logic[j]['name'] for j in logic[i]['in']]
			outputs = logic[i]['out']
			node = BooleanNode(name=name, k=k, inputs=inputs, outputs=outputs)
			self.nodes.append(node)

		#
		self.bin2num = bin2num						# Helper function. Converts binstate to statenum. It gets updated by `_update_trans_func`
		self.num2bin = num2bin						# Helper function. Converts statenum to binstate. It gets updated by `_update_trans_func`
		self._update_trans_func() 					# Updates helper functions and other variables

	def __str__(self):
		node_names = [node.name for node in self.nodes]
		return "<BNetwork(Name='%s', N=%d, Nodes=%s)>" % (self.name, self.Nnodes, node_names)

	#
	# I/O Methods
	#
	@classmethod
	def from_file(cls, input_file, keep_constants=True, **kwargs):
		"""
		Load the Boolean Network from a file.

		Args:
			infile (string) : The name of a file containing the Boolean Network.

		Returns:
			BooleanNetwork (object) : The boolean network object.

		See also:
			:func:`from_string` :func:`from_dict`
		"""
		with open(input_file, 'r', encoding='utf-8') as infile:
			return cls.from_string(infile.read(), keep_constants=keep_constants, **kwargs)

	@classmethod
	def from_string(cls, input_string, keep_constants=True, **kwargs):
		"""
		Load the Boolean Network from a string.

		Args:
			input_string (string): The representation of a Boolean Network.

		Returns:
			(BooleanNetwork)

		See also:
			:func:`from_file` :func:`from_dict`

		Note: see examples for more information.
		"""
		network_file = StringIO(str(input_string))
		logic = defaultdict(dict)

		line = network_file.readline()
		while line != "":
			if line[0] != '#' and line != '\n':
				# .v <#-nodes>
				if '.v' in line:
					Nnodes = int(line.split()[1])
					for inode in range(Nnodes):
						logic[inode] = {'name':'','in':[],'out':[]}
				# .l <node-id> <node-name>
				elif '.l' in line:
					logic[int(line.split()[1])-1]['name'] = line.split()[2]
				# .n <node-id> <#-inputs> <input-node-id>
				elif '.n' in line:
					inode = int(line.split()[1]) - 1
					indegree = int(line.split()[2])
					for jnode in range(indegree):
						logic[inode]['in'].append(int(line.split()[3 + jnode])-1)

					logic[inode]['out'] = [0 for i in range(2**indegree) if indegree > 0]

					logic_line = network_file.readline().strip()

					if indegree <= 0:
						if logic_line == '':
							logic[inode]['in'] = [inode]
							logic[inode]['out'] = [0,1]
						else:
							logic[inode]['out'] = [int(logic_line)]
					else:
						while logic_line != '\n' and logic_line != '' and len(logic_line)>1:
							for nlogicline in expand_logic_line(logic_line):
								logic[inode]['out'][binstate_to_statenum(nlogicline.split()[0])] = int(nlogicline.split()[1])
							logic_line = network_file.readline().strip()

				# .e = end of file
				elif '.e' in line:
					break
			line = network_file.readline()

		return cls.from_dict(logic, keep_constants=keep_constants, **kwargs)

	@classmethod
	def from_dict(cls, logic, keep_constants=True, **kwargs):
		"""Instanciaets a BoolleanNetwork from a logic dictionary.

		Args:
			logic (dict) : The logic dict.
			keep_constants (bool) :

		Returns:
			(BooleanNetwork)

		See also:
			:func:`from_file` :func:`from_dict`
		"""
		Nnodes = len(logic)
		keep_constants = keep_constants
		constants = {}
		if 'name' in kwargs:
			name = kwargs['name']
		else:
			name = ''
		if keep_constants:
			for i, nodelogic in logic.items():
				# No inputs? It's a constant!
				if len(nodelogic['in']) == 0:
					constants[i] = logic[i]['out'][0]

		return BooleanNetwork(name=name, logic=logic, Nnodes=Nnodes, constants=constants, keep_constants=keep_constants)

	#
	# I/O Methods
	#
	def to_cnet(self, file=None, adjust_no_input=False):
		""" Outputs the network logic to ``.cnet`` format, which is similar to the Berkeley Logic Interchange Format (BLIF).
		This is the format used by BNS to compute attractors.

		Args:
			file (string,optional) : A string of the file to write the output to. If not supplied, a string will be returned.
			adjust_no_input (bool) : Adjust output string for nodes with no input.

		Returns:
			(string) : The ``.cnet`` format string.

		Note:
			See `BNS <https://people.kth.se/~dubrova/bns.html>`_ for more information.

		"""
		# Copy
		logic = self.logic.copy()
		#
		if adjust_no_input:
			for i, data in logic.items():
				# updates in place
				if len(data['in']) == 0:
					data['in'] = [i + 1]
					data['out'] = [0,1]

		bns_string = '.v ' + str(self.Nnodes) + '\n' + '\n'
		for i in range(self.Nnodes):
			k = len(logic[i]['in'])
			bns_string += '.n ' + str(i + 1) + " " + str(k) + " " + " ".join([str(v + 1) for v in logic[i]['in']]) + "\n"
			for statenum in range(2**k):
				# If is a constant (TODO: This must come from the BooleanNode, not the logic)
				if len(logic[i]['out']) == 1:
					bns_string += str(logic[i]['out'][statenum]) + "\n"
				# Not a constant, print the state and output
				else:
					bns_string += statenum_to_binstate(statenum, base=k) + " " + str(logic[i]['out'][statenum]) + "\n"
			bns_string += "\n"

		if file is None:
			return bns_string
		else:
			if isinstance(file, string):
				with open(file, 'w') as iofile:
					iofile.write(bns_string)
					iofile.close()
			else:
				raise AttributeError("File format not supported. Please specify a string.")

	#
	# Methods
	#
	def structural_graph(self, remove_constants=False):
		""" Calculates and returns the structural graph of the boolean network.

		Args:
			remove_constants (bool) : Remove constants from the graph. Defaults to ``False``.
		Returns:
			G (networkx.Digraph) : The boolean network structural graph.
		"""
		self._sg = nx.DiGraph(name="Structural Graph: " + self.name)

		# Add Nodes
		self._sg.add_nodes_from( (i, {'label':n.name}) for i,n in enumerate(self.nodes) )
		for target in range(self.Nnodes):
			for source in self.logic[target]['in']:
				self._sg.add_edge(source, target, **{'weight':1.})

		if remove_constants:
			self._sg.remove_nodes_from(self.constants.keys())
		#
		return self._sg

	def number_interactions(self):
		""" Returns the number of interactions in the Structural Graph (SG).
		Practically, it returns the number of edges of the SG.

		Returns:
			int
		"""
		self._check_compute_variables(sg=True)
		return nx.number_of_edges(self._sg)

	def structural_indegrees(self):
		""" Returns the in-degrees of the Structural Graph. Sorted.

		Returns:
			(int) : the number of in-degrees.
		See also:
			:func:`structural_outdegrees`, :func:`effective_indegrees`, :func:`effective_outdegrees`
		"""
		self._check_compute_variables(sg=True)
		return sorted(self._sg.in_degree().values(), reverse=True)

	def structural_outdegrees(self):
		""" Returns the out-degrees of the Structural Graph. Sorted.

		Returns:
			(list)

		See also:
			:func:`structural_indegrees`, :func:`effective_indegrees`, :func:`effective_outdegrees`
		"""
		self._check_compute_variables(sg=True)
		return sorted(self._sg.out_degree().values(), reverse=True)

	def effective_graph(self, mode='input', bound='upper', threshold=None):
		"""Computes and returns the effective graph of the network.
		In practive it asks each :class:`~boolnets.boolean_node.BooleanNode` for their :func:`~boolnets.boolean_node.BooleanNode.effective_connectivity`.

		Args:
			mode (string) : Per "input" or per "node". Defaults to "node".
			bound (string) : The bound to which compute input redundancy.
			threshold (float) : Only return edges above a certain effective connectivity threshold.
				This is usefull when computing graph measures at diffent levels.

		Returns:
			(networkx.DiGraph) : directed graph

		See Also:
			:func:`~boolnets.boolean_node.BooleanNode.effective_connectivity`
		"""
		if threshold is not None:
			self._eg = nx.DiGraph(name="Effective Graph: " + self.name + "(Threshold: %.2f)" % threshold)
		else:
			self._eg = nx.DiGraph(name="Effective Graph: " + self.name + "(Threshold: None)")

		# Add Nodes
		for i, node in enumerate(self.nodes, start=0):
			self._eg.add_node(i, **{'label':node.name})

		# Add Edges
		for i, node in enumerate(self.nodes, start=0):

			if mode == 'node':
				raise Exception('TODO')

			elif mode == 'input':
				e_is = node.effective_connectivity(mode=mode, bound=bound, norm=False)
				for inputs,e_i in zip(self.logic[i]['in'], e_is):
					# If there is a threshold, only return those number above the threshold. Else, return all edges.
					if ((threshold is None) and (e_i > 0)) or ((threshold is not None) and (e_i > threshold)):
						self._eg.add_edge(inputs, i, **{'weight':e_i})
			else:
				raise AttributeError('The mode you selected does not exist. Try "node" or "input".')

		return self._eg

	def effective_indegrees(self):
		""" Returns the in-degrees of the Effective Graph. Sorted.

		Returns:
			(list)
		See also:
			:func:`effective_outdegrees`, :func:`structural_indegrees`, :func:`structural_outdegrees`
		"""
		self._check_compute_variables(eg=True)
		return sorted(self._eg.in_degree().values(), reverse=True)

	def effective_outdegrees(self):
		""" Returns the out-degrees of the Effective Graph. Sorted.

		Returns:
			(list)
		See also:
			:func:`effective_indegrees`, :func:`structural_indegrees`, :func:`structural_outdegrees`
		"""
		self._check_compute_variables(eg=True)
		return sorted(self._eg.out_degree().values(), reverse=True)

	def state_transition_graph(self):
		"""Creates and returns the full State Transition Graph (STG) for the Boolean Network.

		Returns:
			(networkx.DiGraph) : The state transition graph for the Boolean Network.
		"""
		self._stg = nx.DiGraph(name='STG: '+self.name)
		self._stg.add_nodes_from( (i, {'label':self.num2bin(i)}) for i in range(self.Nstates) )
		for i in range(self.Nstates):
			b = self.num2bin(i)
			self._stg.add_edge(i, self.bin2num(self.step(b)))
		#
		return self._stg

	def stg_indegree(self):
		""" Returns the In-degrees of the State-Transition-Graph (STG). Sorted.

		Returns:
			list
		"""
		self._check_compute_variables(stg=True)
		return sorted(self._stg.in_degree().values(), reverse=True)

	def step(self, initial, n=1):
		""" Steps the boolean network 'n' step from the given initial input condition.
		Args:
			initial (string) : the initial state.
			n (int) : the number of steps.
		Returns:
			(string) : The stepped binary state.
		"""
		# for every node:
		#   node input = breaks down initial by node input
		#   asks node to step with the input
		#   append output to list
		# joins the results from each node output
		assert len(initial) == self.Nnodes
		return ''.join( [ str(node.step( ''.join(initial[j] for j in self.logic[i]['in']) ) ) for i,node in enumerate(self.nodes, start=0) ] )

	def trajectory(self, initial, length=2):
		""" Computes the trajectory of ``length`` steps without the State Transition Graph (STG).
		"""
		trajectory = [initial]
		for istep in range(length):
			trajectory.append(self.step(trajectory[-1]))
		return trajectory

	def trajectory_to_attractor(self, initial):
		""" Computes the trajectory starting at ``initial`` until it reaches an attracor (this is garanteed)

		Args:
			initial (string): the initial state.
		Returns:
			(list): the state trajectory between initial and the final attractor state.
		"""
		self._check_compute_variables(attractors=True)
		attractor_states = [s for att in self._attractors for s in att]

		trajectory = [initial]
		while (trajectory[-1] not in attractor_states):
			trajectory.append(self.step(trajectory[-1]))

		return trajectory

	def attractor(self, initial):
		""" Computes the trajectory starting at ``initial`` until it reaches an attracor (this is garanteed)

		Args:
			initial (string): the initial state.
		Returns:
			attractor (string): the atractor state.
		"""
		self._check_compute_variables(attractors=True)

		trajectory = self.trajectory_to_attractor(initial)
		for attractor in self._attractors:
			if trajectory[-1] in attractor:
				return attractor

	def attractors(self, mode='stg'):
		"""Find the attractors of the boolean network.

		Args:
			mode (string) : ``stg`` or ``sat``. Defaults to ``stg``.
				``stg``: Uses the full State Transition Graph (STG) and identifies the attractors as strongly connected components.
				``bns``: Uses the SAT-based :mod:`cana.bns` to find all attractors.
		Returns:
			attractors (list) : A list containing all attractors for the boolean network.
		See also:
			:mod:`cana.bns`
		"""
		self._check_compute_variables(stg=True)

		if mode == 'stg':
			self._attractors = [list(a) for a in nx.attracting_components(self._stg)]

		elif mode == 'bns':
			self._attractors = bns.attractors(self.to_cnet(file=None, adjust_no_input=False))
		else:
			raise AttributeError("Could not find the specified mode. Try 'stg' or 'bns'.")

		self._attractors.sort(key=len,reverse=True)
		return self._attractors

	def network_bias(self):
		"""Network Bias. The sum of individual node biases divided by the number of nodes.
		Practically, it asks each node for their own bias.

		.. math:
			TODO

		See Also:
			:func:`~boolnets.boolean_node.bias`
		"""
		return sum([node.bias() for node in self.nodes]) / self.Nnodes

	def basin_entropy(self, base=2):
		"""

		"""
		self._check_compute_variables(stg=True)

		prob_vec = np.array([len(wcc) for wcc in nx.weakly_connected_components(self._stg)])/2.0**self.Nnodes
		return entropy(prob_vec, base=base)

	def set_constant(self, node, value=None):
		""" Sets or unsets a node as a constant.

		Args:
			node (int) : The node ``id`` in the logic dict.
		Todo:
			This functions needs to better handle node_id and node_name
		"""
		if value is not None:
			self.nodes[node].constant = True
			self.nodes[node].constant_value = value
			self.Nconstants += 1
		else:
			self.nodes[node].constant = False
			self.nodes[node].constant_value = value
			self.Nconstants -= 1

		self._update_trans_func()

	def remove_all_constants(self):
		self.keep_constants = False
		for inode in self.constants:
			self.set_constant(inode, None)

	def _update_trans_func(self):
		"""

		"""
		if not self.keep_constants:
			self.Nstates = 2**(self.Nnodes - self.Nconstants)
			constant_template = [None if not (ivar in self.constants.keys()) else self.constants[ivar] for ivar in range(self.Nnodes)]
			self.bin2num = lambda bs: constantbinstate_to_statenum(bs, constant_template)
			self.num2bin = lambda sn: binstate_to_constantbinstate(
				statenum_to_binstate(sn, base=self.Nnodes - self.Nconstants), constant_template)
		else:
			self.Nstates =  2**self.Nnodes
			self.bin2num = binstate_to_statenum
			self.num2bin = lambda sn: statenum_to_binstate(sn, base = self.Nnodes)


	#
	# Dynamical Control Methods
	#
	def state_transition_graph_reachability(self, filename=None):
		"""Generates a State-Transition-Graph Reachability (STG-R) dictionary.
		This dict/file will be used by the State Transition Graph Control Analysis.

		Args:
			filename (string) : The location to a file where the STG-R will be stored.

		Returns:
			(dict) : The STG-R in dict format.
		"""
		self._check_compute_variables(stg=True)

		self._stg_r = {}

		if (filename is None):
			for source in self._stg:
				self._stg_r[source] = len(self._dfs_reachable(self._stg, source)) - 1.0
		else:
			try:
				with open(filename, 'rb') as handle:
					self._stg_r = pickle.load(handle)
			except IOError:
				print("Finding STG dict")
				for source in self._stg:
					self._stg_r[source] = len(self._dfs_reachable(self._stg, source)) - 1.0
				with open(filename, 'wb') as handle:
					pickle.dump(self._stg_r, handle)
		return self._stg_r

	def attractor_driver_nodes(self, min_dvs=1, max_dvs=4, verbose=False):
		"""Get the minimum necessary driver nodes by iterating the combination of all possible driver nodes of length :math:`min <= x <= max`.

		Args:
			min_dvs (int) : Mininum number of driver nodes to search.
			max_dvs (int) : Maximum number of driver nodes to search.
		Returns:
			(list) : The list of driver nodes found in the search.
		Note:
			This is an inefficient bruit force search, maybe we can think of better ways to do this?
		TODO:
			Parallelize the search on each combination. Each CSTG is independent and can be searched in parallel.
		See also:
			:func:`controlled_state_transition_graph`, :func:`controlled_attractor_graph`.
		"""
		nodeids = list(range(self.Nnodes))
		if self.keep_constants:
			for cv in self.constants.keys():
				nodeids.remove(cv)

		attractor_controllers_found = []
		nr_dvs = min_dvs
		while (len(attractor_controllers_found) == 0) and (nr_dvs <= max_dvs):
			if verbose: print("Trying with %d Driver Nodes" % (nr_dvs))
			for dvs in itertools.combinations(nodeids, nr_dvs):
				dvs = list(dvs)
				cstg = self.controlled_state_transition_graph(dvs)
				cag = self.controlled_attractor_graph(cstg)
				att_reachable_from = self.mean_reachable_attractors(cag)

				if att_reachable_from == 1.0:
					attractor_controllers_found.append(dvs)
			# Add another driver node
			nr_dvs += 1

		if len(attractor_controllers_found) == 0:
			warnings.warn("No attractor control driver variable sets found after exploring all subsets of size {:,d} to {:,d} nodes!!".format(min_dvs, max_dvs))

		return attractor_controllers_found

	def full_control_driver_nodes(self, min_dvs=1, max_dvs=4, verbose=False, poolsize = 0, taskid = 0):
		"""Get the minimum necessary driver nodes by iterating the combination of all possible driver nodes of length :math:`min <= x <= max`.

		Args:
			min_dvs (int) : Mininum number of driver nodes to search.
			max_dvs (int) : Maximum number of driver nodes to search.
		Returns:
			(list) : The list of driver nodes found in the search.
		Note:
			This is an inefficient bruit force search, maybe we can think of better ways to do this?
		TODO:
			Parallelize the search on each combination. Each CSTG is independent and can be searched in parallel.
		See also:
			:func:`controlled_state_transition_graph`, :func:`controlled_attractor_graph`.
		"""
		nodeids = list(range(self.Nnodes))
		if self.keep_constants:
			for cv in self.constants.keys():
				nodeids.remove(cv)

		attractor_controllers_found = []
		nr_dvs = min_dvs
		count = 0
		def write_dvs(dvs):
			with open('dvs.log','a') as logfile:
				logfile.write(repr(dvs)+'\n')

		exist_file = False
		while (len(attractor_controllers_found) == 0) and (nr_dvs <= max_dvs) and (not exist_file):
			if verbose:
				print("Trying with %d Driver Nodes" % (nr_dvs))
			if os.path.isfile('dvs.log'):
				exist_file = True
			for dvs in itertools.combinations(nodeids, nr_dvs):
				count += 1
				if poolsize != 0:
					if count % poolsize != taskid:
						continue
				dvs = list(dvs)
				cstg = self.controlled_state_transition_graph(dvs)
				conf_reachable_from = self.mean_reachable_configurations(cstg)

				if conf_reachable_from == 1:
					attractor_controllers_found.append(dvs)
					write_dvs(dvs)
			# Add another driver node
			nr_dvs += 1

		if len(attractor_controllers_found) == 0:
			warnings.warn("No attractor control driver variable sets found after exploring all subsets of size {:,d} to {:,d} nodes!!".format(min_dvs, max_dvs))

		return attractor_controllers_found


	def controlled_state_transition_graph(self, driver_nodes=[]):
		"""Returns the Controlled State-Transition-Graph (CSTG).
		In practice, it copies the original STG, flips driver nodes (variables), and updates the CSTG.

		Args:
			driver_nodes (list) : The list of driver nodes.
		Returns:
			(networkx.DiGraph) : The Controlled State-Transition-Graph.
		See also:
			:func:`attractor_driver_nodes`, :func:`controlled_attractor_graph`.
		"""
		self._check_compute_variables(attractors=True)

		if self.keep_constants:
			for dv in driver_nodes:
				if dv in self.constants:
					warnings.warn("Cannot control a constant variable '%s'! Skipping" % self.nodes[dv].name )

		attractor_states = [s for att in self._attractors for s in att]

		cstg = copy.deepcopy(self._stg)
		cstg.name = 'C-' + cstg.name +' (' + ','.join(map(str,[self.nodes[dv].name for dv in driver_nodes])) + ')'

		# add the control pertubations applied to all other configurations
		for statenum in range(self.Nstates):
			binstate = self.num2bin(statenum)
			controlled_states = flip_binstate_bit_set(binstate, copy.copy(driver_nodes))
			controlled_states.remove(binstate)

			for constate in controlled_states:
				cstg.add_edge(statenum, self.bin2num(constate))

		return cstg

	def pinned_step(self, initial, pinned_var):
		""" Steps the boolean network 1 step from the given initial input condition when the driver variables are pinned
		to their controlled states.
		Args:
			initial (string) : the initial state.
			n (int) : the number of steps.
		Returns:
			(string) : The stepped binary state.
		"""
		# for every node:
		#   node input = breaks down initial by node input
		#   asks node to step with the input
		#   append output to list
		# joins the results from each node output
		assert len(initial) == self.Nnodes

		return ''.join( [ str(node.step( ''.join(initial[j] for j in self.logic[i]['in']) ) ) if not (i in pinned_var) else initial[i] for i,node in enumerate(self.nodes, start=0) ] )

	def pin_selected_nodes(self, initial, pinned_binstate, pinned_var):
		if len(pinned_binstate) != len(pinned_var):
			print('error! Unmatched arguments in pin_selected_nodes()')
			return initial
		copy_initial = [i for i in initial]
		for i, var in enumerate(pinned_var):
			copy_initial[var] = pinned_binstate[i]
		return ''.join(copy_initial)

	def pinning_controlled_state_transition_graph(self, driver_nodes=[]):
		"""Returns a dictionary of Controlled State-Transition-Graph (CSTG) under the assumptions of
		pinning controllability:
		In practice, it copies the original STG, flips driver nodes (variables), and updates the CSTG.

		Args:
			driver_nodes (list) : The list of driver nodes.
		Returns:
			(networkx.DiGraph) : The Pinning Controlled State-Transition-Graph.
		See also:
			:func: `controlled_state_transition_graph`, :func:`attractor_driver_nodes`, :func:`controlled_attractor_graph`.
		"""
		self._check_compute_variables(attractors=True)

		if self.keep_constants:
			for dv in driver_nodes:
				if dv in self.constants:
					warnings.warn("Cannot control a constant variable '%s'! Skipping" % self.nodes[dv].name )

		uncontrolled_system_size = self.Nnodes - len(driver_nodes)

		pcstg_dict = {}
		for att in self._attractors:
			dn_attractor_transitions = [tuple(''.join([self.num2bin(s)[dn] for dn in driver_nodes]) for s in att_edge)
			for att_edge in self._stg.subgraph(att).edges()]

			pcstg_states = [self.bin2num(binstate_pinned_to_binstate(
				statenum_to_binstate(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes) )
			for statenum in range(2**uncontrolled_system_size) for attsource, attsink in dn_attractor_transitions]

			pcstg = nx.DiGraph(name='STG: '+self.name)
			pcstg.name = 'PC-' + pcstg.name +' (' + ','.join(map(str,[self.nodes[dv].name for dv in driver_nodes])) + ')'

			pcstg.add_nodes_from( (ps, {'label':ps}) for ps in pcstg_states)

			for attsource, attsink in dn_attractor_transitions:
				for statenum in range(2**uncontrolled_system_size):
					initial = binstate_pinned_to_binstate(statenum_to_binstate(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes)
					next_step = self.pin_selected_nodes(self.pinned_step(initial, pinned_var=driver_nodes), pinned_binstate=attsink, pinned_var=driver_nodes)
					# next_step = self.pinned_step(initial, pinned_var=driver_nodes)
					pcstg.add_edge(self.bin2num(initial), self.bin2num(next_step))

			pcstg_dict[tuple(att)] = pcstg

		return pcstg_dict

	def controlled_attractor_graph(self, cstg):
		"""
		Args:
			cstg (networkx.DiGraph) : A Controlled State-Transition-Graph (CSTG)
		Returns:
			(networkx.DiGraph) : The Controlled Attractor Graph (CAG)
		See also:
			:func:`attractor_driver_nodes`, :func:`controlled_state_transition_graph`.
		"""
		self._check_compute_variables(attractors=True)

		Nattract = len(self._attractors)

		cag = nx.DiGraph(name='CAG: ' + cstg.name)
		# Nodes
		for i, attr in enumerate(self._attractors):
			cag.add_node(i, **{'label':'|'.join([self.num2bin(a) for a in attr])})
		# Edges
		for i in range(Nattract):
			ireach = self._dfs_reachable(cstg, self._attractors[i][0])
			for j in range(i + 1, Nattract):
				if self._attractors[j][0] in ireach:
					cag.add_edge(i, j)
				if self._attractors[i][0] in self._dfs_reachable(cstg, self._attractors[j][0]):
					cag.add_edge(j, i)
		return cag

	def pinning_control_nodes(self):
		self._check_compute_variables(attractors=True)
		if len(self._attractors) == 1:
			return []
		min_pn_N = ceil(log2(len(self._attractors)))
		nodeids = list(range(self.Nnodes))

		bin_attractors = [[self.num2bin(state) for state in attr] for attr in self._attractors]

		def check_nodes_neccesary(nodes, attractors):
			if len(nodes) == 0:
				return False
			none_overlap_set = set()
			remaining_attractors = []
			for attr in attractors:
				if len(attr) == 1:
					node_stat = tuple(attr[0][node] for node in nodes)
				else:
					# else, if the nodes value are all the same for the limit cycle, add it too
					node_stat = tuple(attr[0][node] for node in nodes)
					jump = False
					for a_i in range(1, len(attr)):
						current_node_stat = tuple(attr[a_i][node] for node in nodes)
						if current_node_stat != node_stat:
							jump = True
							break
					if jump:
						remaining_attractors.append(attr)
						continue
				if node_stat not in none_overlap_set:
					none_overlap_set.add(node_stat)
				else:
					return False
			for attr in remaining_attractors:
				for state in attr:
					node_stat = tuple(state[node] for node in nodes)
					if node_stat in none_overlap_set:
						return False
			return True

		result = []
		for n_pin in range(min_pn_N, self.Nnodes):
			if len(result) > 0:
				break
			for pvs in itertools.combinations(nodeids, n_pin):
				if check_nodes_neccesary(pvs, bin_attractors):
					controlled = True
					for attr, pcstg in self.pinning_controlled_state_transition_graph(pvs).items():
						if len([1 for i in nx.weakly_connected_components(pcstg)]) > 1:
							controlled = False
							break
					if controlled:
						result.append(pvs)
		if len(result) == 0:
			return [list(range(self.Nnodes))]
		return result

	def mean_reachable_configurations(self, cstg):
		"""Returns the Mean Fraction of Reachable Configurations

		Args:
			cstg (networkx.DiGraph) : The Controlled State-Transition-Graph.
		Returns:
			(float) : Mean Fraction of Reachable Configurations
		"""
		reachable_from = []

		for source in cstg:
			control_reach = len(self._dfs_reachable(cstg, source)) - 1.0
			reachable_from.append(control_reach)

		norm = (2.0**self.Nnodes - 1.0) * len(reachable_from)
		if reachable_from == norm:
			return 1
		reachable_from = sum(reachable_from) / (norm)

		return reachable_from

	def mean_controlable_configurations(self, cstg):
		"""The Mean Fraction of Controlable Configurations

		Args:
			cstg (networkx.DiGraph) : The Controlled State-Transition-Graph.
		Returns:
			(float) : Mean Fraction of Controlable Configurations.
		"""
		self._check_compute_variables(stg_r=True)

		control_from, reachable_from = [], []

		for source in cstg:
			control_reach = len(self._dfs_reachable(cstg, source)) - 1.0
			control_from.append(control_reach - self._stg_r[source])
			reachable_from.append(control_reach)

		norm = (2.0**self.Nnodes - 1.0) * len(reachable_from)
		control_from = sum(control_from) / (norm)

		return control_from

	def mean_reachable_attractors(self, cag, norm=True):
		"""The Mean Fraction of Reachable Attractors to a specific Controlled Attractor Graph (CAG).

		Args:
			cag (networkx.DiGraph) : A Controlled Attractor Graph (CAG).

		Returns:
			(float) Mean Fraction of Reachable Attractors
		"""
		att_norm = (float(len(cag)) - 1.0) * len(cag)

		if att_norm == 0:
			# if there is only one attractor everything is reachable
			att_reachable_from = 1
		else:
			# otherwise find the reachable from each attractor
			att_reachable_from = [len(self._dfs_reachable(cag, idxatt)) - 1.0 for idxatt in cag]
			att_reachable_from = sum(att_reachable_from) / (att_norm)

		return att_reachable_from


	def fraction_pinned_attractors(self, pcstg_dict):
		"""Returns the Number of Accessible Attractors

		Args:
			pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.
		Returns:
			(int) : Number of Accessible Attractors
		"""

		reached_attractors = []
		att_set = [set(i) for i in pcstg_dict]
		for att, pcstg in pcstg_dict.items():
			pinned_att = list(nx.attracting_components(pcstg))
			print(set(att), pinned_att)
			for patt in pinned_att:
				if patt not in att_set:
					print('warning: new att! %s' % patt)
			reached_attractors.append(set(att) in pinned_att)

		return sum(reached_attractors) / float(len(pcstg_dict))

	def fraction_pinned_configurations(self, pcstg_dict):
		"""Returns the Fraction of successfully Pinned Configurations

		Args:
			pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.
		Returns:
			(list) : the Fraction of successfully Pinned Configurations to each attractor
		"""

		pinned_configurations = []
		for att, pcstg in pcstg_dict.items():
			att_reached = False
			for wcc in nx.weakly_connected_components(pcstg):
				if set(att) in list(nx.attracting_components(pcstg.subgraph(wcc))):
					pinned_configurations.append(len(wcc)/ len(pcstg))
					att_reached = True
			if not att_reached:
				pinned_configurations.append(0)

		return pinned_configurations

	def mean_fraction_pinned_configurations(self, pcstg_dict):
		"""Returns the mean Fraction of successfully Pinned Configurations

		Args:
			pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.
		Returns:
			(int) : the mean Fraction of successfully Pinned Configurations
		"""
		return sum(self.fraction_pinned_configurations(pcstg_dict))/len(pcstg_dict)

	def _dfs_reachable(self, G, source):
		"""Produce nodes in a depth-first-search pre-ordering starting from source."""
		return [n for n in nx.dfs_preorder_nodes(G, source)]

	#
	# Feedback Vertex Set (FVS)
	#
	def feedback_vertex_set_driver_nodes(self, graph='structural', method='grasp', max_iter=1, max_search=11, keep_self_loops=True, *args, **kwargs):
		"""The minimum set of necessary driver nodes to control the network based on Feedback Vertex Set (FVS) theory.

		Args:
			graph (string) : Which graph to perform computation
			method (string) : FVS method. ``bruteforce`` or ``grasp`` (default).
			max_iter (int) : The maximum number of iterations used by the grasp method.
			max_search (int) : The maximum number of searched used by the bruteforce method.
			keep_self_loops (bool) : Keep or remove self loop in the graph to be searched.

		Returns:
			(list) : A list-of-lists with FVS solution nodes.

		Note:
			When computing FVS on the structural graph, you might want to use ``remove_constants=True``
			to make sure the resulting set is minimal – since constants are not controlabled by definition.
			Also, when computing on the effective graph, you can define the desired ``threshold`` level.
		"""
		self._check_compute_variables(sg=True)

		if graph == 'structural':
			dg = self.structural_graph(*args, **kwargs)
		elif graph == 'effective':
			dg = self.effective_graph(mode='input', bound='upper', threshold=None)
		else:
			raise AttributeError("The graph type '%s' is not accepted. Try 'structural' or 'effective'." % graph)
		#
		if method == 'grasp':
			fvssets = fvs.fvs_grasp(dg, max_iter=max_iter, keep_self_loops=keep_self_loops)
		elif method == 'bruteforce':
			fvssets = fvs.fvs_bruteforce(dg, max_search=max_search, keep_self_loops=keep_self_loops)
		else:
			raise AttributeError("The FVS method '%s' does not exist. Try 'grasp' or 'bruteforce'." % method)

		return fvssets #[ [self.nodes[i].name for i in fvsset] for fvsset in fvssets]

	#
	# Minimum Dominating Set
	#
	def minimum_dominating_set_driver_nodes(self, max_search=5, keep_self_loops=True):
		"""The minimun set of necessary driver nodes to control the network based on Minimum Dominating Set (MDS) theory.

		Args:
			max_search (int) : Maximum search of additional variables. Defaults to 5.
			keep_self_loops (bool) : If self-loops are used in the computation.
		Returns:
			(list) : A list-of-lists with MDS solution nodes.
		"""
		self._check_compute_variables(sg=True)
		#
		mdssets = mds.mds(self._sg, max_search=max_search, keep_self_loops=keep_self_loops)
		return  mdssets #[ [self.nodes[i].name for i in mdsset] for mdsset in mdssets]

	#
	# Structural Controllability
	#
	def structural_controllability_driver_nodes(self, keep_self_loops=True):
		""" The minimum set of necessary driver nodes to control the network based on Structural Controlability (SC) theory.

		Args:
			keep_self_loops (bool) : If self-loops are used in the computation.
		Returns:
			(list) : A list-of-lists with SC solution nodes.
		"""
		self._check_compute_variables(sg=True)
		#
		scsets = sc.sc(self._sg, keep_self_loops=keep_self_loops)
		return scsets # [ [self.nodes[i].name for i in scset] for scset in scsets]

	#
	# Dynamics Canalization Map (DCM)
	#
	def dynamics_canalization_map(self, output=None, simplify=True, keep_constants=True):
		""" Computes the Dynamics Canalization Map (DCM).
		In practice, it asks each node to compute their Canalization Map and then puts them together, simplifying it if possible.

		Args:
			output (int) : The output DCM to return. Default is ``None``, retuning both [0,1].
			simplify (bool) : Attemps to simpify the DCM by removing thresholds nodes with :math:`\tao=1`.
			keep_constants (bool) : Keep or remove constants from the DCM.
		Returns:
			DCM (networkx.DiGraph) : a directed graph representation of the DCM.
		See Also:
			:func:`boolean_node.canalizing_map` for the CM and :func:`drawing.draw_dynamics_canalizing_map_graphviz` for plotting.
		"""
		CMs = []
		for node in self.nodes:
			if keep_constants or not node.constant:
				CMs.append( node.canalizing_map(output) )
		# https://networkx.readthedocs.io/en/stable/reference/algorithms.operators.html
		DCM = nx.compose_all(CMs)
		DCM.name = 'DCM: %s' % (self.name)

		if simplify:
			#Loop all threshold nodes
			threshold_nodes=[(n,d) for n,d in DCM.nodes(data=True) if d['type']=='threshold']
			for n,d in threshold_nodes:

				# Constant, remove threshold node
				if d['tau'] == 0:
					DCM.remove_node(n)

				# Tau == 1
				if d['tau'] == 1:
					in_nei = list(DCM.in_edges(n))[0]
					out_nei = list(DCM.out_edges(n))[0]

					neis = set( list(in_nei) + list(out_nei) )

					# Convert to self loop
					if (in_nei == out_nei[::-1]):
						DCM.remove_node(n)
						DCM.add_edge(in_nei[0],out_nei[1], **{'type':'simplified','mode':'selfloop'})
					# Link variables nodes directly
					elif not any([DCM.node[tn]['type']=='fusion' for tn in in_nei]):
						DCM.remove_node(n)
						DCM.add_edge(in_nei[0],out_nei[1], **{'type':'simplified','mode':'direct'})
		# Remove Isolates
		DCM.remove_nodes_from(nx.isolates(DCM))

		return DCM

	def _check_compute_variables(self, **kwargs):
		""" Recursevely check if the requested control variables are instantiated/computed, otherwise computes them in order.
		"""
		if 'sg' in kwargs:
			if self._sg is None:
				if self.verbose: print("Computing: Structural Graph")
				self._sg = self.structural_graph()

		elif 'eg' in kwargs:
			if self._eg is None:
				if self.verbose: print("Computing: Effective Graph")
				self._eg = self.effective_graph()

		elif 'stg' in kwargs:
			self._check_compute_variables(sg=True)
			if self._stg is None:
				if self.verbose: print("Computing: State-Transition-Graph")
				self._stg = self.state_transition_graph()

		elif 'attractors' in kwargs:
			self._check_compute_variables(stg=True)
			if self._attractors is None:
				if self.verbose: print("Computing: Attractors")
				self._attractors = self.attractors()

		elif 'stg_r' in kwargs:
			self._check_compute_variables(stg=True)
			if self._stg_r is None:
				self._stg_r = self.state_transition_graph_reachability()
		else:
			raise Exception('Control variable name not found. %s' % kwargs)
		return True

	#
	# Get Node Names from Ids
	#
	def _get_node_name(self, id):
		""" Return the name of the node based on its id.

		Args:
			id (int): id of the node.
		Returns:
			name (string): name of the node.
		"""
		try:
			node = self.nodes[id]
		except:
			raise AttributeError("Node with id '%d' does not exist." % (id))
		else:
			return node.name

	def get_node_name(self, iterable=[]):
		""" Return node names. Optionally, it returns only the names of the ids requested.

		Args:
			iterable (int,list, optional) : The id (or list of ids) of nodes to which return their names.
		Returns:
			names (list) : The name of the nodes.
		"""
		# If only one id is passed, make it a list
		if not isinstance(iterable, list):
			iterable = [iterable]
		# No ids requested, return all the names
		if not len(iterable):
			return [n.name for n in self.nodes]
		# otherwise, use the recursive map to change ids to names
		else:
			return recursive_map(self._get_node_name, iterable)
	#
	# Plotting Methods
	#
	def derrida_curve(self, nsamples=10, random_seed=None, method='random'):
		""" The Derrida Curve (also reffered as criticality measure :math:`D_s`).
		When "mode" is set as "random" (default), it would use random sampling to estimate Derrida value
		If "mode" is set as "sensitivity", it would use c-sensitivity to calculate Derrida value (slower)
		You can refer to :cite:'kadelka2017influence' about why c-sensitivity can be used to caculate Derrida value
		Args:
			nsamples (int) : The number of samples per hammimg distance to get.
			random_seed (int) : The random state seed.
			method (string) : specify the method you want. either 'random' or 'sensitivity'
		Returns:
			(dx,dy) (tuple) : The dx and dy of the curve.
		"""
		random.seed(random_seed)

		dx = np.linspace(0,1,self.Nnodes)
		dy = np.zeros(self.Nnodes)

		if method == 'random':
			# for each possible hamming distance between the starting states
			for hamm_dist in range(1, self.Nnodes + 1):

				# sample nsample times
				for isample in range(nsamples):
					rnd_config = [random.choice(['0', '1']) for b in range(self.Nnodes)]
					perturbed_var = random.sample(range(self.Nnodes), hamm_dist)
					perturbed_config = [flip_bit(rnd_config[ivar]) if ivar in perturbed_var else rnd_config[ivar] for ivar in range(self.Nnodes)]
					dy[hamm_dist-1] += hamming_distance(self.step(rnd_config), self.step(perturbed_config))

			dy /= float(self.Nnodes * nsamples)
		elif method == 'sensitivity':
			for hamm_dist in range(1,self.Nnodes +1):
				dy[hamm_dist-1] = sum([node.c_sensitivity(hamm_dist,mode='forceK',max_k=self.Nnodes) for node in self.nodes])/self.Nnodes

		return dx, dy

	def activity_graph(self, threshold=None):
		if threshold is not None:
			act_g = nx.DiGraph(name="activity Graph: " + self.name + "(Threshold: %.2f)" % threshold)
		else:
			act_g = nx.DiGraph(name="activity Graph: " + self.name + "(Threshold: None)")

		# Add Nodes
		for i, node in enumerate(self.nodes, start=0):
			act_g.add_node(i, **{'label': node.name})

		# Add Edges
		for i, node in enumerate(self.nodes, start=0):

			a_is = node.activities(self.Nnodes)
			for inputs, a_i in zip(self.logic[i]['in'], a_is):
				# If there is a threshold, only return those number above the threshold. Else, return all edges.
				if ((threshold is None) and (a_i > 0)) or ((threshold is not None) and (a_i > threshold)):
					act_g.add_edge(inputs, i, **{'weight': a_i})

		return act_g
