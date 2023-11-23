import math
import numpy as np, os, sys, re, glob, subprocess, math, unittest, shutil, time, string, logging, gc

np.set_printoptions(precision=4)
from time import gmtime, strftime
from random import shuffle, choice, sample, choices
import random
from itertools import product
from functools import partial
import inspect
import itertools


base_folder = './'
try:
	os.mkdir(base_folder+'log')
	os.mkdir(base_folder+'agent_pool')
	os.mkdir(base_folder+'job_pool')
	os.mkdir(base_folder+'result_pool')
except:
	pass


current_time = strftime("%Y%m%d_%H%M%S", gmtime())

log_name = 'sim_{}_initRadius{}_mainRadius{}_initOn{}_initIter{}.log'.format(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG,
										format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class DummyIndv(object): pass

data = np.load(sys.argv[1])
logging.info(data['adj_matrix'])
adjm=data['adj_matrix']
adjm[adjm==0] = 1
actual_elem = np.sum([ np.prod(adjm[d]) for d in range(adjm.shape[0]) ])
logging.info(actual_elem)

np.save(sys.argv[1][0:-4]+'.npy', data['goal'])
evoluation_goal = sys.argv[1][0:-4]+'.npy'

class Individual(object):
	def __init__(self, adj_matrix=None, scope=None, **kwargs):
		super(Individual, self).__init__()
		if adj_matrix is None:
			self.adj_matrix = kwargs['adj_func'](**kwargs)
		else:
			self.adj_matrix = adj_matrix
		self.scope = scope
		self.dim = self.adj_matrix.shape[0]
		adj_matrix_T=np.copy(self.adj_matrix)
		size_data=np.arange(self.dim)
		for i in range(self.dim):
				size_data[i]=adj_matrix_T[i][0][0][0]
		
		rank=np.arange(self.dim)
		for i in range(self.dim):
				rank[i]=adj_matrix_T[i][0][0][1]

		permute_code=np.arange(0, self.dim)
		for i in range(self.dim):
				permute_code[i]=adj_matrix_T[i][0][1]
		adj_matrix_R = np.diag(size_data)
		temp=np.arange(self.dim-1)
		temp=temp[::-1]
		temp[0]=self.dim-3
		connection_index = []
		connection_index.append(0)
		for i in range(self.dim):
				if i==1:
						connection_index.append(connection_index[-1]+1)
				else:
						if i==0:
								connection_index.append(connection_index[-1]+(temp[i]+1))
						else:
								connection_index.append(connection_index[-1]+(temp[i-1]+1))
		connection_index=connection_index[0:self.dim]
		connection =rank.tolist()
		index_tuple=np.triu_indices(self.dim, 1)
		index_tuple1=index_tuple[0]
		index_tuple2=index_tuple[1]
		index_tuple1=index_tuple1[connection_index]
		index_tuple2=index_tuple2[connection_index]
		index_tuple=[index_tuple1,index_tuple2]
		index_tuple=tuple(index_tuple)
		adj_matrix_R[index_tuple] = connection
		adj_matrix_R[np.tril_indices(self.dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(self.dim, -1)]
		index=np.diag_indices_from(adj_matrix_R)
		adj_matrix_R[index]=0

		permute=permute_code

		permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
		for i in range(self.dim):
				permutation_matrix[permute[i],i] = 1

		adj_matrix_R=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())




		self.parents = kwargs['parents'] if 'parents' in kwargs.keys() else None
		self.repeat = kwargs['evaluate_repeat'] if 'evaluate_repeat' in kwargs.keys() else 1
		self.iters = kwargs['max_iterations'] if 'max_iterations' in kwargs.keys() else 10000
		self.rse_therhold=kwargs['rse_therhold'][0]
		self.Adam_Step=kwargs['Adam_Step'][0]

		adj_matrix_k = np.copy(adj_matrix_R)
		adj_matrix_k[adj_matrix_k==0] = 1

		self.present_elements = np.prod(np.diag(adj_matrix_k))
		self.actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(self.dim) ])
		self.sparsity = self.actual_elements/self.present_elements

	def deploy(self, sge_job_id):
		try:
			path = base_folder+'/job_pool/{}.npz'.format(sge_job_id)
			np.savez(path, adj_matrix=self.adj_matrix, scope=self.scope, repeat=self.repeat, iters=self.iters,rse_TH=self.rse_therhold,Adam_Step=self.Adam_Step)
			self.sge_job_id = sge_job_id
			return True
		except Exception as e:
			raise e

	def collect(self, fake_loss=False):
		if not fake_loss:
			try:
				path = base_folder+'/result_pool/{}.npz'.format(self.scope.replace('/', '_'))
				result = np.load(path)
				self.repeat_loss = result['repeat_loss']

				os.remove(path)
				return True
			except Exception:
				return False
		else:
			self.repeat_loss = [9999]*self.repeat
			return True

class Generation(object):
	def __init__(self,rse_interpolation_on,update_index,society_name,adjmatrix_set_temp,best_individual_numbers_evaluation,evaluation_numbers,local_reinitilize_update_flag,meetsamestructure_local_reupdate_flag,times_iteration_stop,local_reupdate_flag,structure_unchange_count,rse_propagate_index,rse_propagate,best_generation,times_of_LocalUpdate,times_of_LocalSampling,local_structure_update_flag,temp_best_local_structure,temp_best_local_structure_fitness,center_structure,center_structure_fitness,optimize_grid_set,n_generation,pG=None, name=None, **kwargs):
		super(Generation, self).__init__()
		self.name = name
		self.N_islands = kwargs['N_islands'] if 'N_islands' in kwargs.keys() else 1
		self.kwargs = kwargs
		self.out = self.kwargs['out']
		self.rank = self.kwargs['rank']
		self.size = self.kwargs['size']

		self.Local_Opt_Iter =self.kwargs['Local_Opt_Iter']
		self.Rse_interpolation_on =rse_interpolation_on
		self.Rse_interpolation_Times = self.kwargs['Rse_interpolation_Times']
		self.indv_to_collect = []
		self.n_generation = n_generation 

		if self.Rse_interpolation_on==1:
			self.Local_Step = self.kwargs['Local_Step_init']
		else:
			self.Local_Step = self.kwargs['Local_Step_main']


		self.indv_to_distribute = []
		self.update_index=update_index
		self.society_name=society_name
		self.adjmatrix_set_temp=adjmatrix_set_temp
		self.best_individual_numbers_evaluation=best_individual_numbers_evaluation
		self.evaluation_numbers=evaluation_numbers
		self.local_reinitilize_update_flag=local_reinitilize_update_flag
		self.meetsamestructure_local_reupdate_flag=meetsamestructure_local_reupdate_flag
		self.times_iteration_stop=times_iteration_stop
		self.local_reupdate_flag=local_reupdate_flag
		self.structure_unchange_count=structure_unchange_count
		self.rse_propagate_index=rse_propagate_index
		self.rse_propagate=rse_propagate
		self.best_generation=best_generation
		self.times_of_LocalSampling= times_of_LocalSampling
		self.times_of_LocalUpdate= times_of_LocalUpdate
		self.center_structure= center_structure
		self.center_structure_fitness= center_structure_fitness
		self.optimize_grid_set= optimize_grid_set
		self.temp_best_local_structure=temp_best_local_structure
		self.temp_best_local_structure_fitness=temp_best_local_structure_fitness
		self.local_structure_update_flag=local_structure_update_flag
        
		if pG is not None:
			self.societies = {}
			for k, v in pG.societies.items():
				self.societies[k] = {}
				self.societies[k]['indv'] = \
						[ Individual( adj_matrix=indv.adj_matrix, parents=indv.parents,
													scope='{}/{}/{:03d}'.format(self.name, k, idx), **self.kwargs) \
						for idx, indv in enumerate(v['indv']) ]
				self.indv_to_distribute += [indv for indv in self.societies[k]['indv']]

		elif 'random_init' in kwargs.keys():
			self.societies = {}
			for n in range(self.kwargs['N_islands']):
				society_name = ''.join(choice(string.ascii_uppercase + string.digits) for _ in range(6))
				self.society_name=society_name
				self.societies[society_name] = {}
				adj_matrix_init=self.__init_adj_matrix__()

				if self.Rse_interpolation_on==0:
					self.societies[society_name]['indv'] = [ \
							Individual(adj_matrix=adj_matrix_init[i],scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
							 **self.kwargs) \
							for i in range(2*self.Local_Step+1) ]
				else:
					self.societies[society_name]['indv'] = [ \
							Individual(adj_matrix=adj_matrix_init[i],scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
							 **self.kwargs) \
							for i in range(3) ]
					self.indv_to_distribute += [indv for indv in self.societies[society_name]['indv']]


	def __call__(self, **kwargs):
		try:
			self.__evaluate__()
			if 'callbacks' in kwargs.keys():
				for c in kwargs['callbacks']:
					c(self)
			self.__evolve__()
			return True
		except Exception as e:
			raise e

	def __init_adj_matrix__(self, **kwargs):
		out=self.out
		out=out.reshape((self.size,1))
        
        #### randomly choose a structure as starting point
		rank=np.random.randint(1, self.rank, (self.size,1))


		permute=np.arange(1,self.size)
		np.random.shuffle(permute)
		permute=np.insert(np.array([0],dtype=int),1,permute,0)
		np.random.shuffle(permute)

		permute=permute.reshape((self.size,1))


		adj_matrix=np.rec.fromarrays((out, rank))
		adj_matrix=np.rec.fromarrays((adj_matrix, permute))
          
		self.center_structure=adj_matrix

        #### Generating grid of local subtensor
		grid_set=[]
		for i in range(self.size):
				temp_r=np.arange(1,self.Local_Step+1)
				temp_l=-1*temp_r
				temp_l=temp_l[::-1]
				temp_grid=np.concatenate([temp_l+rank[i],rank[i],rank[i]+temp_r])
				if len(temp_grid)>=self.rank-1:
						grid_set.append(np.arange(1,self.rank).tolist())
				else:
						if len(np.where(temp_grid<1)[0])>=1:
								for j1 in range(len(np.where(temp_grid<1)[0])):
										temp_grid=np.delete(temp_grid,0)
										temp_grid=np.concatenate([temp_grid,np.array([temp_grid[-1]+1])])  
								grid_set.append(temp_grid.tolist())
						elif len(np.where(temp_grid>(self.rank-1))[0])>=1:
								for j2 in range(len(np.where(temp_grid>(self.rank-1))[0])):
										temp_grid=temp_grid[:-1]
										temp_grid=np.concatenate([np.array([temp_grid[0]-1]),temp_grid])    
								grid_set.append(temp_grid.tolist())
						else:
								grid_set.append(temp_grid.tolist())
        ### addding local permutation to the grid set
		temp_permutation_set=[]
		temp_permutation_set.append(permute.reshape((self.size)).tolist())
		for i in range(self.size):
				for j in range(i+1,self.size):
						trans=np.array([i,j])
						temp_permute=np.copy(permute.reshape((self.size)))
						temp_permute[trans]=temp_permute[trans[::-1]]
						temp_permutation_set.append(temp_permute.tolist())          
		grid_set.append(temp_permutation_set)                

		#### generating the optimize grid_set
		optimize_grid_set=[]
		optimize_grid_set.append(grid_set[0])
		for i in range(1,self.size):
				optimize_grid_set.append(rank.reshape((self.size))[i].tolist())
		optimize_grid_set.append(permute.reshape((self.size)).tolist())
		self.optimize_grid_set=optimize_grid_set

        #### generating the init_adj_mat       
		adj_matrix_set=[]
		for i in range(len(optimize_grid_set[0])):
				rank_temp=np.concatenate([np.array([optimize_grid_set[0][i]]),np.array(optimize_grid_set[1:self.size])]).reshape((self.size,1))
				permute_temp=np.array(optimize_grid_set[-1]).reshape((self.size,1))
				adj_matrix_temp=np.rec.fromarrays((out, rank_temp))
				adj_matrix_temp=np.rec.fromarrays((adj_matrix_temp, permute_temp))
				adj_matrix_set.append(adj_matrix_temp)        

		if self.Rse_interpolation_on==1:
				self.adjmatrix_set_temp=adj_matrix_set.copy()
				adj_matrix_set_new=[]
				adj_matrix_set_new.append(np.copy(adj_matrix_set[0]))
				adj_matrix_set_new.append(np.copy(adj_matrix_set[int(np.ceil(len(adj_matrix_set)/2))-1]))      
				adj_matrix_set_new.append(np.copy(adj_matrix_set[-1]))
				adj_matrix_set=adj_matrix_set_new.copy()
				self.evaluation_numbers=self.evaluation_numbers+len(adj_matrix_set)
		else:
			self.evaluation_numbers=self.evaluation_numbers+len(optimize_grid_set[0])

		return adj_matrix_set

	def __evolve__(self):

		def elimination(island,generation):

			#### check if structure remains unchange from left to right (from the first rank to the last permutation)
			if generation>2*self.size*self.Local_Opt_Iter+1:
				temp_structure=island['indv'][island['rank'][0]].adj_matrix
				adj_matrix_center=np.copy(self.center_structure)

				rank=np.arange(self.size)
				for i in range(self.size):
						rank[i]=adj_matrix_center[i][0][0][1]

				permute=np.arange(0, self.size)
				for i in range(self.size):
						permute[i]=adj_matrix_center[i][0][1]

				adj_matrix_temp=np.copy(temp_structure)

				rank_temp=np.arange(self.size)
				for i in range(self.size):
						rank_temp[i]=adj_matrix_temp[i][0][0][1]

				permute_temp=np.arange(0, self.size)
				for i in range(self.size):
						permute_temp[i]=adj_matrix_temp[i][0][1]

				if np.sum(np.absolute(rank-rank_temp))==0 and np.sum(np.absolute(permute-permute_temp))==0:
						self.structure_unchange_count=self.structure_unchange_count+1
						if self.local_reinitilize_update_flag==1:
							self.meetsamestructure_local_reupdate_flag=1

				if self.structure_unchange_count==self.size+1:
						self.local_reupdate_flag=1
						self.times_iteration_stop=self.times_iteration_stop+1

			###### Generate new center structure
			if self.local_structure_update_flag==1 or self.local_reupdate_flag==1 or self.meetsamestructure_local_reupdate_flag==1:
				self.times_of_LocalSampling=self.times_of_LocalSampling+1
				self.times_of_LocalUpdate=1
				self.local_structure_update_flag=0
				self.structure_unchange_count=0
				self.update_index=0
				if self.times_of_LocalSampling>self.Rse_interpolation_Times and self.Rse_interpolation_on==1:
						self.Local_Step = self.kwargs['Local_Step_main']

				if island['total'][island['rank'][0]]<=self.temp_best_local_structure_fitness:
						self.temp_best_local_structure=island['indv'][island['rank'][0]].adj_matrix
						self.temp_best_local_structure_fitness=island['total'][island['rank'][0]]

				##### The best structure
				if island['total'][island['rank'][0]]<=self.temp_best_local_structure_fitness and island['total'][island['rank'][0]]<=self.center_structure_fitness and self.meetsamestructure_local_reupdate_flag==0:
						self.best_generation=generation
						self.best_individual_numbers_evaluation=self.evaluation_numbers

				if self.times_of_LocalSampling==self.Rse_interpolation_Times+1 and self.Rse_interpolation_on==1:
						if self.temp_best_local_structure_fitness<=self.center_structure_fitness:
							self.center_structure=self.temp_best_local_structure
							self.center_structure_fitness=999999
							self.meetsamestructure_local_reupdate_flag=0 
							self.local_reupdate_flag=0
						else:
							self.center_structure_fitness=999999
							self.meetsamestructure_local_reupdate_flag=0
							self.local_reupdate_flag=0

				if self.temp_best_local_structure_fitness<=self.center_structure_fitness and self.local_reupdate_flag==0 and self.meetsamestructure_local_reupdate_flag==0:

					if self.times_of_LocalSampling==self.Rse_interpolation_Times+1 and self.Rse_interpolation_on==1:
							self.center_structure_fitness=999999
							adj_matrix_temp=np.copy(self.center_structure)
							self.local_reinitilize_update_flag=0
					else:
						self.local_reinitilize_update_flag=0
						adj_matrix_temp=np.copy(self.temp_best_local_structure)
						self.center_structure_fitness=self.temp_best_local_structure_fitness
						self.center_structure=self.temp_best_local_structure


					out=np.arange(self.size)
					for i in range(self.size):
							out[i]=adj_matrix_temp[i][0][0][0]
			
					rank=np.arange(self.size)
					for i in range(self.size):
							rank[i]=adj_matrix_temp[i][0][0][1]

					permute=np.arange(0, self.size)
					for i in range(self.size):
							permute[i]=adj_matrix_temp[i][0][1]
					out=out.reshape((self.size,1))
					rank=rank.reshape((self.size,1))
					permute=permute.reshape((self.size,1))

					grid_set=[]
					for i in range(self.size):
							temp_r=np.arange(1,self.Local_Step+1)
							temp_l=-1*temp_r
							temp_l=temp_l[::-1]
							temp_grid=np.concatenate([temp_l+rank[i],rank[i],rank[i]+temp_r])
							if len(temp_grid)>=self.rank-1:
									grid_set.append(np.arange(1,self.rank).tolist())
							else:
									if len(np.where(temp_grid<1)[0])>=1:
											for j1 in range(len(np.where(temp_grid<1)[0])):
													temp_grid=np.delete(temp_grid,0)
													temp_grid=np.concatenate([temp_grid,np.array([temp_grid[-1]+1])])  
											grid_set.append(temp_grid.tolist())
									elif len(np.where(temp_grid>(self.rank-1))[0])>=1:
											for j2 in range(len(np.where(temp_grid>(self.rank-1))[0])):
													temp_grid=temp_grid[:-1]
													temp_grid=np.concatenate([np.array([temp_grid[0]-1]),temp_grid])    
											grid_set.append(temp_grid.tolist())
									else:
											grid_set.append(temp_grid.tolist())
			        ### addding local permutation to the grid set
					temp_permutation_set=[]
					temp_permutation_set.append(permute.reshape((self.size)).tolist())
					for i in range(self.size):
							for j in range(i+1,self.size):
									trans=np.array([i,j])
									temp_permute=np.copy(permute.reshape((self.size)))
									temp_permute[trans]=temp_permute[trans[::-1]]
									temp_permutation_set.append(temp_permute.tolist())          
					grid_set.append(temp_permutation_set)                
			        #### generating the optimize grid_set
					optimize_grid_set=[]
					optimize_grid_set.append(grid_set[0])
					for i in range(1,self.size):
							optimize_grid_set.append(rank.reshape((self.size))[i].tolist())
					optimize_grid_set.append(permute.reshape((self.size)).tolist())
					self.optimize_grid_set=optimize_grid_set

					######## Finding the RSE propogation index
					temp_results=int(rank[0])
					for i in range(len(grid_set[0])):
						if np.sum(np.absolute(np.array([grid_set[0][i]])-np.array([temp_results])))==0:
							self.rse_propagate_index=i
							break
					self.rse_propagate=island['estimated_rse'][island['rank'][0]]

					if self.times_of_LocalSampling==self.Rse_interpolation_Times+1 and self.Rse_interpolation_on==1:
						self.rse_propagate=999999
						self.rse_propagate_index=0
						self.Rse_interpolation_on=0

			        #### generating the init_adj_mat       
					adj_matrix_set=[]
					for i in range(len(optimize_grid_set[0])):
							rank_temp=np.concatenate([np.array([optimize_grid_set[0][i]]),np.array(optimize_grid_set[1:self.size])]).reshape((self.size,1))
							permute_temp=np.array(optimize_grid_set[-1]).reshape((self.size,1))
							adj_matrix_temp=np.rec.fromarrays((out, rank_temp))
							adj_matrix_temp=np.rec.fromarrays((adj_matrix_temp, permute_temp))
							adj_matrix_set.append(adj_matrix_temp)        

					if self.Rse_interpolation_on==1:
							self.adjmatrix_set_temp=adj_matrix_set.copy()
							adj_matrix_set_new=[]

							adj_matrix_set_new.append(np.copy(adj_matrix_set[0]))
							adj_matrix_set_new.append(np.copy(adj_matrix_set[int(np.ceil(len(adj_matrix_set)/2))-1]))
							adj_matrix_set_new.append(np.copy(adj_matrix_set[-1]))

							adj_matrix_set=adj_matrix_set_new.copy()
							self.evaluation_numbers=self.evaluation_numbers+len(adj_matrix_set)
					else:
						self.evaluation_numbers=self.evaluation_numbers+len(optimize_grid_set[0])

					if len(island['indv'])<len(adj_matrix_set):
						for j in range(len(adj_matrix_set)-len(island['indv'])):
							island['indv'].append(DummyIndv())
						for i in range(len(adj_matrix_set)):
											island['indv'][i].adj_matrix=adj_matrix_set[i]
											island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))
					elif len(island['indv'])>len(adj_matrix_set):
						for j in range(len(island['indv'])-len(adj_matrix_set)):
							island['indv'].pop()
						for i in range(len(adj_matrix_set)):
											island['indv'][i].adj_matrix=adj_matrix_set[i]
											island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))
					else:
						for i in range(len(adj_matrix_set)):
											island['indv'][i].adj_matrix=adj_matrix_set[i]
											island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))
				if self.temp_best_local_structure_fitness>self.center_structure_fitness or self.local_reupdate_flag==1 or self.meetsamestructure_local_reupdate_flag==1:
					self.local_reinitilize_update_flag=1
					if self.local_reupdate_flag==1:
						adj_matrix_temp=np.copy(self.temp_best_local_structure)
						self.center_structure_fitness=self.temp_best_local_structure_fitness
						self.center_structure=self.temp_best_local_structure
					else:
						adj_matrix_temp=np.copy(self.center_structure)
					self.local_reupdate_flag=0
					self.meetsamestructure_local_reupdate_flag=0
					out=np.arange(self.size)
					for i in range(self.size):
							out[i]=adj_matrix_temp[i][0][0][0]
			
					rank=np.arange(self.size)
					for i in range(self.size):
							rank[i]=adj_matrix_temp[i][0][0][1]

					permute=np.arange(0, self.size)
					for i in range(self.size):
							permute[i]=adj_matrix_temp[i][0][1]
					out=out.reshape((self.size,1))
					rank=rank.reshape((self.size,1))
					permute=permute.reshape((self.size,1))


					grid_set=[]
					for i in range(self.size):
							temp_r=np.arange(1,self.Local_Step+1)
							temp_l=-1*temp_r
							temp_l=temp_l[::-1]
							temp_grid=np.concatenate([temp_l+rank[i],rank[i],rank[i]+temp_r])
							if len(temp_grid)>=self.rank-1:
									grid_set.append(np.arange(1,self.rank).tolist())
							else:
									if len(np.where(temp_grid<1)[0])>=1:
											for j1 in range(len(np.where(temp_grid<1)[0])):
													temp_grid=np.delete(temp_grid,0)
													temp_grid=np.concatenate([temp_grid,np.array([temp_grid[-1]+1])])  
											grid_set.append(temp_grid.tolist())
									elif len(np.where(temp_grid>(self.rank-1))[0])>=1:
											for j2 in range(len(np.where(temp_grid>(self.rank-1))[0])):
													temp_grid=temp_grid[:-1]
													temp_grid=np.concatenate([np.array([temp_grid[0]-1]),temp_grid])    
											grid_set.append(temp_grid.tolist())
									else:
											grid_set.append(temp_grid.tolist())
			        ### addding local permutation to the grid set
					temp_permutation_set=[]
					temp_permutation_set.append(permute.reshape((self.size)).tolist())
					for i in range(self.size):
							for j in range(i+1,self.size):
									trans=np.array([i,j])
									temp_permute=np.copy(permute.reshape((self.size)))
									temp_permute[trans]=temp_permute[trans[::-1]]
									temp_permutation_set.append(temp_permute.tolist())          
					grid_set.append(temp_permutation_set)                
			        #### generating the optimize grid_set
					optimize_grid_set=[]
					optimize_grid_set.append(grid_set[0])
					for i in range(1,self.size+1):
							optimize_grid_set.append(choice(grid_set[i]))
					self.optimize_grid_set=optimize_grid_set

					self.rse_propagate=999999
					self.rse_propagate_index=0

					while np.sum(np.absolute(np.array(optimize_grid_set[1:self.size])-rank.reshape((self.size))[1:self.size]))==0 and np.sum(np.absolute(np.array(optimize_grid_set[-1])-permute.reshape((self.size))))==0:
						optimize_grid_set=[]
						optimize_grid_set.append(grid_set[0])
						for i in range(1,self.size+1):
								optimize_grid_set.append(choice(grid_set[i]))
						self.optimize_grid_set=optimize_grid_set
						self.rse_propagate=999999

			        #### generating the init_adj_mat       
					adj_matrix_set=[]
					for i in range(len(optimize_grid_set[0])):
							rank_temp=np.concatenate([np.array([optimize_grid_set[0][i]]),np.array(optimize_grid_set[1:self.size])]).reshape((self.size,1))
							permute_temp=np.array(optimize_grid_set[-1]).reshape((self.size,1))
							adj_matrix_temp=np.rec.fromarrays((out, rank_temp))
							adj_matrix_temp=np.rec.fromarrays((adj_matrix_temp, permute_temp))
							adj_matrix_set.append(adj_matrix_temp)        

					if self.Rse_interpolation_on==1:
							self.adjmatrix_set_temp=adj_matrix_set.copy()
							adj_matrix_set_new=[]

							adj_matrix_set_new.append(np.copy(adj_matrix_set[0]))
							adj_matrix_set_new.append(np.copy(adj_matrix_set[int(np.ceil(len(adj_matrix_set)/2))-1]))
							adj_matrix_set_new.append(np.copy(adj_matrix_set[-1]))

							adj_matrix_set=adj_matrix_set_new.copy()
							self.evaluation_numbers=self.evaluation_numbers+len(adj_matrix_set)
					else:
						self.evaluation_numbers=self.evaluation_numbers+len(optimize_grid_set[0])


					if len(island['indv'])<len(adj_matrix_set):
						for j in range(len(adj_matrix_set)-len(island['indv'])):
							island['indv'].append(DummyIndv())
						for i in range(len(adj_matrix_set)):
											island['indv'][i].adj_matrix=adj_matrix_set[i]
											island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))
					elif len(island['indv'])>len(adj_matrix_set):
						for j in range(len(island['indv'])-len(adj_matrix_set)):
							island['indv'].pop()
						for i in range(len(adj_matrix_set)):
											island['indv'][i].adj_matrix=adj_matrix_set[i]
											island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))
					else:
						for i in range(len(adj_matrix_set)):
											island['indv'][i].adj_matrix=adj_matrix_set[i]
											island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))

			else:
				#### position initilization
				temp1=np.arange(self.size+1)
				temp2=np.arange(1,self.size)
				position_set=np.concatenate([np.tile(np.concatenate([temp1,temp2[::-1]]),self.Local_Opt_Iter),np.array([0])])
				former_index=position_set[self.times_of_LocalUpdate-1]
				update_index=position_set[self.times_of_LocalUpdate]
				self.update_index=update_index
				######## break the center_structure and generate the grid_set
				adj_matrix_temp=np.copy(self.center_structure)
				out=np.arange(self.size)
				for i in range(self.size):
						out[i]=adj_matrix_temp[i][0][0][0]
		
				rank=np.arange(self.size)
				for i in range(self.size):
						rank[i]=adj_matrix_temp[i][0][0][1]

				permute=np.arange(0, self.size)
				for i in range(self.size):
						permute[i]=adj_matrix_temp[i][0][1]
				out=out.reshape((self.size,1))
				rank=rank.reshape((self.size,1))
				permute=permute.reshape((self.size,1))

				grid_set=[]
				for i in range(self.size):
						temp_r=np.arange(1,self.Local_Step+1)
						temp_l=-1*temp_r
						temp_l=temp_l[::-1]
						temp_grid=np.concatenate([temp_l+rank[i],rank[i],rank[i]+temp_r])
						if len(temp_grid)>=self.rank-1:
								grid_set.append(np.arange(1,self.rank).tolist())
						else:
								if len(np.where(temp_grid<1)[0])>=1:
										for j1 in range(len(np.where(temp_grid<1)[0])):
												temp_grid=np.delete(temp_grid,0)
												temp_grid=np.concatenate([temp_grid,np.array([temp_grid[-1]+1])])  
										grid_set.append(temp_grid.tolist())
								elif len(np.where(temp_grid>(self.rank-1))[0])>=1:
										for j2 in range(len(np.where(temp_grid>(self.rank-1))[0])):
												temp_grid=temp_grid[:-1]
												temp_grid=np.concatenate([np.array([temp_grid[0]-1]),temp_grid])    
										grid_set.append(temp_grid.tolist())
								else:
										grid_set.append(temp_grid.tolist())
				temp_permutation_set=[]
				temp_permutation_set.append(permute.reshape((self.size)).tolist())
				for i in range(self.size):
						for j in range(i+1,self.size):
								trans=np.array([i,j])
								temp_permute=np.copy(permute.reshape((self.size)))
								temp_permute[trans]=temp_permute[trans[::-1]]
								temp_permutation_set.append(temp_permute.tolist())          
				grid_set.append(temp_permutation_set)

				######## Updating the optimize_grid_set based on the former results
				temp_optimize_grid_set=self.optimize_grid_set.copy()
				temp_optimize_grid_set[former_index]=temp_optimize_grid_set[former_index][island['rank'][0]]

				temp_results=temp_optimize_grid_set[update_index]

				temp_optimize_grid_set[update_index]=grid_set[update_index]
				self.optimize_grid_set=temp_optimize_grid_set


				######## Finding the RSE propogation index
				for i in range(len(temp_optimize_grid_set[update_index])):
					if np.sum(np.absolute(np.array([temp_optimize_grid_set[update_index][i]])-np.array([temp_results])))==0:
						self.rse_propagate_index=i
						break
				self.rse_propagate=island['estimated_rse'][island['rank'][0]]



				if self.times_of_LocalUpdate==1:
					self.temp_best_local_structure=island['indv'][island['rank'][0]].adj_matrix
					self.temp_best_local_structure_fitness=island['total'][island['rank'][0]]
				else:
					if island['total'][island['rank'][0]]<=self.temp_best_local_structure_fitness:
						self.temp_best_local_structure=island['indv'][island['rank'][0]].adj_matrix
						self.temp_best_local_structure_fitness=island['total'][island['rank'][0]]

				##### The best structure
				if island['total'][island['rank'][0]]<=self.temp_best_local_structure_fitness and island['total'][island['rank'][0]]<=self.center_structure_fitness:
						self.best_generation=generation
						self.best_individual_numbers_evaluation=self.evaluation_numbers



				######## Generating the next adj_matrix

				adj_matrix_set=[]
				for i in range(len(temp_optimize_grid_set[update_index])):
						rank_temp=np.arange(self.size)
						for j in range(self.size):
							if j==update_index:
								rank_temp[j]=temp_optimize_grid_set[j][i]
							else:
								rank_temp[j]=temp_optimize_grid_set[j]
						rank_temp=rank_temp.reshape((self.size,1))
						if update_index==self.size:
							permute_temp=np.array(temp_optimize_grid_set[-1][i]).reshape((self.size,1))
						else:
							permute_temp=np.array(temp_optimize_grid_set[-1]).reshape((self.size,1))
						adj_matrix_temp=np.rec.fromarrays((out, rank_temp))
						adj_matrix_temp=np.rec.fromarrays((adj_matrix_temp, permute_temp))
						adj_matrix_set.append(adj_matrix_temp)

				if self.Rse_interpolation_on==1 and self.update_index!=self.size:
						self.adjmatrix_set_temp=adj_matrix_set.copy()
						adj_matrix_set_new=[]

						adj_matrix_set_new.append(np.copy(adj_matrix_set[0]))
						adj_matrix_set_new.append(np.copy(adj_matrix_set[int(np.ceil(len(adj_matrix_set)/2))-1]))
						adj_matrix_set_new.append(np.copy(adj_matrix_set[-1]))

						adj_matrix_set=adj_matrix_set_new.copy()
						self.evaluation_numbers=self.evaluation_numbers+len(adj_matrix_set)
				else:
					self.evaluation_numbers=self.evaluation_numbers+len(temp_optimize_grid_set[update_index])



				self.times_of_LocalUpdate=self.times_of_LocalUpdate+1
				######## Generating the next individual
				if len(island['indv'])<len(adj_matrix_set):
					for j in range(len(adj_matrix_set)-len(island['indv'])):
						island['indv'].append(DummyIndv())
					for i in range(len(adj_matrix_set)):
										island['indv'][i].adj_matrix=adj_matrix_set[i]
										island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))
				elif len(island['indv'])>len(adj_matrix_set):
					for j in range(len(island['indv'])-len(adj_matrix_set)):
						island['indv'].pop()
					for i in range(len(adj_matrix_set)):
										island['indv'][i].adj_matrix=adj_matrix_set[i]
										island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))
				else:
					for i in range(len(adj_matrix_set)):
										island['indv'][i].adj_matrix=adj_matrix_set[i]
										island['indv'][i].parents=('generation of the best result:%d'%(self.best_generation)+'|'+''.join([str(self.times_of_LocalSampling)])+','+''.join([str(self.times_of_LocalUpdate)])+'|'+''.join([str(self.times_iteration_stop)])+'|#Eva of the best result:'+''.join([str(self.best_individual_numbers_evaluation)]))





			###### check center_structure update
			Local_Update_Step_Num,update_position=divmod(self.times_of_LocalUpdate,2*self.size*self.Local_Opt_Iter+1)
			if update_position==0:
				self.local_structure_update_flag=1


		for idx, (k, v) in enumerate(self.societies.items()):
				elimination(v,self.n_generation)



	def __evaluate__(self):

		def score2rank(island, idx):
			sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))
			score = island['score']
			sparsity_score = [ s for s, l in score ]
			loss_score = [ l for s, l in score ]
			##### The Rse interpolation

			if self.Rse_interpolation_on==1 and self.update_index!=self.size:
				if self.n_generation<=1:
				#### Rse interpolation
					#### left part interpolation
					inter_rse_left=[]
					Y_axis=loss_score[0:2]
					X_axis=[-1*self.Local_Step,0]
					X_predict_axis=np.arange(-1*self.Local_Step,1,1)
					X_predict_axis=np.setdiff1d(X_predict_axis,np.array(X_axis))
					z1 = np.polyfit(X_axis, Y_axis, 1)
					inter_rse_left=(z1[0]*X_predict_axis+z1[1]).tolist()
					#### right part interpolation
					inter_rse_right=[]
					Y_axis=loss_score[1:3]
					X_axis=[0,self.Local_Step]
					X_predict_axis=np.arange(0,self.Local_Step+1,1)
					X_predict_axis=np.setdiff1d(X_predict_axis,np.array(X_axis))
					z1 = np.polyfit(X_axis, Y_axis, 1)
					inter_rse_right=(z1[0]*X_predict_axis+z1[1]).tolist()
					#### final rse
					loss_score_temp=[]
					loss_score_temp=[loss_score[0]]+inter_rse_left+[loss_score[1]]+inter_rse_right+[loss_score[2]]

					#### sparsity
					inter_sparsity=[]
					inter_actual_elements=[]
					for j in range(len(loss_score_temp)):
						if j!=0 and j!=len(loss_score_temp)-1 and j!=int(np.ceil(len(loss_score_temp)/2))-1:
							adj_matrix=np.copy(self.adjmatrix_set_temp[j])
							dim = adj_matrix.shape[0]
							adj_matrix_T=np.copy(adj_matrix)
							size_data=np.arange(dim)
							for i in range(dim):
									size_data[i]=adj_matrix_T[i][0][0][0]
		
							rank=np.arange(dim)
							for i in range(dim):
									rank[i]=adj_matrix_T[i][0][0][1]

							permute_code=np.arange(0, dim)
							for i in range(dim):
									permute_code[i]=adj_matrix_T[i][0][1]
							adj_matrix_R = np.diag(size_data)
							temp=np.arange(dim-1)
							temp=temp[::-1]
							temp[0]=dim-3
							connection_index = []
							connection_index.append(0)
							for i in range(dim):
									if i==1:
											connection_index.append(connection_index[-1]+1)
									else:
											if i==0:
													connection_index.append(connection_index[-1]+(temp[i]+1))
											else:
													connection_index.append(connection_index[-1]+(temp[i-1]+1))
							connection_index=connection_index[0:dim]
							connection =rank.tolist()
							index_tuple=np.triu_indices(dim, 1)
							index_tuple1=index_tuple[0]
							index_tuple2=index_tuple[1]
							index_tuple1=index_tuple1[connection_index]
							index_tuple2=index_tuple2[connection_index]
							index_tuple=[index_tuple1,index_tuple2]
							index_tuple=tuple(index_tuple)
							adj_matrix_R[index_tuple] = connection
							adj_matrix_R[np.tril_indices(dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(dim, -1)]
							index=np.diag_indices_from(adj_matrix_R)
							adj_matrix_R[index]=0

							permute=permute_code
					
							permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
							for i in range(dim):
									permutation_matrix[permute[i],i] = 1

							adj_matrix_R=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())

				
							adj_matrix_k = np.copy(adj_matrix_R)
							adj_matrix_k[adj_matrix_k==0] = 1

							present_elements = np.prod(np.diag(adj_matrix_k))
							actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(dim) ])
							sparsity = actual_elements/present_elements
							inter_actual_elements.append(actual_elements)
							inter_sparsity.append(sparsity)
					if len(inter_sparsity)==2:
						sparsity_score_temp=[]
						sparsity_score_temp=[sparsity_score[0]]+[inter_sparsity[0]]+[sparsity_score[1]]+[inter_sparsity[1]]+[sparsity_score[2]]
					else:
						sparsity_score_temp=[]
						sparsity_score_temp=[sparsity_score[0]]+inter_sparsity[0:(int(len(inter_sparsity)/2))]+[sparsity_score[1]]+inter_sparsity[(int(len(inter_sparsity)/2)):len(inter_sparsity)]+[sparsity_score[2]]

					#### adding individual (to include the intepolation ones)
					inter_individual=[]
					temp=0
					for j in range(len(sparsity_score_temp)):
						if j!=0 and j!=len(sparsity_score_temp)-1 and j!=int(np.ceil(len(sparsity_score_temp)/2))-1:
							inter_individual.append(DummyIndv())
							inter_individual[-1].adj_matrix=np.copy(self.adjmatrix_set_temp[j])
							inter_individual[-1].sparsity=sparsity_score_temp[j]
							inter_individual[-1].repeat_loss=[]
							inter_individual[-1].repeat_loss.append(loss_score_temp[j])
							inter_individual[-1].parents=None
							inter_individual[-1].actual_elements=inter_actual_elements[temp]
							temp=temp+1


					if len(inter_individual)==2:
							island['indv']=[island['indv'][0]]+[inter_individual[0]]+[island['indv'][1]]+[inter_individual[1]]+[island['indv'][2]]
					else:
							island['indv']=[island['indv'][0]]+inter_individual[0:(int(len(inter_individual)/2))]+[island['indv'][1]]+inter_individual[(int(len(inter_individual)/2)):len(inter_individual)]+[island['indv'][2]]



					for j in range(len(island['indv'])):
						island['indv'][j].scope='{}/{}/{:03d}'.format(self.name, self.society_name, j)

					sparsity_score=sparsity_score_temp
					loss_score=loss_score_temp

				else:
					if self.rse_propagate_index==0 or self.rse_propagate_index==len(self.adjmatrix_set_temp)-1 or self.rse_propagate_index==int(np.ceil(len(self.adjmatrix_set_temp)/2))-1:

						if self.rse_propagate_index==0:
							index_temp=0
						elif self.rse_propagate_index==len(self.adjmatrix_set_temp)-1:
							index_temp=2
						else:
							index_temp=1
						if loss_score[index_temp]>self.rse_propagate:
							loss_score[index_temp]=self.rse_propagate

						#### Rse interpolation

						#### left part interpolation
						inter_rse_left=[]
						Y_axis=loss_score[0:2]
						X_axis=[-1*self.Local_Step,0]
						X_predict_axis=np.arange(-1*self.Local_Step,1,1)
						X_predict_axis=np.setdiff1d(X_predict_axis,np.array(X_axis))
						z1 = np.polyfit(X_axis, Y_axis, 1)
						inter_rse_left=(z1[0]*X_predict_axis+z1[1]).tolist()
						#### right part interpolation
						inter_rse_right=[]
						Y_axis=loss_score[1:3]
						X_axis=[0,self.Local_Step]
						X_predict_axis=np.arange(0,self.Local_Step+1,1)
						X_predict_axis=np.setdiff1d(X_predict_axis,np.array(X_axis))
						z1 = np.polyfit(X_axis, Y_axis, 1)
						inter_rse_right=(z1[0]*X_predict_axis+z1[1]).tolist()
						#### final rse
						loss_score_temp=[]
						loss_score_temp=[loss_score[0]]+inter_rse_left+[loss_score[1]]+inter_rse_right+[loss_score[2]]


						#### sparsity
						inter_sparsity=[]
						inter_actual_elements=[]
						for j in range(len(loss_score_temp)):
							if j!=0 and j!=len(loss_score_temp)-1 and j!=int(np.ceil(len(loss_score_temp)/2))-1:
								adj_matrix=np.copy(self.adjmatrix_set_temp[j])
								dim = adj_matrix.shape[0]
								adj_matrix_T=np.copy(adj_matrix)
								size_data=np.arange(dim)
								for i in range(dim):
										size_data[i]=adj_matrix_T[i][0][0][0]
		
								rank=np.arange(dim)
								for i in range(dim):
										rank[i]=adj_matrix_T[i][0][0][1]

								permute_code=np.arange(0, dim)
								for i in range(dim):
										permute_code[i]=adj_matrix_T[i][0][1]
								adj_matrix_R = np.diag(size_data)
								temp=np.arange(dim-1)
								temp=temp[::-1]
								temp[0]=dim-3
								connection_index = []
								connection_index.append(0)
								for i in range(dim):
										if i==1:
												connection_index.append(connection_index[-1]+1)
										else:
												if i==0:
														connection_index.append(connection_index[-1]+(temp[i]+1))
												else:
														connection_index.append(connection_index[-1]+(temp[i-1]+1))
								connection_index=connection_index[0:dim]
								connection =rank.tolist()
								index_tuple=np.triu_indices(dim, 1)
								index_tuple1=index_tuple[0]
								index_tuple2=index_tuple[1]
								index_tuple1=index_tuple1[connection_index]
								index_tuple2=index_tuple2[connection_index]
								index_tuple=[index_tuple1,index_tuple2]
								index_tuple=tuple(index_tuple)
								adj_matrix_R[index_tuple] = connection
								adj_matrix_R[np.tril_indices(dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(dim, -1)]
								index=np.diag_indices_from(adj_matrix_R)
								adj_matrix_R[index]=0

								permute=permute_code
					
								permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
								for i in range(dim):
										permutation_matrix[permute[i],i] = 1

								adj_matrix_R=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())

				
								adj_matrix_k = np.copy(adj_matrix_R)
								adj_matrix_k[adj_matrix_k==0] = 1

								present_elements = np.prod(np.diag(adj_matrix_k))
								actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(dim) ])
								sparsity = actual_elements/present_elements
								inter_actual_elements.append(actual_elements)
								inter_sparsity.append(sparsity)
						if len(inter_sparsity)==2:
							sparsity_score_temp=[]
							sparsity_score_temp=[sparsity_score[0]]+[inter_sparsity[0]]+[sparsity_score[1]]+[inter_sparsity[1]]+[sparsity_score[2]]
						else:
							sparsity_score_temp=[]
							sparsity_score_temp=[sparsity_score[0]]+inter_sparsity[0:(int(len(inter_sparsity)/2))]+[sparsity_score[1]]+inter_sparsity[(int(len(inter_sparsity)/2)):len(inter_sparsity)]+[sparsity_score[2]]


						#### adding individual (to include the intepolation ones)
						inter_individual=[]
						temp=0
						for j in range(len(sparsity_score_temp)):
							if j!=0 and j!=len(sparsity_score_temp)-1 and j!=int(np.ceil(len(sparsity_score_temp)/2))-1:
								inter_individual.append(DummyIndv())
								inter_individual[-1].adj_matrix=np.copy(self.adjmatrix_set_temp[j])
								inter_individual[-1].sparsity=sparsity_score_temp[j]
								inter_individual[-1].repeat_loss=[]
								inter_individual[-1].repeat_loss.append(loss_score_temp[j])
								inter_individual[-1].parents=island['indv'][0].parents
								inter_individual[-1].actual_elements=inter_actual_elements[temp]
								temp=temp+1


						if len(inter_individual)==2:
								island['indv']=[island['indv'][0]]+[inter_individual[0]]+[island['indv'][1]]+[inter_individual[1]]+[island['indv'][2]]
						else:
								island['indv']=[island['indv'][0]]+inter_individual[0:(int(len(inter_individual)/2))]+[island['indv'][1]]+inter_individual[(int(len(inter_individual)/2)):len(inter_individual)]+[island['indv'][2]]



						for j in range(len(island['indv'])):
							island['indv'][j].scope='{}/{}/{:03d}'.format(self.name, self.society_name, j)

						sparsity_score=sparsity_score_temp
						loss_score=loss_score_temp

					else:
						#### Rse interpolation

						#### left part interpolation
						inter_rse_left=[]
						Y_axis=loss_score[0:2]
						X_axis=[-1*self.Local_Step,0]
						X_predict_axis=np.arange(-1*self.Local_Step,1,1)
						X_predict_axis=np.setdiff1d(X_predict_axis,np.array(X_axis))
						z1 = np.polyfit(X_axis, Y_axis, 1)
						inter_rse_left=(z1[0]*X_predict_axis+z1[1]).tolist()
						#### right part interpolation
						inter_rse_right=[]
						Y_axis=loss_score[1:3]
						X_axis=[0,self.Local_Step]
						X_predict_axis=np.arange(0,self.Local_Step+1,1)
						X_predict_axis=np.setdiff1d(X_predict_axis,np.array(X_axis))
						z1 = np.polyfit(X_axis, Y_axis, 1)
						inter_rse_right=(z1[0]*X_predict_axis+z1[1]).tolist()
						#### final rse
						loss_score_temp=[]
						loss_score_temp=[loss_score[0]]+inter_rse_left+[loss_score[1]]+inter_rse_right+[loss_score[2]]

						#### sparsity
						inter_sparsity=[]
						inter_actual_elements=[]
						for j in range(len(loss_score_temp)):
							if j!=0 and j!=len(loss_score_temp)-1 and j!=int(np.ceil(len(loss_score_temp)/2))-1:
								adj_matrix=np.copy(self.adjmatrix_set_temp[j])
								dim = adj_matrix.shape[0]
								adj_matrix_T=np.copy(adj_matrix)
								size_data=np.arange(dim)
								for i in range(dim):
										size_data[i]=adj_matrix_T[i][0][0][0]
		
								rank=np.arange(dim)
								for i in range(dim):
										rank[i]=adj_matrix_T[i][0][0][1]

								permute_code=np.arange(0, dim)
								for i in range(dim):
										permute_code[i]=adj_matrix_T[i][0][1]
								adj_matrix_R = np.diag(size_data)
								temp=np.arange(dim-1)
								temp=temp[::-1]
								temp[0]=dim-3
								connection_index = []
								connection_index.append(0)
								for i in range(dim):
										if i==1:
												connection_index.append(connection_index[-1]+1)
										else:
												if i==0:
														connection_index.append(connection_index[-1]+(temp[i]+1))
												else:
														connection_index.append(connection_index[-1]+(temp[i-1]+1))
								connection_index=connection_index[0:dim]
								connection =rank.tolist()
								index_tuple=np.triu_indices(dim, 1)
								index_tuple1=index_tuple[0]
								index_tuple2=index_tuple[1]
								index_tuple1=index_tuple1[connection_index]
								index_tuple2=index_tuple2[connection_index]
								index_tuple=[index_tuple1,index_tuple2]
								index_tuple=tuple(index_tuple)
								adj_matrix_R[index_tuple] = connection
								adj_matrix_R[np.tril_indices(dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(dim, -1)]
								index=np.diag_indices_from(adj_matrix_R)
								adj_matrix_R[index]=0

								permute=permute_code
					
								permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
								for i in range(dim):
										permutation_matrix[permute[i],i] = 1

								adj_matrix_R=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())

				
								adj_matrix_k = np.copy(adj_matrix_R)
								adj_matrix_k[adj_matrix_k==0] = 1

								present_elements = np.prod(np.diag(adj_matrix_k))
								actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(dim) ])
								inter_actual_elements.append(actual_elements)
								sparsity = actual_elements/present_elements
								inter_sparsity.append(sparsity)
						if len(inter_sparsity)==2:
							sparsity_score_temp=[]
							sparsity_score_temp=[sparsity_score[0]]+[inter_sparsity[0]]+[sparsity_score[1]]+[inter_sparsity[1]]+[sparsity_score[2]]
						else:
							sparsity_score_temp=[]
							sparsity_score_temp=[sparsity_score[0]]+inter_sparsity[0:(int(len(inter_sparsity)/2))]+[sparsity_score[1]]+inter_sparsity[(int(len(inter_sparsity)/2)):len(inter_sparsity)]+[sparsity_score[2]]

						#### adding individual (to include the intepolation ones)
						inter_individual=[]
						temp=0
						for j in range(len(sparsity_score_temp)):
							if j!=0 and j!=len(sparsity_score_temp)-1 and j!=int(np.ceil(len(sparsity_score_temp)/2))-1:
								inter_individual.append(DummyIndv())
								inter_individual[-1].adj_matrix=np.copy(self.adjmatrix_set_temp[j])
								inter_individual[-1].sparsity=sparsity_score_temp[j]
								inter_individual[-1].repeat_loss=[]
								inter_individual[-1].repeat_loss.append(loss_score_temp[j])
								inter_individual[-1].parents=island['indv'][0].parents
								inter_individual[-1].actual_elements=inter_actual_elements[temp]
								temp=temp+1

						if len(inter_individual)==2:
								island['indv']=[island['indv'][0]]+[inter_individual[0]]+[island['indv'][1]]+[inter_individual[1]]+[island['indv'][2]]
						else:
								island['indv']=[island['indv'][0]]+inter_individual[0:(int(len(inter_individual)/2))]+[island['indv'][1]]+inter_individual[(int(len(inter_individual)/2)):len(inter_individual)]+[island['indv'][2]]


						for j in range(len(island['indv'])):
							island['indv'][j].scope='{}/{}/{:03d}'.format(self.name, self.society_name, j)

						sparsity_score=sparsity_score_temp
						loss_score=loss_score_temp
						if loss_score[self.rse_propagate_index]>self.rse_propagate:
							loss_score[self.rse_propagate_index]=self.rse_propagate

			else:
				if self.n_generation>1:
					if loss_score[self.rse_propagate_index]>self.rse_propagate:
						loss_score[self.rse_propagate_index]=self.rse_propagate




			if 'fitness_func' in self.kwargs.keys():
				if isinstance(self.kwargs['fitness_func'], list):
					fitness_func = self.kwargs['fitness_func'][idx]
				else:
					fitness_func = self.kwargs['fitness_func']
			else:		
				fitness_func = lambda s, l: 1*s+200*l
			
			total_score = [ fitness_func(s, l) for s, l in zip(sparsity_score, loss_score) ]



			island['rank'] = np.argsort(total_score)
			island['total'] = total_score
			island['estimated_rse']=loss_score.copy()


		# RANKING
		for idx, (k, v) in enumerate(self.societies.items()):
			v['score'] = [ (indv.sparsity ,np.min(indv.repeat_loss)) for indv in v['indv'] ]
			score2rank(v, idx)

	def distribute_indv(self, agent):
		if self.indv_to_distribute:
			indv = self.indv_to_distribute.pop(0)
			if np.log10(indv.sparsity)<1:
				agent.receive(indv)
				self.indv_to_collect.append(indv)
				logging.info('Assigned individual {} to agent {}.'.format(indv.scope, agent.sge_job_id))
			else:
				indv.collect(fake_loss=True)
				logging.info('Individual {} is killed due to its sparsity = {} / {}.'.format(indv.scope, np.log10(indv.sparsity), indv.sparsityB))

	def collect_indv(self):
		for indv in self.indv_to_collect:
			if indv.collect():
				logging.info('Collected individual result {}.'.format(indv.scope))
				self.indv_to_collect.remove(indv)

	def is_finished(self):
		if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
			return True
		else:
			return False

	def get_center(self):
			return self.Rse_interpolation_on,self.update_index,self.society_name,self.adjmatrix_set_temp,self.best_individual_numbers_evaluation,self.evaluation_numbers,self.local_reinitilize_update_flag,self.meetsamestructure_local_reupdate_flag,self.times_iteration_stop,self.local_reupdate_flag,self.structure_unchange_count,self.rse_propagate_index,self.rse_propagate,self.best_generation,self.times_of_LocalUpdate,self.times_of_LocalSampling,self.local_structure_update_flag,self.temp_best_local_structure,self.temp_best_local_structure_fitness,self.center_structure,self.center_structure_fitness,self.optimize_grid_set


class Agent(object):
	def __init__(self, **kwargs):
		super(Agent, self).__init__()
		self.kwargs = kwargs
		self.sge_job_id = self.kwargs['sge_job_id']

	def receive(self, indv):
		indv.deploy(self.sge_job_id)
		with open(base_folder+'/agent_pool/{}.POOL'.format(self.sge_job_id), 'a') as f:
			f.write(evoluation_goal)

	def is_available(self):
		return True if os.stat(base_folder+'/agent_pool/{}.POOL'.format(self.kwargs['sge_job_id'])).st_size == 0 else False

class Overlord(object):
	def __init__(self, max_generation=100, **kwargs):
		super(Overlord, self).__init__()
		self.dummy_func = lambda *args, **kwargs: None
		self.max_generation = max_generation
		self.current_generation = None
		self.previous_generation = None
		self.N_generation = 0
		self.kwargs = kwargs
		self.generation = kwargs['generation']
		self.generation_list = []
		self.available_agents = []
		self.known_agents = {}
		self.time = 0

		self.rse_interplo_flag =  kwargs['Rse_interpolation_on']

        
	def __call_with_interval__(self, func, interval):
		return func if self.time%interval == 0 else self.dummy_func

	def __tik__(self, sec):

		self.time += sec
		time.sleep(sec)

	def __check_available_agent__(self):
		self.available_agents.clear()
		agents = glob.glob(base_folder+'/agent_pool/*.POOL')
		agents_id = [ a.split('/')[-1][:-5] for a in agents ]

		for aid in list(self.known_agents.keys()):
			if aid not in agents_id:
				logging.info('Dead agent id = {} found!'.format(aid))
				self.known_agents.pop(aid, None)

		for aid in agents_id:
			if aid in self.known_agents.keys():
				if self.known_agents[aid].is_available():
					self.available_agents.append(self.known_agents[aid])
			else:
				self.known_agents[aid] = Agent(sge_job_id=aid)
				logging.info('New agent id = {} found!'.format(aid))

	def __assign_job__(self):
		self.__check_available_agent__()
		if len(self.available_agents)>0:
			for agent in self.available_agents:
				self.current_generation.distribute_indv(agent)

	def __collect_result__(self):
		self.current_generation.collect_indv()

	def __report_agents__(self):
		logging.info('Current number of known agents is {}.'.format(len(self.known_agents)))
		logging.info(list(self.known_agents.keys()))

	def __report_generation__(self):
		logging.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
		logging.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
		logging.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])

	def __generation__(self):
		if self.N_generation > self.max_generation:
			return False
		else:
			if self.current_generation is None:
				self.current_generation = self.generation(self.rse_interplo_flag,0,None,[],0,0,0,0,0,0,0,None,None,0,1,1,0,[],[],[],9999,[],0,name='generation_init', **self.kwargs)
				self.current_generation.indv_to_distribute = []

			if self.current_generation.is_finished():
				if self.previous_generation is not None:
					self.current_generation(**self.kwargs)
				self.N_generation += 1
				self.previous_generation = self.current_generation
				rse_interplo_flag,update_index,society_name,adjmatrix_set_temp,best_individual_numbers_evaluation,evaluation_numbers,local_reinitilize_update_flag,meetsamestructure_local_reupdate_flag,times_iteration_stop,local_reupdate_flag,structure_unchange_count,rse_propagate_index,rse_propagate,best_generation,times_of_LocalUpdate,times_of_LocalSampling,local_structure_update_flag,temp_best_local_structure,temp_best_local_structure_fitness,center_structure,center_structure_fitness, optimize_grid_set=self.previous_generation.get_center()


				self.current_generation = self.generation(rse_interplo_flag,update_index,society_name,adjmatrix_set_temp,best_individual_numbers_evaluation,evaluation_numbers,local_reinitilize_update_flag,meetsamestructure_local_reupdate_flag,times_iteration_stop,local_reupdate_flag,structure_unchange_count,rse_propagate_index,rse_propagate,best_generation,times_of_LocalUpdate,times_of_LocalSampling,local_structure_update_flag,temp_best_local_structure,temp_best_local_structure_fitness,center_structure,center_structure_fitness,optimize_grid_set,self.N_generation,self.previous_generation,  
														name='generation_{:03d}'.format(self.N_generation), **self.kwargs)

			return True

	def __call__(self):

			self.N_generation = 0
			self.current_generation = None
			self.previous_generation = None


			while self.__generation__():
				self.__call_with_interval__(self.__check_available_agent__, 4)()
				self.__call_with_interval__(self.__assign_job__, 4)()
				self.__call_with_interval__(self.__collect_result__, 4)()
				self.__call_with_interval__(self.__report_agents__, 180)()
				self.__call_with_interval__(self.__report_generation__, 160)()
				self.__tik__(2)

def score_summary(obj):
	logging.info('===== {} ====='.format(obj.name))

	for k, v in obj.societies.items():
		logging.info('===== ISLAND {} ====='.format(k))

		for idx, indv in enumerate(v['indv']):
			if idx == v['rank'][0]:
				logging.info('\033[31m{} | {:.3f} | RSE:{} | {:.12f} | {:.5f} | {}\033[0m'.format(indv.scope, np.log10(indv.sparsity), [ float('{:.12f}'.format(l)) for l in indv.repeat_loss ],v['estimated_rse'][idx], v['total'][idx], indv.parents))

				logging.info(indv.adj_matrix)

				dim = indv.adj_matrix.shape[0]
				adj_matrix_T=np.copy(indv.adj_matrix)
				size_data=np.arange(dim)
				for i in range(dim):
								size_data[i]=adj_matrix_T[i][0][0][0]
				rank=np.arange(dim)
				for i in range(dim):
								rank[i]=adj_matrix_T[i][0][0][1]

				permute_code=np.arange(0, dim)
				for i in range(dim):
								permute_code[i]=adj_matrix_T[i][0][1]
				adj_matrix_R = np.diag(size_data)
				temp=np.arange(dim-1)
				temp=temp[::-1]
				temp[0]=dim-3
				connection_index = []
				connection_index.append(0)
				for i in range(dim):
								if i==1:
										connection_index.append(connection_index[-1]+1)
								else:
										if i==0:
												connection_index.append(connection_index[-1]+(temp[i]+1))
										else:
												connection_index.append(connection_index[-1]+(temp[i-1]+1))
				connection_index=connection_index[0:dim]
				connection =rank.tolist()
				index_tuple=np.triu_indices(dim, 1)
				index_tuple1=index_tuple[0]
				index_tuple2=index_tuple[1]
				index_tuple1=index_tuple1[connection_index]
				index_tuple2=index_tuple2[connection_index]
				index_tuple=[index_tuple1,index_tuple2]
				index_tuple=tuple(index_tuple)
				adj_matrix_R[index_tuple] = connection
				adj_matrix_R[np.tril_indices(dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(dim, -1)]
				index=np.diag_indices_from(adj_matrix_R)
				adj_matrix_R[index]=0

				permute=permute_code

				permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
				for i in range(dim):
										permutation_matrix[permute[i],i] = 1
				adj_matrix_in=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())
				adj_matrix_in[adj_matrix_in==1] = 0
				logging.info(adj_matrix_in)
				logging.info('Parameters:{} '.format(indv.actual_elements))


			else:
				logging.info('{} | {:.3f} | RSE:{} | {:.12f} | {:.5f} | {}'.format(indv.scope, np.log10(indv.sparsity), [ float('{:.12f}'.format(l)) for l in indv.repeat_loss ], v['estimated_rse'][idx],v['total'][idx], indv.parents))
				logging.info(indv.adj_matrix)

				dim = indv.adj_matrix.shape[0]
				adj_matrix_T=np.copy(indv.adj_matrix)
				size_data=np.arange(dim)
				for i in range(dim):
								size_data[i]=adj_matrix_T[i][0][0][0]
				rank=np.arange(dim)
				for i in range(dim):
								rank[i]=adj_matrix_T[i][0][0][1]

				permute_code=np.arange(0, dim)
				for i in range(dim):
								permute_code[i]=adj_matrix_T[i][0][1]
				adj_matrix_R = np.diag(size_data)
				temp=np.arange(dim-1)
				temp=temp[::-1]
				temp[0]=dim-3
				connection_index = []
				connection_index.append(0)
				for i in range(dim):
								if i==1:
										connection_index.append(connection_index[-1]+1)
								else:
										if i==0:
												connection_index.append(connection_index[-1]+(temp[i]+1))
										else:
												connection_index.append(connection_index[-1]+(temp[i-1]+1))
				connection_index=connection_index[0:dim]
				connection =rank.tolist()
				index_tuple=np.triu_indices(dim, 1)
				index_tuple1=index_tuple[0]
				index_tuple2=index_tuple[1]
				index_tuple1=index_tuple1[connection_index] 
				index_tuple2=index_tuple2[connection_index]
				index_tuple=[index_tuple1,index_tuple2]
				index_tuple=tuple(index_tuple)
				adj_matrix_R[index_tuple] = connection
				adj_matrix_R[np.tril_indices(dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(dim, -1)]  
				index=np.diag_indices_from(adj_matrix_R)
				adj_matrix_R[index]=0

				permute=permute_code

				permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
				for i in range(dim): 
										permutation_matrix[permute[i],i] = 1
				adj_matrix_in=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())          
				adj_matrix_in[adj_matrix_in==1] = 0
				logging.info(adj_matrix_in)
				logging.info('Parameters:{} '.format(indv.actual_elements))   

if __name__ == '__main__':

	pipeline = Overlord(		# GENERATION PROPERTIES  
													max_generation=1000, generation=Generation, random_init=True, 
													# ISLAND PROPERTIES  
													N_islands=1,                  
													# INVIDUAL PROPERTIES        
													size=6, rank=8, out=np.array([3,3,3,3,3,3],dtype=int),Local_Opt_Iter=1,
													Local_Step_init=int(sys.argv[2]),Local_Step_main=int(sys.argv[3]),Rse_interpolation_on=int(sys.argv[4]),
													Rse_interpolation_Times=int(sys.argv[5]),
													####Local_Opt_Iter->numbers of the round trip in the local optimization.
													####Local_Step_init->the rank-related radius in the initial phase.
													####Local_Step_main->the rank-related radius in the main phase.
													####Rse_interpolation_on->the switch that decides whether or not to include the initial phase.
													####Rse_interpolation_Times->L_{0} of the paper.
													rse_therhold=[1e-8],Adam_Step=[0.001],
													# EVALUATION PROPERTIES 
													evaluate_repeat=2, max_iterations=10000,    
													fitness_func=[ lambda s, l: s+200*l], 

													# FOR COMP1.5
													callbacks=[score_summary])
	pipeline()
