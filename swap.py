import numpy as np
import os, sys, warnings
from itertools import compress
import pandas as pd
import argparse
import matplotlib
matplotlib.use('TKAgg')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 30})

import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import pickle as pkl
from math import sqrt
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rc('font', family='serif')

thresh = 0.01
verbose = False
use_thresh = True

class arm(object):
	""" An arm of a multi-armed bandit problem. Arms have the following properties:

	Attributes:
		true_utility: A number for the true utility of the arm
		arm_pull_strong_strategy: A function called when strong pulling an arm
		arm_pull_weak_strategy: A function called when weak pulling an arm
		args: Any arguments needed for the armpull strategy
		group: The group the arm belongs to
	"""
	def __init__(self, true_utility,arm_pull_strong_strategy,arm_pull_weak_strategy,weak_args=[],strong_args=[],group=None):
		self.true_utility = true_utility
		self.arm_pull_strong_strategy = arm_pull_strong_strategy
		self.arm_pull_weak_strategy = arm_pull_weak_strategy
		self.weak_args = weak_args
		self.strong_args = strong_args
		self.group = group

	def get_true_utility(self):
		return self.true_utility

	def weak_pull_arm(self):
		return self.arm_pull_weak_strategy(*self.weak_args)

	def strong_pull_arm(self):
		return self.arm_pull_strong_strategy(*self.strong_args)

	def get_group(self):
		return self.group

# Finds the top k
def basic_oracle(u_e,k):
	u_sorted = np.argsort(-1*u_e)
	return u_sorted[0:k]

# Addition of all in the set
def basic_utility(u_e,indices,k):
	return sum(u_e[indices])

# Greedy algorithm that maximizes the submodular function (in submodular_max_utility)
# (1-1/e) Optimal
def submodular_max_oracle(u_e,k,arms,groups,arms_groups):
	indices = {}
	look_set = {}
	for i in range(u_e.shape[0]):
		look_set[i] = True
	mx = 0
	mx_ind = -1
	# print k
	while len(indices) < k: #Want to make sure we have things to look at
		for i in look_set:
			# if i not in indices:
			val = submodular_max_utility(u_e,np.array(indices.keys() + [i]),k,arms,groups,arms_groups)
			if val > mx:
				mx = val
				mx_ind = i
		if mx_ind > -1:
			indices[mx_ind] = True
			look_set.pop(mx_ind)
			mx_ind = -1
		else:
			break
		# print k
		# print int(len(indices)) -k
		# print int(len(indices)) < k
	return indices.keys()

def other_submod_oracle(util,k,arms,groups,arms_groups):
    # indices_set = set(A_i)
    A = np.array(range(len(arms)))
    # print(arms_groups[A] == 1)
    indices_set = set()
    look_set = set()
    gp_set = {}
    arms_groups = np.array(arms_groups)
    gp_i = {}
    gp_sum = {}
    util = np.array(util)
    for gp in groups:

        tmp = np.array(list(set(A[arms_groups[A] == gp])))
        # print(tmp)
        if tmp.shape[0] > 0:
            gp_set[gp] = tmp[np.argsort(util[tmp])]
        else:
            gp_set[gp] = np.array([])
        
        gp_sum[gp] = 0
        gp_i[gp] = -1
    # print (gp_set)

    while len(indices_set) < k:
        mx = 0
        mx_gp = -1
        for gp in groups:
            if np.abs(gp_i[gp]) <= gp_set[gp].shape[0]:
                gp_sum[gp] += util[gp_set[gp][gp_i[gp]]]
                # print(util)
                u = 0
                for gp1 in groups:
                    u += np.sqrt(gp_sum[gp1])
                # print(u)
                if  mx <= u:
                    mx = u
                    mx_gp = gp
                gp_sum[gp] -= util[gp_set[gp][gp_i[gp]]]
        # print(mx)
        if mx_gp == -1:
            break
        else:
            indices_set.add(gp_set[mx_gp][gp_i[mx_gp]])
            gp_sum[mx_gp] += util[gp_set[mx_gp][gp_i[mx_gp]]]
            gp_i[mx_gp] -= 1
    return np.array(list(indices_set))


def submodular_max_utility(u_e,indices,k,arms,groups,arms_groups):
	add = np.zeros(groups.shape)
	for i in groups:
		ind = arms_groups[indices] == i
		if ind.shape[0] > 0:
			add[i] = np.sum(u_e[indices[ind]])
			if add[i] < 0:
				add[i] = 0

	add = np.sqrt(add)
	return np.sum(add)

def rad(T,n,t,delta,sigma):
	return sigma*np.sqrt(2*np.log(4*n*(t**3)/delta)/T)


def phi(p,a,u,Rad):
	if (u[p]+Rad[p])>(u[a]-Rad[a]) and (u[a]+Rad[a])>(u[p]-Rad[p]):
		return 1
	else:
		return 0

def strong_pull_decision(indices,k,s,j):
	alpha = (1.0*s/(1.0*s+j))*(1.0 - len(indices)/(2.0*k))
	return alpha

def strong_without_difference(indices,k,s,j):
	alpha = (1.0*s/(1.0*s+j))
	return alpha

def better_sj_frac(indices,k,s,j):
	if s == 1:
		return 0
	alpha = (1.0*s-j)/(s-1.0)
	return alpha

def weak_only(indices,k,s,j):
	return 0

def strong_only(indices,k,s,j):
	return 1

def get_optimal(arms,oracle,oracle_args):
	u = np.zeros(arms.shape)
	for i in range(arms.shape[0]):
		u[i] = arms[i].get_true_utility()
	return oracle(u,*oracle_args)

#No longer in use - use the animate function instead
def plotgraph(u,rad,M,sym,groups):
	color = np.array(['black' for _ in range(arms.shape[0])])
	color[M] = 'blue'
	color[sym] = 'red'
	pdt = pd.DataFrame({'i': range(arms.shape[0]), 'utility' : u, 'rad' : rad, 'groups' : groups,'color' : color})
	pdt.sort_values(['groups','utility'],inplace=True,ascending=False)
	a,b,c = plt.errorbar(range(arms.shape[0]),pdt.utility,yerr=pdt.rad[pdt.i],fmt='o')
	c[0].set_color(pdt.color)
	plt.xlim([-1,arms.shape[0]])
	plt.show()

def animate(i):
	color = np.array(['black' for _ in range(arms.shape[0])])
	color[M_groups[i]] = 'blue'
	if i < len(M_groups)-1:
		color[sym_t[i]] = 'red'
	s_color = np.array(['yellow' for _ in range(arms.shape[0])])
	s_color[M_star] = 'green'
	pdt = pd.DataFrame({'i': range(arms.shape[0]), 'utility' : u_t[i], 'rad' : rad_t[i], 'groups' : arms_groups,'color' : color,'s_color' : s_color,'true_utility' : x})
	pdt.sort_values(['groups','true_utility'],inplace=True,ascending=False)
	ax1.clear()
	#Make the colors of the dots different
	ax1.scatter(range(arms.shape[0]),pdt.utility,c=pdt.s_color[pdt.i],s=50,zorder=100)
	ind = np.array(range(arms.shape[0]))
	#Need to do each of the colors separately since matplotlib is stupid!
	if len(list(compress(ind,pdt.color == "blue"))) > 0:
		a,b,c = ax1.errorbar(list(compress(ind,pdt.color == "blue")),list(pdt.utility[pdt.color == "blue"]),yerr=list(pdt.rad[pdt.color == "blue"]),fmt='o',zorder=0,color="blue",lw=2, capsize=5, capthick=2)
	if len(list(compress(ind,pdt.color == "red"))) > 0:
		a,b,c = ax1.errorbar(list(compress(ind,pdt.color == "red")),list(pdt.utility[pdt.color == "red"]),yerr=list(pdt.rad[pdt.color == "red"]),fmt='o',zorder=0,color="red",lw=2, capsize=5, capthick=2)
	if len(list(compress(ind,pdt.color == "black"))) > 0:
		a,b,c = ax1.errorbar(list(compress(ind,pdt.color == "black")),list(pdt.utility[pdt.color == "black"]),yerr=list(pdt.rad[pdt.color == "black"]),fmt='o',zorder=0,color="black",lw=2, capsize=5, capthick=2)
	# a,b,c = ax1.errorbar(range(arms.shape[0]),pdt.utility,yerr=pdt.rad[pdt.i],fmt='o',zorder=0)
	# c[0].set_color(pdt.color[pdt.i])
	ax1.set_xlim([-1,arms.shape[0]])
	title = "Simulation of SWAP n=" + str(n) + " k=" + str(k) + " (itteration " + str(i)
	if i == len(strong_weak):
		title += ")"
	else:
		title += ",   weak pull)" if strong_weak[i] == 0 else ", strong pull)"
	ax1.set_title(title)


def swap(arms,oracle,oracle_utility,oracle_args,delta,s,j,k,decide_strong_pull=strong_pull_decision,sigma=0.5,chart=False,stopping=10000):
	start = time.time()
	n = arms.shape[0]
	#Pull each arm once
	u_e = np.zeros(arms.shape)
	for i in range(n):
		u_e[i] = arms[i].weak_pull_arm()
	#Update reward bound and cost
	T = np.ones(arms.shape)
	Rad = np.zeros(arms.shape)
	u_tilde = np.zeros(arms.shape)
	cost = [n]
	# print(cost)
	#Keep track of closeness, what groups were chosen, strong vs weak, and probabilities of strong pull
	looked_at = np.ones(arms.shape)
	closeness = []
	M_groups = []
	strong_weak = []
	pull_p = []
	end = time.time()
	u_t = None
	rad_t = None
	sym_t = None
	if chart:
		u_t = []
		rad_t = []
		sym_t = []
	if verbose:
		print "\tSetup time: " + str(end-start)
	# Have a rediculous cap just incase something bad happens
	print(stopping)
	for i in range(stopping):
		itter_start = time.time()
		if verbose:
			print "\nItteration " + str(i)
		#First run of oracle
		start = time.time()
		M = oracle(u_e,*oracle_args)
		# print M
		M_groups.append(M)
		end = time.time()
		if verbose:
			print "\tFirst oracle time: " + str(end-start)
		#Update extreme utility with bounds and rerun oracle
		start = time.time()
		for idx in range(n):
			Rad[idx] = rad(T[idx],n,cost[-1],delta,sigma)
			if idx in M:
				u_tilde[idx] = u_e[idx]-Rad[idx]
			else:
				u_tilde[idx] = u_e[idx]+Rad[idx]
		end = time.time()
		if verbose:
			print "\tnew utility: " + str(end-start)
		start = time.time()
		#Second run of oracle
		M_tilde = oracle(u_tilde,*oracle_args)
		#Find the symmetric difference between the two sets
		indices = list((set(M)-set(M_tilde))|(set(M_tilde)-set(M)))
		#If we are charting at the end we need to keep track of everything
		if chart:
			u_t.append(np.copy(u_e))
			rad_t.append(np.copy(Rad))
			sym_t.append(np.array(indices))
		end = time.time()
		if verbose:
			print "\tNew oracle and symmetric difference time: " + str(end-start)
		#Using extreme utility find what the two sets are worth
		start = time.time()
		M_util = oracle_utility(u_tilde,np.array(M),*oracle_args)
		M_tilde_util = oracle_utility(u_tilde,np.array(M_tilde),*oracle_args)
		closeness.append(abs(M_util-M_tilde_util))
		# Once the difference between the two groups is very very small, end
		# Unless we are using no symmetric difference then end when the two sets are equal.
		if i == 0:
			thr = thresh*abs(M_util-M_tilde_util)
		if (use_thresh and abs(M_util-M_tilde_util) < thresh) or (not use_thresh and len(indices) == 0) :
			return (M,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t)
		end = time.time()
		if verbose:
			print "\tStopping condition time: " + str(end-start)

		#Find the most uncertain arm in the symmetic differencde
		start = time.time()
		p = indices[0]
		for idx in indices:
			if Rad[idx] > Rad[p]:
				p = idx
		looked_at[p] += 1

		#Decide on strong vs weak pull and pull arm
		strong_pull_p = decide_strong_pull(indices,k,s,j)
		pull_p.append(strong_pull_p)
		strong_pull = strong_pull_p > np.random.uniform()
		if strong_pull:
			if verbose:
				print "Strong pulling arm " + str(p)
			u_e[p] = (u_e[p]*T[p]+arms[p].strong_pull_arm()*s)/(T[p]+s)
			T[p] += s
			cost.append(cost[-1]+j)
			strong_weak.append(1)
		else:
			if verbose:
				print "Weak pulling arm " + str(p)
			u_e[p] = (u_e[p]*T[p]+arms[p].weak_pull_arm())/(T[p]+1)
			T[p] += 1
			cost.append(cost[-1]+1)
			strong_weak.append(0)
		end = time.time()
		if verbose:
			print "\tStrong/weak pull time: " + str(end-start)
			print "Total time: " + str(end-itter_start)
	#Hit too many reviews - time out (This is what I mean by something bad happening)
	print("\tSomething bad happened")
	return (M,cost[:-1],closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t)

def calculate_C(util,arms,R,groups,arms_groups,k):
	opt = submodular_max_oracle(util,k,arms,groups,arms_groups)
	opt = np.array(opt)
	opt_util = submodular_max_utility(util,opt,k,arms,groups,arms_groups)
	# print opt_util
	H = 0
	for i in range(util.shape[0]):
		# if i in opt:
		util_t = np.delete(util,i)
		arms_groups_t = np.delete(arms_groups,i)
		if i in opt:
			opt_t = np.array(submodular_max_oracle(util_t,k,arms,groups,arms_groups_t))
			# print opt_util - submodular_max_utility(util_t,opt_t,k,arms,groups,arms_groups_t)
		else:
			opt_t = np.array(submodular_max_oracle(util_t,k-1,arms,groups,arms_groups_t))
			opt_t = np.append(opt_t,i)
			gap = opt_util - submodular_max_utility(util,opt_t,k,arms,groups,arms_groups)
		H += 1.0/(gap*gap)

	return 4*(util.shape[0])*H/delta

	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Runs the SWAP algorithm')
	parser.add_argument('pull_strat', help="What pulling strategy (weak,strong,sj_frac,sj_sym_dif,better_sj_frac)")
	parser.add_argument('-n', '--n', help="The number of total arms")
	parser.add_argument('-k', '--k', help="The number of arms to select")
	parser.add_argument('-a', '--a', help="The file to save the animation")
	parser.add_argument('-f', '--folder', help="Save folder")
	parser.add_argument('-i', '--num_iter', help="Number of iterations for vary sj")
	parser.add_argument('-r', '--run', help="Use to parallelize")
	parser.add_argument('--single_test', dest="single_test", action="store_true",help="Test on normally distributed arms")
	parser.add_argument('--vary_sj', dest="vary_sj", action="store_true",help="Run test on normally distributed arms varying s and j")
	parser.add_argument('--vary_sj_big', dest="vary_sj_big", action="store_true",help="Run test on normally distributed arms varying s and j")
	parser.add_argument('--animate', dest="animate", action="store_true",help="Create an animation of swap running")
	parser.add_argument('--compare', dest="compare", action="store_true",help="Compare sets")
	parser.add_argument('--compare_big', dest="compare_big", action="store_true",help="Compare sets")
	parser.add_argument('--mean', dest="mean", action="store_true",help="Use mean instead of median")
	parser.add_argument('--use_thresh', dest="use_thresh", action="store_true", help="Use closeness threshold instead of zero symmetric difference")
	parser.add_argument('--show_plot', dest="show_plot", action="store_true",help="Show the plots while running (It will still save)")
	parser.add_argument('--combine', dest='combine', action='store_true', help='Combine files')
	parser.add_argument('--numerical_test', dest='numerical_test', action='store_true', help='Numerical test')
	parser.add_argument('--numerical_test_submod', dest='numerical_test_submod', action='store_true', help='Numerical test submodular')
	parser.add_argument('--numerical_test_submod_threshold', dest='numerical_test_submod_threshold', action='store_true', help='Numerical test submodular')
	# parser.add_argument('--no_symdif', dest="no_symdif", action="store_true",help="Use the s/(s+j) pulling strategy without symmetric difference")



	args = parser.parse_args()

	#Setting the seed
	np.random.seed(2345)

	#Are we using threshold or zero symmetric difference?
	if args.use_thresh:
		use_thresh = True

	#Set up the problem
	n = int(args.n) if args.n else 50
	k = int(args.k) if args.k else 7
	groups = np.array([0,1,2])
	sigma = 5
	delta = 0.5
	s = 10
	j = 5
	x = np.random.normal(loc=200,scale=50,size=n)
	arms_groups = np.random.randint(0,groups.shape[0],[n])
	arms = np.array([arm(x[i],np.random.normal,np.random.normal,[x[i],sigma],[x[i],1.0*sigma/10],arms_groups[i]) for i in range(n)])
	oracle = submodular_max_oracle
	utility = submodular_max_utility
	oracle_args = [k,arms,groups,arms_groups]
	# pull_strat = strong_without_difference if args.no_symdif else strong_pull_decision
	if args.pull_strat == "sj_frac":
		pull_strat = strong_without_difference
	elif args.pull_strat == "sj_sym_dif":
		pull_strat = strong_pull_decision
	elif args.pull_strat == "better_sj_frac":
		pull_strat = better_sj_frac
	elif args.pull_strat == "weak":
		pull_strat = weak_only
	elif args.pull_strat == "strong":
		pull_strat = strong_only
	else:
		sys.exit("Not a valid pulling strategy")

	#Save folder
	folder = args.folder if args.folder else "images/"
	folder = folder if folder[-1] == "/" else folder + "/"
	folder += "thresh/" if args.use_thresh else "zero_sym_dif/"
	folder += args.pull_strat + "/"
	if args.compare_big:
		folder = "big_images/" + folder
		folder += "mean/" if args.mean else "median/"
	if (not os.path.exists(folder)):
		os.makedirs(folder)


	run = int(args.run) if args.run else 0
	for i in range(run):
		np.random.random()

	#Setting the seed again
	np.random.seed(np.random.randint(0,2**32-1))

	if args.numerical_test:
		data = pd.read_csv('num_analysis.csv')
		data['actual_cost'] = -1
		oracle = basic_oracle
		utility = basic_utility
		for index,row in data.iterrows():
			print "Running {} for hardness {} and upper {}".format(index, row['H'], row['T']) 
			np.random.seed((int)(row['seed']))
			oracle_args = [(int)(row['K'])]
			x = np.random.normal(10.0,1,size=((int)(row['n'])))
			arms = np.array([arm(x[i],np.random.normal,np.random.normal,[x[i],row['sigma']/np.sqrt(row['s'])],[x[i],row['sigma']],1) for i in range((int)(row['n']))])
			(_,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t) = swap(arms,oracle,utility,oracle_args,row['delta'],row['s'],row['j'],(int)(row['K']),sigma=row['sigma'],decide_strong_pull=pull_strat, stopping=((int) (row['T']/3)))
			data.actual_cost[index] = cost[-1]
			print "Finished at {} cost".format(cost[-1])
			# print(data)
		data.to_csv('num_analysis_with_real.csv')

	if args.numerical_test_submod:
		data = pd.read_csv('num_analysis.csv')
		np.random.seed(2345)
		arms_groups = np.random.randint(0,groups.shape[0],[(int)(data.n[0])])
		data['actual_cost'] = -1
		oracle = other_submod_oracle
		utility = submodular_max_utility
		for index,row in data.iterrows():
			print "Running {} for hardness {} and upper {}".format(index, row['H'], row['T']) 
			np.random.seed((int)(row['seed']))
			x = np.random.normal(10.0,1,size=((int)(row['n'])))
			arms = np.array([arm(x[i],np.random.normal,np.random.normal,[x[i],row['sigma']/np.sqrt(row['s'])],[x[i],row['sigma']],arms_groups[i]) for i in range((int)(row['n']))])
			oracle_args = [(int)(row['K']),arms,groups,arms_groups]
			(_,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t) = swap(arms,oracle,utility,oracle_args,row['delta'],row['s'],row['j'],(int)(row['K']),sigma=row['sigma'],decide_strong_pull=pull_strat, stopping=((int)(27454812/3)))
			data.actual_cost[index] = cost[-1]
			print "Finished at {} cost".format(cost[-1])
			# print(data)
			# break
		data.to_csv('num_analysis_submod_with_real.csv')
	if args.numerical_test_submod_threshold:
		use_thresh = True
		data = pd.read_csv('num_analysis.csv')
		np.random.seed(2345)
		arms_groups = np.random.randint(0,groups.shape[0],[(int)(data.n[0])])
		data['actual_cost'] = -1
		oracle = other_submod_oracle
		utility = submodular_max_utility
		for index,row in data.iterrows():
			print "Running {} for hardness {} and upper {}".format(index, row['H'], row['T']) 
			np.random.seed((int)(row['seed']))
			x = np.random.normal(10.0,1,size=((int)(row['n'])))
			arms = np.array([arm(x[i],np.random.normal,np.random.normal,[x[i],row['sigma']/np.sqrt(row['s'])],[x[i],row['sigma']],arms_groups[i]) for i in range((int)(row['n']))])
			oracle_args = [(int)(row['K']),arms,groups,arms_groups]
			(_,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t) = swap(arms,oracle,utility,oracle_args,row['delta'],row['s'],row['j'],(int)(row['K']),sigma=row['sigma'],decide_strong_pull=pull_strat, stopping=((int)(27454812/3)))
			data.actual_cost[index] = cost[-1]
			print "Finished at {} cost".format(cost[-1])
			# print(data)
			# break
		data.to_csv('num_analysis_submod_pac.csv')
	# ************************************************************************************
	# Single Test
	# ************************************************************************************
	if args.single_test:
		verbose = True

		#Run SWAP
		(_,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t) = swap(arms,oracle,utility,oracle_args,delta,s,j,k,sigma=sigma,decide_strong_pull=pull_strat)

		#Make sure we have the right directories
		if (not os.path.exists(folder + str(n) + "_" + str(k) + "/")):
			os.makedirs(folder + str(n) + "_" + str(k) + "/")

		#Plot closeness over time (of empiricle vs worst case)
		plt.plot(cost,closeness)
		plt.title("Closeness of empiricle vs worst case over time")
		plt.savefig(folder + str(n) + "_" + str(k) + "/closeness.png")
		plt.savefig(folder + str(n) + "_" + str(k) + "/closeness.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		#Plot sets over time
		M_star = set(get_optimal(arms,oracle,oracle_args))
		convergence = []
		for M in M_groups:
			convergence.append(1-len(M_star-set(M))*1.0/len(M_star))
		plt.plot(cost,convergence)
		plt.axis([0, cost[-1],0,1.1])
		plt.title("Convergence of optimal vs empiricle over time")
		plt.savefig(folder + str(n) + "_" + str(k) + "/convergence.png")
		plt.savefig(folder + str(n) + "_" + str(k) + "/convergence.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		#Plot closeness and convergence over time
		fig, ax1 = plt.subplots()
		ax1.plot(cost, closeness, 'b-')
		ax1.set_xlabel('Cost')
		ax1.set_ylabel('Closeness', color='b')
		ax1.tick_params('y', colors='b')
		plt.title("Closeness and convergence over time")
		ax2 = ax1.twinx()
		ax2.plot(cost, convergence, 'r-')
		ax2.set_ylabel('convergence', color='r')
		ax2.tick_params('y', colors='r')
		ax2.set_ylim([0,1.1])
		plt.savefig(folder + str(n) + "_" + str(k) + "/closeness_convergence.png")
		plt.savefig(folder + str(n) + "_" + str(k) + "/closeness_convergence.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()


		#Plot strong vs weak over time
		plt.plot(strong_weak,"ro")
		plt.title("Strong or weak arm pulls over time")
		plt.savefig(folder + str(n) + "_" + str(k) + "/strong_weak.png")
		plt.savefig(folder + str(n) + "_" + str(k) + "/strong_weak.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		#Plot pulling probabilities over time
		plt.plot(pull_p,"ro")
		plt.title("Probability of strong arm pulls over time")
		plt.axis([0,len(pull_p),0,1.1])
		plt.savefig(folder + str(n) + "_" + str(k) + "/strong_prob.png")
		plt.savefig(folder + str(n) + "_" + str(k) + "/strong_prob.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		#Plot pulls of arms
		u = np.array([a.true_utility for a in arms])
		ind = np.argsort(u)
		plt.plot(looked_at[ind])
		plt.title("How many times each arm was looked at")
		plt.axis([0,looked_at.shape[0],0,np.max(looked_at)+1])
		plt.savefig(folder + str(n) + "_" + str(k) + "/looked_at.png")
		plt.savefig(folder + str(n) + "_" + str(k) + "/looked_at.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

	# ************************************************************************************
	# Animate
	# ************************************************************************************
	elif args.animate:

		#Run SWAP
		print "Running SWAP..."
		(_,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t) = swap(arms,oracle,utility,oracle_args,delta,s,j,k,sigma=sigma,chart=True,decide_strong_pull=pull_strat)

		#Animate
		print "Creating animation..."
		M_star = np.array(get_optimal(arms,oracle,oracle_args))
		fig = plt.figure()
		ax1 = fig.add_subplot(1,1,1)
		ani = animation.FuncAnimation(fig, animate, interval=250,	save_count=len(u_t))
		ani.save(folder + args.a if args.a else folder + 'swap_animation.mp4', fps=30)
		if (args.show_plot):
			plt.show()
		else:
			plt.close()



	elif args.vary_sj_big:
		num_iter = int(args.num_iter) if args.num_iter else 20
		sj_cost = np.zeros([20,20,num_iter])
		for s in range(1,20):
			for j in range(1,20):
				if s >= j:
					arms = np.array([arm(x[i],np.random.normal,np.random.normal,[x[i],sigma],[x[i],1.0*sigma/sqrt(s)],arms_groups[i]) for i in range(n)])
					print "Running s=" + str(s) + " and j=" + str(j)
					for i in range(num_iter):
						print "\t Itteration=" + str(i)
						(_,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t) = swap(arms,oracle,utility,oracle_args,delta,s,j,k,sigma=sigma,decide_strong_pull=pull_strat)
						sj_cost[j][s][i] = cost[-1]
					if args.pull_strat == "weak":
						break
			if args.pull_strat == "weak":
				break
		folder = args.folder if args.folder else "data/" #Redoing this since we are saving data not images
		folder = folder if folder[-1] == "/" else folder + "/"
		folder += "thresh/" if args.use_thresh else "zero_sym_dif/"
		folder += args.pull_strat + "/"
		if (not os.path.exists(folder)):
			os.makedirs(folder)
		with open(folder + "sj_cost" + str(run) + ".pkl","w") as f:
			pkl.dump(sj_cost,f)

	# ************************************************************************************
	# Vary SJ
	# ************************************************************************************
	elif args.vary_sj:
		sj_cost = np.zeros([20,20])
		itterations = np.zeros([20,20])
		strong = np.zeros([20,20])
		M_star = set(get_optimal(arms,oracle,oracle_args))
		for s in range(1,20): #Go over 20 different s values
			for j in range(1,20): #Go over 20 different j values
				if s >= j: #Only look at cases when s is greater than j
					print "Running s=" + str(s) + " and j=" + str(j)
					for i in range(20): #Run 20 times to smooth out
						print "\t Itteration=" + str(i)
						(_,cost,closeness,M_groups,strong_weak,pull_p,T,looked_at,u_t,rad_t,sym_t) = swap(arms,oracle,utility,oracle_args,delta,s,j,k,sigma=sigma,decide_strong_pull=pull_strat)
						sj_cost[j][s] += cost[-1]
						itterations[j][s] += len(cost)
						strong[j][s] += 1.0*sum(strong_weak)/len(strong_weak)
					#Make sure we average over the 20 runs
					sj_cost[j][s] = sj_cost[j][s]/20
					itterations[j][s] = itterations[j][s]/20
					strong[j][s] = strong[j][s]/20


		#Save original pictures because why not right?
		plt.clf()
		A = sj_cost[1:,1:]
		mask =  np.tri(A.shape[0], k=-1)
		A = np.ma.array(A, mask=mask)
		plt.imshow(A,interpolation="nearest",cmap="OrRd",extent=[1,19,19,1])
		plt.xlabel("s values")
		plt.ylabel("j values")
		plt.title("Total cost when varying s and j")
		plt.colorbar()
		plt.savefig(folder + "cost.png")
		plt.savefig(folder + "cost.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		plt.clf()
		A = itterations[1:,1:]
		mask =  np.tri(A.shape[0], k=-1)
		A = np.ma.array(A, mask=mask) # mask out the lower triangle
		plt.imshow(A,interpolation="nearest",cmap="OrRd",extent=[1,19,19,1])
		plt.xlabel("s values")
		plt.ylabel("j values")
		plt.title("Total itterations when varying s and j")
		plt.colorbar()
		plt.savefig(folder + "itterations.png")
		plt.savefig(folder + "itterations.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		plt.clf()
		A = strong[1:,1:]
		mask =  np.tri(A.shape[0], k=-1)
		A = np.ma.array(A, mask=mask)
		plt.imshow(A,interpolation="nearest",cmap="OrRd",extent=[1,19,19,1])
		plt.xlabel("s values")
		plt.ylabel("j values")
		plt.title("Percentage of strong pulls")
		plt.colorbar()
		plt.savefig(folder + "strong_pulls_percent.png")
		plt.savefig(folder + "strong_pulls_percent.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		#Save data so John will stop yelling at me :P
		folder = args.folder if args.folder else "data/" #Redoing this since we are saving data not images
		folder = folder if folder[-1] == "/" else folder + "/"
		folder += "thresh/" if args.use_thresh else "zero_sym_dif/"
		folder += args.pull_strat + "/"
		if (not os.path.exists(folder)):
			os.makedirs(folder)
		with open(folder + "sj_cost.pkl","w") as f:
			pkl.dump(sj_cost,f)
		with open(folder + "itterations.pkl","w") as f:
			pkl.dump(itterations,f)
		with open(folder + "strong.pkl","w") as f:
			pkl.dump(strong,f)

	# ************************************************************************************
	# Vary SJ compare - run this after running vary_sj
	# ************************************************************************************
	elif args.compare:
		#Directories
		image_dir = folder
		folder = args.folder if args.folder else "data/" #Getting data saved previously
		folder = folder if folder[-1] == "/" else folder + "/"
		folder += "thresh/" if args.use_thresh else "zero_sym_dif/"
		weak_folder = folder + "weak/"
		strong_folder = folder + "strong/"
		folder += args.pull_strat + "/"

		#First look at cost
		try:
			with open(folder + "sj_cost.pkl") as f:
				swap_cost = pkl.load(f)
			with open(weak_folder + "sj_cost.pkl") as f:
				weak_cost = pkl.load(f)
			with open(strong_folder + "sj_cost.pkl") as f:
				strong_cost = pkl.load(f)

			weak_cost_dif = weak_cost - swap_cost

			plt.clf()
			A = weak_cost_dif[1:,1:]
			mask =  np.tri(A.shape[0], k=-1)
			A = np.ma.array(A, mask=mask) # mask out the lower triangle
			mx = np.max(np.abs(A))
			plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
			plt.xlabel("s values")
			plt.ylabel("j values")
			plt.title("Difference of cost values of SWAP and Weak Pull")
			plt.colorbar()
			plt.savefig(image_dir + "weak_cost_dif.png")
			plt.savefig(image_dir + "weak_cost_dif.pdf")
			if (args.show_plot):
				plt.show()
			else:
				plt.close()

			strong_cost_dif = strong_cost - swap_cost

			plt.clf()
			A = strong_cost_dif[1:,1:]
			mask =  np.tri(A.shape[0], k=-1)
			A = np.ma.array(A, mask=mask) # mask out the lower triangle
			mx = np.max(np.abs(A))
			plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
			plt.xlabel("s values")
			plt.ylabel("j values")
			plt.title("Difference of cost values of SWAP and Strong Pull")
			plt.colorbar()
			plt.savefig(image_dir + "strong_cost_dif.png")
			plt.savefig(image_dir + "strong_cost_dif.pdf")
			if (args.show_plot):
				plt.show()
			else:
				plt.close()
		except:
			warnings.warn("Something wrong with cost pickle files",UserWarning)

		#Cost optimal zone
		# try:
		with open(folder + "sj_cost.pkl") as f:
			swap_cost = pkl.load(f)
		with open(weak_folder + "sj_cost.pkl") as f:
			weak_cost = pkl.load(f)
		with open(strong_folder + "sj_cost.pkl") as f:
			strong_cost = pkl.load(f)

		weak_cost_dif = weak_cost - swap_cost
		strong_cost_dif = strong_cost - swap_cost
		dif = (weak_cost_dif >= 0) & (strong_cost_dif >= 0)
		print dif + 1


		plt.clf()
		A = dif[1:,1:]
		mask =  np.tri(A.shape[0], k=-1)
		A = np.ma.array(A, mask=mask) # mask out the lower triangle
		mx = np.max(np.abs(A))
		plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[0,1])
		plt.xlabel("s values")
		plt.ylabel("j values")
		plt.title("Optimal zone")
		plt.colorbar()
		plt.savefig(image_dir + "opt.png")
		plt.savefig(image_dir + "opt.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()
		# except:
		# 	warnings.warn("Something wrong with cost pickle files",UserWarning)



		#Second look at itterations (if you have run it Candice :| )
		try:
			with open(folder + "itterations.pkl") as f:
				swap_iter = pkl.load(f)
			with open(weak_folder + "itterations.pkl") as f:
				weak_iter = pkl.load(f)
			with open(strong_folder + "itterations.pkl") as f:
				strong_iter = pkl.load(f)

			weak_iter_dif = weak_iter -  swap_iter

			plt.clf()
			A = weak_iter_dif[1:,1:]
			mask =  np.tri(A.shape[0], k=-1)
			A = np.ma.array(A, mask=mask) # mask out the lower triangle
			mx = np.max(np.abs(A))
			plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
			plt.xlabel("s values")
			plt.ylabel("j values")
			plt.title("Difference of iterations values of SWAP and Weak Pull")
			plt.colorbar()
			plt.savefig(image_dir + "weak_iter_dif.png")
			plt.savefig(image_dir + "weak_iter_dif.pdf")
			if (args.show_plot):
				plt.show()
			else:
				plt.close()

			strong_iter_dif = strong_iter - swap_iter

			plt.clf()
			A = strong_iter_dif[1:,1:]
			mask =  np.tri(A.shape[0], k=-1)
			A = np.ma.array(A, mask=mask) # mask out the lower triangle
			mx = np.max(np.abs(A))
			plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
			plt.xlabel("s values")
			plt.ylabel("j values")
			plt.title("Difference of iterations values of SWAP and Strong Pull")
			plt.colorbar()
			plt.savefig(image_dir + "strong_iter_dif.png")
			plt.savefig(image_dir + "weak_iter_dif.pdf")
			if (args.show_plot):
				plt.show()
			else:
				plt.close()
		except:
			warnings.warn("Something wrong with iterations pickle files",UserWarning)

		#Third look at strong pulls (again...if you actually saved them...)
		try:
			with open(folder + "strong.pkl") as f:
				swap_strong = pkl.load(f)
			with open(weak_folder + "strong.pkl") as f:
				weak_strong = pkl.load(f)
			with open(strong_folder + "strong.pkl") as f:
				strong_strong = pkl.load(f)

			weak_strong_dif = weak_strong -  swap_strong

			plt.clf()
			A = weak_strong_dif[1:,1:]
			mask =  np.tri(A.shape[0], k=-1)
			A = np.ma.array(A, mask=mask) # mask out the lower triangle
			mx = np.max(np.abs(A))
			plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
			plt.xlabel("s values")
			plt.ylabel("j values")
			plt.title("Difference of strong pulls percent values of SWAP and Weak Pull")
			plt.colorbar()
			plt.savefig(image_dir + "weak_strong_dif.png")
			plt.savefig(image_dir + "weak_strong_dif.pdf")
			if (args.show_plot):
				plt.show()
			else:
				plt.close()

			strong_strong_dif = strong_strong - swap_strong

			plt.clf()
			A = strong_strong_dif[1:,1:]
			mask =  np.tri(A.shape[0], k=-1)
			A = np.ma.array(A, mask=mask) # mask out the lower triangle
			mx = np.max(np.abs(A))
			plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
			plt.xlabel("s values")
			plt.ylabel("j values")
			plt.title("Difference of strong pulls percent values of SWAP and Strong Pull")
			plt.colorbar()
			plt.savefig(image_dir + "strong_strong_dif.png")
			plt.savefig(image_dir + "weak_strong_dif.pdf")
			if (args.show_plot):
				plt.show()
			else:
				plt.close()
		except:
			warnings.warn("Something wrong with strong pull pickle files",UserWarning)

	elif args.compare_big:
		image_dir = folder
		print image_dir
		folder = args.folder if args.folder else "data_images/big_data/data/" #Getting data saved previously
		folder = folder if folder[-1] == "/" else folder + "/"
		folder += "thresh/" if args.use_thresh else "zero_sym_dif/"
		weak_folder = folder + "weak/"
		strong_folder = folder + "strong/"
		folder += args.pull_strat + "/"

		#First look at cost
		# try:
		with open(folder + "sj_cost.pkl") as f:
			swap_cost = pkl.load(f)
		with open(weak_folder + "sj_cost.pkl") as f:
			weak_cost = pkl.load(f)
			for i in range(1,weak_cost.shape[0]):
				for j in range(1,weak_cost.shape[0]):
					weak_cost[i][j] = weak_cost[1][1]
			# print weak_cost
		with open(strong_folder + "sj_cost.pkl") as f:
			strong_cost = pkl.load(f)

		num = swap_cost.shape[-1]

		weak_cost_dif = weak_cost-swap_cost
		strong_cost_dif = strong_cost - swap_cost
		dif = (weak_cost_dif >= 0) & (strong_cost_dif >= 0)
		dif = np.sum(dif,axis = 2)/(num*1.0)
		dif = dif >= 0.5
		for i in range(20):
			dif[1][i] = 0
			dif[i][i] = 0
		# print dif & 1


		if args.mean:
			swap_cost = np.mean(swap_cost,axis=2)
			weak_cost = np.mean(weak_cost,axis=2)
			strong_cost = np.mean(strong_cost,axis=2)
		else:
			swap_cost = np.median(swap_cost,axis=2)
			weak_cost = np.median(weak_cost,axis=2)
			strong_cost = np.median(strong_cost,axis=2)

		weak_cost_dif = weak_cost - swap_cost
		strong_cost_dif = strong_cost - swap_cost

		A_weak = weak_cost_dif[1:,1:]
		mask =  np.tri(A_weak.shape[0], k=-1)
		A_weak = np.ma.array(A_weak, mask=mask) # mask out the lower triangle
		mx_weak = np.max(np.abs(A_weak))

		A_strong = strong_cost_dif[1:,1:]
		mask =  np.tri(A_strong.shape[0], k=-1)
		A_strong = np.ma.array(A_strong, mask=mask) # mask out the lower triangle
		mx_strong = np.max(np.abs(A_strong))

		mx = np.max(np.array([mx_weak,mx_strong]))

		plt.figure(figsize=(8, 8))
		im = plt.imshow(A_weak,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
		plt.xlabel("$s$ values",fontsize="x-large")
		plt.ylabel("$j$ values",fontsize="x-large")
		if (args.pull_strat == "strong"):
			plt.title("Strong vs Weak Pull")
		else:
			plt.title("Difference of cost values of\nSWAP and Weak Pull")

		if (args.pull_strat == "strong"):
			print "here"
			C = 4
			C = calculate_C(x,arms,sigma,groups,arms_groups,k)
			s = [i for i in range(1,20)]
			j = [C**((i)/3.0-2.0/3.0) for i in range(1,20)]
			plt.plot(j,s)
			plt.axis((1,19,19,1))
		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", "5%", pad="3%")
		plt.colorbar(im, cax=cax)
		plt.tight_layout()
		plt.savefig(image_dir + "weak_cost_dif.png")
		plt.savefig(image_dir + "weak_cost_dif.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()


		plt.figure(figsize=(8, 8))
		im = plt.imshow(A_strong,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[-mx,mx])
		plt.xlabel("$s$ values",fontsize="x-large")
		plt.ylabel("$j$ values",fontsize="x-large")
		plt.title("Difference of cost values of\nSWAP and Strong Pull")
		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", "5%", pad="3%")
		plt.colorbar(im, cax=cax)
		plt.tight_layout()
		plt.savefig(image_dir + "strong_cost_dif.png")
		plt.savefig(image_dir + "strong_cost_dif.pdf")

		if (args.show_plot):
			plt.show()
		else:
			plt.close()

		# dif = (weak_cost_dif >= 0) & (strong_cost_dif >= 0)
		# # print dif + 1


		plt.clf()
		plt.figure(figsize=(8, 8))
		A = dif[1:,1:]
		mask =  np.tri(A.shape[0], k=-1)
		A = np.ma.array(A, mask=mask) # mask out the lower triangle
		mx = np.max(np.abs(A))
		im = plt.imshow(A,interpolation="nearest",cmap="PiYG",extent=[1,19,19,1],clim=[0,1])
		plt.xlabel("$s$ values",fontsize="x-large")
		plt.ylabel("$j$ values",fontsize="x-large")
		plt.title("Optimal zone of SWAP")
		
		plt.tight_layout()
		plt.savefig(image_dir + "opt.png")
		plt.savefig(image_dir + "opt.pdf")
		if (args.show_plot):
			plt.show()
		else:
			plt.close()
		# except:
		# 	warnings.warn("Something wrong with pull cost files")

	elif (args.combine):
		folder = args.folder if args.folder else "big_data/data/" #Getting data saved previously
		folder = folder if folder[-1] == "/" else folder + "/"
		folder += "thresh/" if args.use_thresh else "zero_sym_dif/"
		folder += args.pull_strat + "/"

		data = []
		i = 1
		while (os.path.exists(folder + "sj_cost" + str(i) + ".pkl")):
			with open(folder + "sj_cost" + str(i) + ".pkl") as f:
				data.append(pkl.load(f))
			# print data[-1].shape
			i += 1
		# print len(data)
		cost = data[0]
		for i in range(1,len(data)):
			cost = np.concatenate((cost,data[i]),axis=2)
			# print cost.shape
		with open(folder + "sj_cost.pkl","w") as f:
			pkl.dump(cost,f)


	else:
		# print arms
		print "Please give me something to do."
		print "Run python \"swap.py -h\" for help\n"
