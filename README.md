*Recurrence Analysis and Symbolic Complexity in Python (RASCoPy)* is an analysis
toolbox written by Gauthier Daubes and Axel Hut. An original implementation in 
Matlab has been developed by Peter beim Graben. 

The techniques implemented have been developed by Peter beim Graben and Axel Hutt
and published in 

- M. Fedotenkova, P. beim Graben, J. Sleigh, A. Hutt, 
Time-Frequency Representations as Phase Space Reconstruction in Symbolic Recurrence Structure Analysis,
In: I. Rojas, H. Pomares and O. Valenzuela (Eds.). Advances in Time Series Analysis and Forecasting, pp. 89-102 (2017) 

- P. beim Graben, K. K. Sellers, F. Fröhlich and A. Hutt, Optimal estimation of recurrence structures from time series,
Europhysics Letters 114(3): 38003 (2016)

- P. beim Graben and A. Hutt, Detecting event-related recurrences by symbolic analysis: Applications to human language processing, 
Philosphical Transactions of the Royal Society A 373:20140089 (2015)

- P. beim Graben and A. Hutt, Detecting metastable states of dynamical systems by recurrence-based symbolic dynamics,
Physical Review Letters 110: 154101 (2013) 




'Package description'

**************************************************************
**********************Module 1: recurrence********************
**************************************************************

1- rec_mat(y, epsilon):

* Inputs : numpy array (y), integer (epsilon)
* Outputs : numpy array (R)

	y is the coordinate array of each point of a trajectory (1D, 2D, or 3D), and epsilon is the criterion that determines the maximum admissible distance between two points of the trajectory to be considered as recurrent.

	The output is the recurrence matrix R, composed of 0s and 1s, representing the presence (1) or absence (0) of recurrence between each pair of points in the trajectory. It is computed by calculating the distance matrix of each pair of points of the trajectory y. 1 appears when the distance is smaller than epsilon and 0 when it bigger.

	Moreover this function permits to save the recurrence matrix in a file.txt.


2- rec_plt(R):

* Inputs : numpy array 2 dimensions (R)
* Outputs : recurrence plot

	This function plots the recurrence plot of the input recurrence matrix R and saves it as a file.png.

**************************************************************
*******************Module 2: symbolic_series******************
**************************************************************

1- symbolic_serie(R):

* Inputs : numpy array 2 dimensions (R)
* Outputs : numpy array 1 dimension (newSerie)

	This function computes the symbolic series of the metastable states of the trajectory thanks to its recurrence matrix. All recurrent points creates together a metastable state. The symbolic serie is a list of integer that describes through which states the trajectory goes. For example, if the serie is [1,1,1,0,0,2,2,2,2,0,0,1,1], then the trajectory has two metastable states (1 and 2) and goes from the state 1 to state 2 to state 1 again. 0 are called transcient, they appear when a point is not recurrent in the trajectory. They are passage states from a metastable state to another.
	
	Moreover the function give the possibility to the user to save the symbolic serie in a file.txt.


2- colored_sym_serie(serie,y):

* Inputs : numpy array 1 dimension (serie), numpy array (y)
* Outputs : figure of the colored symbolic serie

	This function plots the colored symbolic serie in a figure where each different state has a different color and transcients are represented in red. 

3- plot_col_traj(serie,y):

* Inputs : numpy array 1 dimension (serie), numpy array (y)
* Outputs : plot the trajectory with colored different states

	This function plots the trajectory (y) in colors with respect to the symbolic serie (serie) where each different state has a different color and transcients are represented in red. 


4- complexity(serie):

* Inputs : numpy array 1 dimension (serie)
* Outputs : 3 integer (C_alphabet_size, C_nbr_words, C_LZ)

	This function computes and returns the complexity of a symbolic serie (serie) with 3 different methods: the alphabet size, the number of words, and the Lempel_Ziv complexity. Moreover, the user can save these 3 integer into a file.txt.



**************************************************************
**********************Module 3: espi_opti*********************
**************************************************************

1- epsi_entropy(y, step):

* Inputs : numpy array (y), integer (step)
* Outputs : plot the entropy function of epsilon, 
			float	(EpsioptiH)

	This function iterates through various epsilon values within a range from 0 to the maximum distance between each pair of points in the trajectory, advancing from a user-defined step. It computes during each iteration the Shannon's entropy of the symbolic serie. Then, it returns and save in a file.txt the optimal epsilon corresponding to the maximal entropy. Moreover, it plots the entropy function of epsilon and save it in a file.png.


2- epsi_utility(y, step):

* Inputs : numpy array (y), integer (step)
* Outputs : plot the utility function of epsilon, 
			float	(EpsioptiU)

	This function iterates through various epsilon values within a range from 0 to the maximum distance between each pair of points in the trajectory, advancing from a user-defined step. It uses the Markov model of the symbolic serie the calculate the composed maximal trace of the transition matrix from the Markov model, the probability to leave a transcient, and the probability to come to a transcient. This composed computation is called utility function. Then, it returns and save in a file.txt the optimal epsilon corresponding to the composed maximal utility function. Moreover, it plots the utility function with respect to epsilon and save it in a file.png.


3- test_epsi(y):

* Inputs : numpy array (y)
* Outputs : *

	This function permits to quickly change the epsilon's value while iterating the whole RASCoPy program (all the functions described in module 1 and 2). When the RASCoPy program ends, you can decide either to save the results if they are satisfying, or try a new value of epsilon if they are not.


**************************************************************
************************Module 4: demo************************
**************************************************************

1- demo_LotkaVoltera():

	This is a demo function that computes all the previous functions on the trajectory of a Lotka Voltera model.
