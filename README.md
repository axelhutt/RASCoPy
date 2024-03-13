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

1- rec_mat(y, epsilon, visu=None, back_file=None):

* Inputs : (:,{1,2, or 3}) numpy array (y), integer (epsilon), two optional parameters : "visu" (if not None -> print the resulting matrix), and "back_file" (name of the backup file if you want to save the results) 
* Outputs : numpy array (R)

	"y" is the coordinate array of each point of a trajectory (1D, 2D, or 3D), the dimensions must correspond to the number of columns of "y", and "epsilon" is the criterion that determines the maximum admissible distance between two points of the trajectory to be considered as recurrent.

	The output is the recurrence matrix "R", composed of 0s and 1s, representing the presence (1) or absence (0) of recurrence between each pair of points in the trajectory. It is computed by calculating the distance matrix of each pair of points of the trajectory y. 1 appears when the distance is smaller than "epsilon" and 0 when it bigger.


2- rec_plt(R):

* Inputs : numpy array of 2 dimensions (R)
* Outputs : recurrence plot

	This function plots the recurrence plot of the input recurrence matrix R and saves it as a file.png.


3- anim_traj(y):

* Inputs : (:,{2 or 3}) numpy array (y)
* Ouputs : GIF file representing points moving on the trajectory "y"

	This function plots an animation of trajectory contained in "y" with points moving on it.


4- col_rec_plt(serie, R):

* Inputs : list of integer (serie), numpy array of 2 dimensions (R)
* Outputs : colored recurrence plot

	This function plots the recurrence plot of the input recurrence matrix "R" with colored dots corresponding to the state attributed to each point during the "symbolic serie" function. (see "Module 3: 		symbolic_series"). Moreover this functions permits to save the figure in a file.png.


**************************************************************
*******************Module 2: symbolic_series******************
**************************************************************

1- symbolic_serie(R, visu=None, back_file=None):

* Inputs : numpy array 2 dimensions (R), two optional parameters : "visu" (if not None -> print the resulting matrix), and "back_file" (name of the backup file if you want to save the results) 
* Outputs : numpy array 1 dimension (newSerie)

	This function computes the symbolic series of the metastable states of the trajectory thanks to its recurrence matrix. All recurrent points creates together a metastable state. The symbolic serie is a 	list of integer that describes through which states the trajectory goes. For example, if the serie is [1,1,1,0,0,2,2,2,2,0,0,1,1], then the trajectory has two metastable states (1 and 2) and goes from 	the state 1 to state 2 to state 1 again. 0 are called transcient, they appear when a point is not recurrent in the trajectory. They are passage states from a metastable state to another.
	
	Moreover the function give the possibility to the user to save the symbolic serie in a back_file.txt.


2- colored_sym_serie(serie, y):

* Inputs : numpy array 1 dimension (serie), (:,{1,2, or 3}) numpy array (y)
* Outputs : figure of the colored symbolic serie

	This function plots the colored symbolic serie in a figure where each different state has a different color and transcients are represented in red.
	Moreover this functions permits to save the figure in a file.png.


3- plot_col_traj(serie, y):

* Inputs : numpy array 1 dimension (serie), (:,{1,2, or 3}) numpy array (y)
* Outputs : plot the trajectory with colored different states

	This function plots the trajectory "y" in colors with respect to the symbolic serie (serie) where each different state has a different color and transcients are represented in red.
	If "y" is a 3-Dimensions trajectory, the function will also plot three 2D colored view (x;y / x;z / y;z).
	Moreover this functions permits to save the figure in a file.png.


4- complexity(serie, visu=None, back_file=None):

* Inputs : numpy array 1 dimension (serie), two optional parameters : "visu" (if not None -> print the resulting matrix), and "back_file" (name of the backup file if you want to save the results) 
* Outputs : 3 integer (C_alphabet_size, C_nbr_words, C_LZ)

	This function computes and returns the complexity of a symbolic serie (serie) with 3 different methods: the alphabet size, the number of words, and the Lempel_Ziv complexity. Moreover, the user can save these 3 integer into a back_file.txt.


5- complexity_shuffle(y, step, count=100, back_file = None):

* Inputs : (:,{1,2, or 3}) numpy array (y), float (step), optionnal input : integer (count)
* Ouputs : plot an histogram of the complexity measures

	This function randomise multiple times (default : "count" = 100) the 3D points of "y". Then, for each randomisation, it computes the three complexity measures.
	Finally, it plots the histogram of the complexity measures and plot the one before randomisation in red. This permits to see if the real complexity can be interpreted or if it is too close to the one of 	a random trajectory.
	Moreover this functions permits to save the figure in a file.png.


**************************************************************
**********************Module 3: espi_opti*********************
**************************************************************

1- epsi_entropy(y, step, visu=None, back_file=None):

* Inputs : (:,{1,2, or 3}) numpy array (y), float (step), two optional parameters : "visu" (if not None -> print the resulting matrix), and "back_file" (name of the backup file if you want to save the results) 
* Outputs : plot the entropy function of epsilon, float	(EpsioptiH)

	This function iterates through various epsilon values within a range from 0 to the maximum distance between each pair of points in the trajectory, advancing from a user-defined step. It computes during 	each iteration the Shannon's entropy of the symbolic serie. Then, it returns and save in a file.txt the optimal epsilon corresponding to the maximal entropy. Moreover, it plots the entropy function of 	epsilon and save it in a back_file.png.


2- epsi_utility(y, step, visu=None, back_file=None):

* Inputs : (:,{1,2, or 3}) numpy array (y), float (step), two optional parameters : "visu" (if not None -> print the resulting matrix), and "back_file" (name of the backup file if you want to save the results) 
* Outputs : plot the utility function of epsilon, float	(EpsioptiU)

	This function iterates through various epsilon values within a range from 0 to the maximum distance between each pair of points in the trajectory, advancing from a user-defined step. It uses the Markov 	model of the symbolic serie the calculate the composed maximal trace of the transition matrix from the Markov model, the probability to leave a transcient, and the probability to come to a transcient. 	This composed computation is called utility function. Then, it returns and save in a file.txt the optimal epsilon corresponding to the composed maximal utility function. Moreover, it plots the utility 	function with respect to epsilon and save it in a back_file.png.


3- test_epsi(y):

* Inputs : (:,{1,2, or 3}) numpy array (y)
* Outputs : *

	This function permits to quickly change the epsilon's value while iterating the whole RASCoPy program (all the functions described in module 1 and 2). When the RASCoPy program ends, you can decide either 	to save the results if they are satisfying, or try a new value of epsilon if they are not.

4- opti_epsi_phi(y, length, step, visu=None, back_file=None):

* Inputs : (:,{1,2, or 3}) numpy array (y), integer (length), float (step), two optional parameters : "visu" (if not None -> print the resulting matrix), and "back_file" (name of the backup file if you want to save the results) 
* Outputs : float (EpsioptiH)

	This function determines the optimal value for epsilon. It takes into account an ideal transition matrix based on a markov model where to go from a metastable states to another, trajectory must before 	goes through the "0" state. The function tests a range of epsilons and determines the one that gives the transititon matrix the closest to the ideal one.


**************************************************************
************************Module 4: demo************************
**************************************************************

1- demo_LotkaVoltera(method):

	This is a demo function that computes all the previous functions on the trajectory of a Lotka Voltera model.
 	The input "method" defines the epsilon determination's method. (see paragraph "methods")
 
2- demo_Lorenz(method):

 	This is a demo function that computes all the previous functions on the trajectory of a Lorenz Attractor model.
  	The input "method" defines the epsilon determination's method. (see paragraph "methods")

3- demo_Square():

	This is a demo function that computes all the previous functions on a simple 2D square trajectory with deceleration arround the corners and acceleration in the straight lines.


**************************************************************
**********************Module 5: rascopy***********************
**************************************************************

MAIN FUNCTION OF THE PACKAGE :

1- rascopy(start_frame=None, end_frame=None, dimensions=3):

* Inputs : 3 optionnal inputs : integer (start_frame), integer (end_frame), integer (dimensions)
* Ouputs : The entire recurrence analysis of videos contained in a file.json.

	This function is like a software of recurrence analysis. With this function you can decide : which json to open, which video, which joint or group of joints you want to analyze (see paragraph "joints"), 	which features of analysis you want to see and/or save (see paragraph "features"), in which dimensions ("dimensions") you want to make the analysis and how many frames you want to take into account 		("start_frame", and "end_frame").  


***********************************************************************************************Methods**********************************************************************************************************

These are the numbers attributed to each method to determine the optimal epsilon :
0 : entropy
1 : utility function
2 : mean of entropy and utility
3 : test_epsi
4 : phi function


***********************************************************************************************Features**********************************************************************************************************

These are the numbers attributed to each feature of the analysis process :
1 : 1, 2 or 3 coordinates function of time plot
2 : 1, 2 or 3D Trajectory
3 : Animated 2D or 3D trajectory
4 : Epsilon results (plot (for entropy and utility functions) and value)
5 : Recurrence matrix
6 : Recurrence plot
7 : Symbolic serie
8 : Colored symbolic serie
9 : Colored recurrence plot
10 : Colored 1, 2 and/or 3D trajectory 
11 : Complexity measures


***********************************************************************************************Joints**********************************************************************************************************
0 : Spine Base
1 : Spine Mid
2 : Neck
3 : Head
4 : Shoulder Left
5 : Elbow Left
6 : Wrist Left
7 : Hand Left
8 : Shoulder Right
9 : Elbow Right
10 : Wrist Right
11 : Hand Right
12 : Hip Left
13 : Knee Left
14 : Ankle Left
15 : Foot Left
16 : Hip Right
17 : Knee Right
18 : Ankle Right
19 : Foot Right
20 : Spine-Shoulder intersection 
21 : Hand Tip Left
22 : Thumb Left
23 : Hand Tip Right
24 : Thumb Right


