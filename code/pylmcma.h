
#include "math.h"
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>

/* 	reference: Ilya Loshchilov "A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization", GECCO 2014, to appear.
	This code implements (together with sep-CMA-ES, Cholesky-CMA-ES, (1+1)-CMA-ES):
	i) the original LM-CMA-ES as published in GECCO (options: line 1613 algorithmType = 0, USE_ZIGGURAT = 0, line 1616 sample_symmetry = false)
	ii) a version of LM-CMA-ES where v vectors are computed correctly as suggested by Oswin Krause (July 2014) (options: line 1613 algorithmType = 10, USE_ZIGGURAT = 0, line 1616 sample_symmetry = false)
	iii) algorithms i) or ii) with a nice approarch to reduce CPU complexity by a factor of 2 by sampling m+sigma*A*z and m-sigma*A*z with the same z (line 1616 sample_symmetry = true)
		and the Ziggurat method for Gaussian sampling (USE_ZIGGURAT = 1) 
*/

#define USE_ZIGGURAT 0

typedef struct 
/* random_t 
 * sets up a pseudo random number generator instance 
 */
{
  /* Variables for Uniform() */
  long int startseed;
  long int aktseed;
  long int aktrand;
  long int *rgrand;
  
  /* Variables for Gauss() */
  short flgstored;
  double hold;
} random_t; 

typedef struct 
{
	double value;
	int id;
} sortedvals;

typedef struct
{
	random_t ttime;
	double*	func_tempdata;
	double*	x_tempdata;
	double*	rotmatrix;
	double* func_shiftxi;
	time_t	time_tic_t;
	clock_t	time_tic_c;
	time_t	time_toc_t;
	clock_t	time_toc_c;
} global_t;


class MyLMCMA {
	public:
		MyLMCMA(int N, int lambda, int inseed, double init_sigma, double* init_xmean);
		~MyLMCMA();
		void get_arx(double* output);
		void set_arfitness(double* arf);
		void update();

	private:
		int N;
		int lambda;
		int mu;
		double ccov;
		int nvectors;
		int maxsteps;
		double cc;
		double val_target;
		double sigma;
		double c_s;
		double target_f;
		int maxevals;
		// m*n
		double* arx;
		double* v_arr;
		double* pc_arr;
		// n
		double* pc;
		double* xmean;
		double* xold;
		double* z;
		double* Az;
		double* Av;
		// lambda, mu, nvectors
		double* weights;
		int* iterator;
		double* arfitness;
		double* prev_arfitness;
		int* arindex;
		double* mixed;
		int* ranks;
		int* ranks_tmp;
		double* Nj_arr;
		double* Lj_arr;
		sortedvals* arr_tmp;
		int* t;
		int* vec;
		global_t gt;
		int inseed;
		bool sample_symmetry;
		int counteval;
		int iterator_sz;
		double K;
		double M;
		double BestF;
		double mueff;
		int itr;
		double s;
};
