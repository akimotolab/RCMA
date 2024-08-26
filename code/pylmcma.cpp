#include "pylmcma.h"

#include <iostream>
using namespace std;

void init_gt(global_t* gt)
{
	gt->func_tempdata = NULL;
	gt->x_tempdata = NULL;
	gt->rotmatrix = NULL;
	gt->func_shiftxi = NULL;
}

void free_gt(global_t* gt)
{
	if (gt->func_tempdata)	{	delete[] gt->func_tempdata;		gt->func_tempdata = NULL;	}
	if (gt->x_tempdata)		{	delete[] gt->x_tempdata;		gt->x_tempdata = NULL;		}
	if (gt->rotmatrix)		{	delete[] gt->rotmatrix;			gt->rotmatrix = NULL;		}
	if (gt->func_shiftxi)	{	delete[] gt->func_shiftxi;		gt->func_shiftxi = NULL;		}
}

/* random_Start(), random_init(), random_exit(), random_Unifo()rm, random_Gauss(), time_tic(), time_tictoc(), time_toc() are adopted 
   from Nikolaus Hansen's source code for CMA-ES
*/

void random_exit(random_t *t)
{
	free( t->rgrand);
}

long random_Start( random_t *t, long unsigned inseed)
{
	long tmp;
	int i;

	t->flgstored = 0;
	t->startseed = inseed;
	if (inseed < 1)
		inseed = 1; 
	t->aktseed = inseed;
	for (i = 39; i >= 0; --i)
	{
		tmp = t->aktseed/127773;
		t->aktseed = 16807 * (t->aktseed - tmp * 127773)
			- 2836 * tmp;
		if (t->aktseed < 0) t->aktseed += 2147483647;
		if (i < 32)
			t->rgrand[i] = t->aktseed;
	}
	t->aktrand = t->rgrand[0];
	return inseed;
}

long random_init( random_t *t, long unsigned inseed)
{
	clock_t cloc = clock();

	t->flgstored = 0;
	t->rgrand = new long[32];
	if (inseed < 1) {
		while ((long) (cloc - clock()) == 0)
			; 
		inseed = (long)abs(100.0*time(NULL)+clock());
	}
	return random_Start(t, inseed);
}

double random_Uniform( random_t *t)
{
	long tmp;

	tmp = t->aktseed/127773;
	t->aktseed = 16807 * (t->aktseed - tmp * 127773)
		- 2836 * tmp;
	if (t->aktseed < 0) 
		t->aktseed += 2147483647;
	tmp = t->aktrand / 67108865;
	t->aktrand = t->rgrand[tmp];
	t->rgrand[tmp] = t->aktseed;
	return (double)(t->aktrand)/(2.147483647e9);
}

/* --------------------------------------------------------- */

/* position of right-most step */
#define PARAM_R 3.44428647676

/* tabulated values for the heigt of the Ziggurat levels */
static const double ytab[128] = {
  1, 0.963598623011, 0.936280813353, 0.913041104253,
  0.892278506696, 0.873239356919, 0.855496407634, 0.838778928349,
  0.822902083699, 0.807732738234, 0.793171045519, 0.779139726505,
  0.765577436082, 0.752434456248, 0.739669787677, 0.727249120285,
  0.715143377413, 0.703327646455, 0.691780377035, 0.68048276891,
  0.669418297233, 0.65857233912, 0.647931876189, 0.637485254896,
  0.62722199145, 0.617132611532, 0.607208517467, 0.597441877296,
  0.587825531465, 0.578352913803, 0.569017984198, 0.559815170911,
  0.550739320877, 0.541785656682, 0.532949739145, 0.524227434628,
  0.515614886373, 0.507108489253, 0.498704867478, 0.490400854812,
  0.482193476986, 0.47407993601, 0.466057596125, 0.458123971214,
  0.450276713467, 0.442513603171, 0.434832539473, 0.427231532022,
  0.419708693379, 0.41226223212, 0.404890446548, 0.397591718955,
  0.390364510382, 0.383207355816, 0.376118859788, 0.369097692334,
  0.362142585282, 0.355252328834, 0.348425768415, 0.341661801776,
  0.334959376311, 0.328317486588, 0.321735172063, 0.31521151497,
  0.308745638367, 0.302336704338, 0.29598391232, 0.289686497571,
  0.283443729739, 0.27725491156, 0.271119377649, 0.265036493387,
  0.259005653912, 0.253026283183, 0.247097833139, 0.241219782932,
  0.235391638239, 0.229612930649, 0.223883217122, 0.218202079518,
  0.212569124201, 0.206983981709, 0.201446306496, 0.195955776745,
  0.190512094256, 0.185114984406, 0.179764196185, 0.174459502324,
  0.169200699492, 0.1639876086, 0.158820075195, 0.153697969964,
  0.148621189348, 0.143589656295, 0.138603321143, 0.133662162669,
  0.128766189309, 0.123915440582, 0.119109988745, 0.114349940703,
  0.10963544023, 0.104966670533, 0.100343857232, 0.0957672718266,
  0.0912372357329, 0.0867541250127, 0.082318375932, 0.0779304915295,
  0.0735910494266, 0.0693007111742, 0.065060233529, 0.0608704821745,
  0.056732448584, 0.05264727098, 0.0486162607163, 0.0446409359769,
  0.0407230655415, 0.0368647267386, 0.0330683839378, 0.0293369977411,
  0.0256741818288, 0.0220844372634, 0.0185735200577, 0.0151490552854,
  0.0118216532614, 0.00860719483079, 0.00553245272614, 0.00265435214565
};

/* tabulated values for 2^24 times x[i]/x[i+1],
 * used to accept for U*x[i+1]<=x[i] without any floating point operations */
static const unsigned long ktab[128] = {
  0, 12590644, 14272653, 14988939,
  15384584, 15635009, 15807561, 15933577,
  16029594, 16105155, 16166147, 16216399,
  16258508, 16294295, 16325078, 16351831,
  16375291, 16396026, 16414479, 16431002,
  16445880, 16459343, 16471578, 16482744,
  16492970, 16502368, 16511031, 16519039,
  16526459, 16533352, 16539769, 16545755,
  16551348, 16556584, 16561493, 16566101,
  16570433, 16574511, 16578353, 16581977,
  16585398, 16588629, 16591685, 16594575,
  16597311, 16599901, 16602354, 16604679,
  16606881, 16608968, 16610945, 16612818,
  16614592, 16616272, 16617861, 16619363,
  16620782, 16622121, 16623383, 16624570,
  16625685, 16626730, 16627708, 16628619,
  16629465, 16630248, 16630969, 16631628,
  16632228, 16632768, 16633248, 16633671,
  16634034, 16634340, 16634586, 16634774,
  16634903, 16634972, 16634980, 16634926,
  16634810, 16634628, 16634381, 16634066,
  16633680, 16633222, 16632688, 16632075,
  16631380, 16630598, 16629726, 16628757,
  16627686, 16626507, 16625212, 16623794,
  16622243, 16620548, 16618698, 16616679,
  16614476, 16612071, 16609444, 16606571,
  16603425, 16599973, 16596178, 16591995,
  16587369, 16582237, 16576520, 16570120,
  16562917, 16554758, 16545450, 16534739,
  16522287, 16507638, 16490152, 16468907,
  16442518, 16408804, 16364095, 16301683,
  16207738, 16047994, 15704248, 15472926
};

/* tabulated values of 2^{-24}*x[i] */
static const double wtab[128] = {
  1.62318314817e-08, 2.16291505214e-08, 2.54246305087e-08, 2.84579525938e-08,
  3.10340022482e-08, 3.33011726243e-08, 3.53439060345e-08, 3.72152672658e-08,
  3.8950989572e-08, 4.05763964764e-08, 4.21101548915e-08, 4.35664624904e-08,
  4.49563968336e-08, 4.62887864029e-08, 4.75707945735e-08, 4.88083237257e-08,
  5.00063025384e-08, 5.11688950428e-08, 5.22996558616e-08, 5.34016475624e-08,
  5.44775307871e-08, 5.55296344581e-08, 5.65600111659e-08, 5.75704813695e-08,
  5.85626690412e-08, 5.95380306862e-08, 6.04978791776e-08, 6.14434034901e-08,
  6.23756851626e-08, 6.32957121259e-08, 6.42043903937e-08, 6.51025540077e-08,
  6.59909735447e-08, 6.68703634341e-08, 6.77413882848e-08, 6.8604668381e-08,
  6.94607844804e-08, 7.03102820203e-08, 7.11536748229e-08, 7.1991448372e-08,
  7.2824062723e-08, 7.36519550992e-08, 7.44755422158e-08, 7.52952223703e-08,
  7.61113773308e-08, 7.69243740467e-08, 7.77345662086e-08, 7.85422956743e-08,
  7.93478937793e-08, 8.01516825471e-08, 8.09539758128e-08, 8.17550802699e-08,
  8.25552964535e-08, 8.33549196661e-08, 8.41542408569e-08, 8.49535474601e-08,
  8.57531242006e-08, 8.65532538723e-08, 8.73542180955e-08, 8.8156298059e-08,
  8.89597752521e-08, 8.97649321908e-08, 9.05720531451e-08, 9.138142487e-08,
  9.21933373471e-08, 9.30080845407e-08, 9.38259651738e-08, 9.46472835298e-08,
  9.54723502847e-08, 9.63014833769e-08, 9.71350089201e-08, 9.79732621669e-08,
  9.88165885297e-08, 9.96653446693e-08, 1.00519899658e-07, 1.0138063623e-07,
  1.02247952126e-07, 1.03122261554e-07, 1.04003996769e-07, 1.04893609795e-07,
  1.05791574313e-07, 1.06698387725e-07, 1.07614573423e-07, 1.08540683296e-07,
  1.09477300508e-07, 1.1042504257e-07, 1.11384564771e-07, 1.12356564007e-07,
  1.13341783071e-07, 1.14341015475e-07, 1.15355110887e-07, 1.16384981291e-07,
  1.17431607977e-07, 1.18496049514e-07, 1.19579450872e-07, 1.20683053909e-07,
  1.21808209468e-07, 1.2295639141e-07, 1.24129212952e-07, 1.25328445797e-07,
  1.26556042658e-07, 1.27814163916e-07, 1.29105209375e-07, 1.30431856341e-07,
  1.31797105598e-07, 1.3320433736e-07, 1.34657379914e-07, 1.36160594606e-07,
  1.37718982103e-07, 1.39338316679e-07, 1.41025317971e-07, 1.42787873535e-07,
  1.44635331499e-07, 1.4657889173e-07, 1.48632138436e-07, 1.50811780719e-07,
  1.53138707402e-07, 1.55639532047e-07, 1.58348931426e-07, 1.61313325908e-07,
  1.64596952856e-07, 1.68292495203e-07, 1.72541128694e-07, 1.77574279496e-07,
  1.83813550477e-07, 1.92166040885e-07, 2.05295471952e-07, 2.22600839893e-07
};

double gsl_ran_gaussian_ziggurat (random_t *t)
{
  unsigned long  U, sign, i, j;
  double  x, y;

  while (1) {
    U = random_Uniform(t)*4294967295;
    i = U & 0x0000007F;		/* 7 bit to choose the step */
    sign = U & 0x00000080;	/* 1 bit for the sign */
    j = U>>8;			/* 24 bit for the x-value */

    x = j*wtab[i];
    if (j < ktab[i])  break;

    if (i<127) {
      double  y0, y1;
      y0 = ytab[i];
      y1 = ytab[i+1];
      y = y1+(y0-y1)*random_Uniform(t);
    } else {
      x = PARAM_R - log(1.0-random_Uniform(t))/PARAM_R;
      y = exp(-PARAM_R*(x-0.5*PARAM_R))*random_Uniform(t);
    }
    if (y < exp(-0.5*x*x))  break;
  }
  return  sign ? x : -x;
}
 

double random_Gauss(random_t *t)
{
	if (USE_ZIGGURAT)
		return gsl_ran_gaussian_ziggurat (t);
	double x1, x2, rquad, fac;

	if (t->flgstored)
	{    
		t->flgstored = 0;
		return t->hold;
	}
	do 
	{
		x1 = 2.0 * random_Uniform(t) - 1.0;
		x2 = 2.0 * random_Uniform(t) - 1.0;
		rquad = x1*x1 + x2*x2;
	} while(rquad >= 1 || rquad <= 0);
	fac = sqrt(-2.0*log(rquad)/rquad);
	t->flgstored = 1;
	t->hold = fac * x1;
	return fac * x2;
}

void	time_tic(global_t* t)
{
	t->time_tic_t = time(NULL);	// measure time in seconds
	t->time_tic_c = clock();	// measure time in microseconds up to ~2k seconds
}

double	time_tictoc(global_t* t)
{
	double dt = difftime(t->time_toc_t,t->time_tic_t);
	if (dt < 1000)
		dt = (double)(t->time_toc_c - t->time_tic_c) / CLOCKS_PER_SEC;
	return dt;
}

double	time_toc(global_t* t)
{
	t->time_toc_t = time(NULL);
	t->time_toc_c = clock();
	return time_tictoc(t);
}

void matrix_eye(double* M, int m)
{
	for (int i=0; i<m; i++)
		for (int j=0; j<m; j++)
			if (i == j)	M[i*m + j] = 1.0;
			else		M[i*m + j] = 0.0;
}


// vector res = matrix a X vector b
void matrix_mult_vector(double* res, double* a, double* b,int m)
{
	double val = 0.0;
	for (int i=0; i<m; i++)
	{
		val = 0.0;
		for (int j=0; j<m; j++)
			val += a[i*m + j] * b[j];
		res[i] = val;
	}
}


// maxtrix res = matrix a X matrix b
void matrix_mult_matrix(double* res, double* a, double* b,int m)
{
	double val;
	for (int i=0; i<m; i++)
		for (int j=0; j<m; j++)
		{
			val = 0;
			for (int k=0; k<m; k++)
				val += a[i*m + k] * b[k*m + j];
			res[i*m + j] = val;
		}
}


// matrix res = vector a X vector b
void vector_mult_vector(double* res, double* a, double* b,int m)
{
	for (int i=0; i<m; i++)
		for (int j=0; j<m; j++)
			res[i*m + j] = a[i] * b[j];
}


// vector res = vector a X matrix b
void vector_mult_matrix(double* res, double* a, double* b,int m)
{
	double val;
	for (int i=0; i<m; i++)
	{
		val = 0;
		for (int j=0; j<m; j++)
			val += a[j] * b[j*m + i];
		res[i] = val;
	}
}

double vector_prod(double* a, double* b,int m)
{
	double res = 0.0;
	for (int i=0; i<m; i++)
		res += a[i] * b[i];
	return res;
}

void generateRotationMatrix(double* B, int N, double* tmp1, random_t* rnd)
{
	double* pB;

	for (int i=0; i<N; i++)
		for (int j=0; j<N; j++)
			B[i*N + j] = random_Gauss(rnd);
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<i; j++)
		{
			double ariarj = 0;
			for (int k=0; k<N; k++)
				ariarj = ariarj + B[k*N+i]*B[k*N+j];
			
			for (int k=0; k<N; k++)
				B[k*N+i] = B[k*N+i] - ariarj * B[k*N+j];
		}
		double normv = 0;
		for (int k=0; k<N; k++)
			normv = normv + B[k*N+i]*B[k*N+i];
			
		normv = sqrt(normv);
		for(int k=0; k<N; k++)
			B[k*N+i] = B[k*N+i] / normv;
	}
}

double minv(double a, double b)
{
	if (a < b)	return a;
	else		return b;
}

double maxv(double a, double b)
{
	if (a > b)	return a;
	else		return b;
}


int compare(const void * a, const void * b)
{
	if (((sortedvals*)a)->value < ((sortedvals*)b)->value)	return -1;
	if (((sortedvals*)a)->value == ((sortedvals*)b)->value)	return 0;
	if (((sortedvals*)a)->value > ((sortedvals*)b)->value)	return 1;
}

void myqsort(int sz, double* arfitness, int* arindex, sortedvals* arr)
{
	for(int i=0; i<sz; i++)
	{
		arr[i].value = arfitness[i];
		arr[i].id = i;
	}
	
	qsort( arr, sz, sizeof(sortedvals), compare);
	for(int i=0; i<sz; i++)
	{
		arfitness[i] = arr[i].value;
		arindex[i] = arr[i].id;
	}
}


void invAz(int N, double* Av, int iterator_sz, int* iterator, double* v_arr, double* Lj_arr, double K)
{
	for(int j=0; j<iterator_sz; j++)
	{
		int jcur = iterator[j];			
		double* v_j = &v_arr[jcur * N];
		double v_j_mult_Av = 0;
		for(int p=0; p<N; p++)					
			v_j_mult_Av += v_j[p] * Av[p];
		v_j_mult_Av = Lj_arr[jcur] * v_j_mult_Av; 
		for(int p=0; p<N; p++)
			Av[p] = K * Av[p] - v_j_mult_Av * v_j[p];
	}
}


MyLMCMA::MyLMCMA(int N, int lambda, int inseed, double init_sigma, double* init_xmean) {

	this->N = N;
	this->lambda = lambda;
	this->mu = int(lambda/2);		// 	number of parents, e.g., floor(lambda/2);
	this->ccov = 1/(10*log(double(N)+1.0));// 	learning rate for covariance matrix, e.g., 1/(10*log(N+1))
	this->nvectors = lambda;	//	number of stored direction vectors, e.g., nvectors = 4+floor(3*log(N))
	this->maxsteps = nvectors;	//	target number of generations between vectors, e.g., maxsteps = nvectors
	this->cc = double(1.0/double(nvectors));	// learning rate for mean vector's evolution path, e.g., cc = 1/nvectors
	this->val_target = 0.25;	// target success rate for new population, e.g., 0.25
	this->sigma = init_sigma;	// initial step-size
	this->c_s = 0.3;	//	decay factor for step-size adaptation, e.g., 0.3
	this->counteval = 0;
	this->iterator_sz = 0;
	this->itr = 0;
	this->s = 0;
	this->sample_symmetry = false;

	// memory allocation
	// m*n
	this->arx = new double[N*lambda];
	this->v_arr = new double[N*(nvectors)];
	this->pc_arr = new double[N*(nvectors)];

	// n
	this->pc = new double[N];
	this->xmean = new double[N];
	this->xold = new double[N];
	this->z = new double[N];
	this->Az = new double[N];
	this->Av = new double[N];

	// lambda, mu, nvectors
	this->weights = new double[mu];
	this->iterator = new int[nvectors];		
	this->arfitness = new double[lambda];
	this->prev_arfitness = new double[lambda];
	this->arindex = new int[lambda];
	this->mixed = new double[2*lambda];
	this->ranks = new int[2*lambda];
	this->ranks_tmp = new int[2*lambda];
	this->Nj_arr = new double[nvectors];
	this->Lj_arr = new double[nvectors];
	this->arr_tmp = new sortedvals[2*lambda];
	this->t = new int[nvectors];
	this->vec = new int[nvectors];

	init_gt(&gt);
	// memory initialization
	random_init(&gt.ttime , inseed);

	double sum_weights = 0;
	for(int i=0; i<mu; i++)
	{
		weights[i] = log(double(mu+0.5)) - log(double(1+i));
		sum_weights = sum_weights + weights[i];
	}
	mueff = 0;
	for(int i=0; i<mu; i++)
	{
		weights[i] = weights[i] / sum_weights;
		mueff = mueff + weights[i]*weights[i];
	}
	mueff = 1 / mueff;

	for(int i=0; i<N; i++) {
		pc[i] = 0;
		this->xmean[i] = init_xmean[i];
	}
	K = 1/sqrt(1 - ccov);
	M = sqrt(1 - ccov);
}

MyLMCMA::~MyLMCMA() {
	random_exit(&gt.ttime);
	delete[] arr_tmp;		delete[] weights;	delete[] pc;	delete[] xmean;	
	delete[] xold; 			delete[] z;			delete[] Az;	delete[] iterator;	
	delete[] v_arr;			delete[] pc_arr;	delete[] arx;	delete[] arfitness;
	delete[] prev_arfitness;delete[] arindex;	delete[] Av;	delete[] t;	
	delete[] vec;			delete[] mixed;		delete[] ranks;	delete []ranks_tmp;
	delete[] Nj_arr;		delete[] Lj_arr;	
	free_gt(&gt);
}

void MyLMCMA::get_arx(double* output) {
	int sign = 1; 
	for( int i=0; i<lambda; i++) // O(lambda*m*n)
	{
		if (sign == 1)
		{ 
			for(int k=0; k<N; k++)	// O(n)
			{
				z[k] = random_Gauss(&gt.ttime);
				Az[k] = z[k];
			}
		
			for(int k=0; k<iterator_sz; k++)	// O(m*n)
			{
				int jcur = iterator[k];				
				double* pc_j = &pc_arr[jcur*N];
				double* v_j = &v_arr[jcur*N];
				double v_j_mult_z = 0;
				for(int p=0; p<N; p++)	
					v_j_mult_z = v_j_mult_z + v_j[p] * z[p];
				v_j_mult_z = Nj_arr[jcur] * v_j_mult_z; 				
				for(int p=0; p<N; p++)
					Az[p] = M * Az[p] + v_j_mult_z * pc_j[p];
			}

			if (0) // check A and invA
			{
				for(int k=0; k<N; k++)
					Av[k] = Az[k];
				invAz(N, Av,iterator_sz, iterator, v_arr, Lj_arr, K);
				double div = 0;
				for(int k=0; k<N; k++)
				{
					div += abs(Av[k] - z[k]);
				}
				if ((abs(div) > 1e-6))//&&(counteval%10000 == 0))
				{
					printf("%d div:%g\n",counteval,div);
				}
			}
		}
		for(int k=0; k<N; k++) {// O(n)
			arx[i*N + k] = xmean[k] + sign*sigma*Az[k];
			output[i*N + k] = arx[i*N + k];
		}
		if (sample_symmetry) // sample in the opposite direction, seems to work better in most cases AND decreases CPU time by 2.0
			sign = -sign;
	}
}

void MyLMCMA::set_arfitness(double* arf) {
	for( int i=0; i<lambda; i++) // O(lambda*m*n)
	{
		arfitness[i] = arf[i];
		counteval = counteval + 1;
		if (counteval == 1)	BestF = arfitness[i];
		if (arfitness[i] < BestF)	BestF = arfitness[i];

		if (counteval%100000 == 0)
			printf("%d %g\n",counteval,BestF);
	}
}

void MyLMCMA::update() {

	myqsort(lambda, arfitness, arindex, arr_tmp);
	
	for(int i=0; i<N; i++)
	{			
		xold[i] = xmean[i];
		xmean[i] = 0;
	}

	for(int i=0; i<mu; i++)
	{
		double* cur_x = &arx[arindex[i] * N];
		for(int j=0; j<N; j++)
			xmean[j] += weights[i] * cur_x[j];
	}

	for(int i=0; i<N; i++)
		pc[i] = (1 - cc)*pc[i] + sqrt(cc*(2-cc)*mueff)*(xmean[i] - xold[i])/sigma;

	int imin = 1;
	if (itr < nvectors)
	{
		t[itr] = itr;			
	}
	else
	{
		int dmin = vec[t[1]] - vec[t[0]];
		
		for(int j=1; j<(nvectors-1); j++)
		{
			int dcur = vec[t[j+1]] - vec[t[j]];
			if (dcur < dmin)
			{
				dmin = dcur;
				imin = j + 1;
			}
		}
		if (dmin >= maxsteps)
			imin = 0;
		if (imin != (nvectors-1))
		{
			int sav = t[imin];
			for(int j=imin; j<(nvectors-1); j++)
				t[j] = t[j+1];
			t[nvectors-1] = sav;
		}
	}

	iterator_sz = itr+1;
	if (iterator_sz > nvectors)	iterator_sz = nvectors;
	for(int i=0; i<iterator_sz; i++)
		iterator[i] = t[i];
	int newidx = t[iterator_sz-1];
	vec[newidx] = itr;
		
	for(int i=0; i<N; i++)
		pc_arr[newidx*N + i] = pc[i];

	// this procedure recomputes v vectors correctly, in the original LM-CMA-ES they were outdated/corrupted.
	// the procedure allows to improve the performance on some problems (up to 20% on Ellipsoid with D=128..512) 
	// and sometimes better on other problems
//		for(int i=0; i<iterator_sz; i++) // makes the loop ca. 2 times slower
	if (imin == 1) imin = 0;
	for(int i=imin; i<iterator_sz; i++)
	{
//			int indx = iterator[i]; // makes the loop ca. 2 times slower
		int indx = t[i];
		for(int j=0; j<N; j++)
			Av[j] = pc_arr[indx*N + j]; 
		invAz(N, Av, i, iterator, v_arr, Lj_arr, K);

		for(int j=0; j<N; j++)
			v_arr[indx*N + j] = Av[j];

		double nv = 0;
		for(int j=0; j<N; j++)
			nv += v_arr[indx*N + j]*v_arr[indx*N + j];
		Nj_arr[indx] = (sqrt(1-ccov)/nv)*(sqrt(1+(ccov/(1-ccov))*nv)-1);
		Lj_arr[indx] = (1/(sqrt(1-ccov)*nv))*(1-(1/sqrt(1+((ccov)/(1-ccov))*nv)));
	}
	// end of procedure

	if (itr > 0)
	{
		for(int i=0; i<lambda; i++)
		{
			mixed[i] = arfitness[i];
			mixed[lambda+i] = prev_arfitness[i];
		}
		myqsort(2*lambda, mixed, ranks, arr_tmp);
		double meanprev = 0;
		double meancur = 0;
		for(int i=0; i<2*lambda; i++)
			ranks_tmp[i] = ranks[i];
		for(int i=0; i<2*lambda; i++)
			ranks[ranks_tmp[i]] = i;
		for(int i=0; i<lambda; i++)
		{
			meanprev = meanprev + ranks[i];
			meancur = meancur + ranks[lambda + i];
		}
		meanprev = meanprev / lambda;
		meancur = meancur / lambda;
		double diffv = (meancur - meanprev)/lambda;
		double z1 = diffv - val_target;
		s = (1-c_s)*s + c_s*z1;
		double d_s = 1;//2.0*(N-1.0)/N;
		sigma = sigma * exp(s/d_s);
	}

	for(int i=0; i<lambda; i++)
		prev_arfitness[i] = arfitness[i];
			
	itr = itr + 1;

}

// void main()
// {	
	
// 	int N = 2*128;			//	problem dimension
// 	int lambda = 4+floor(3*log(double(N)));	// 	population size, e.g., 4+floor(3*log(N));
// 	int mu = int(lambda/2);		// 	number of parents, e.g., floor(lambda/2);
// 	double ccov = 1/(10*log(double(N)+1.0));// 	learning rate for covariance matrix, e.g., 1/(10*log(N+1))
// 	double xmin = -5;//	x parameters lower bound
// 	double xmax = 5;//	x parameters upper bound
// 	int nvectors = lambda;	//	number of stored direction vectors, e.g., nvectors = 4+floor(3*log(N))
// 	int maxsteps = nvectors;	//	target number of generations between vectors, e.g., maxsteps = nvectors
// 	double cc = double(1.0/double(nvectors));	// learning rate for mean vector's evolution path, e.g., cc = 1/nvectors
// 	double val_target = 0.25;	// target success rate for new population, e.g., 0.25
// 	double sigma = 0.5*(xmax - xmin);	// initial step-size, e.g., 0.5
// 	double c_s = 0.3;	//	decay factor for step-size adaptation, e.g., 0.3
// 	double target_f = 1e-10;	// target fitness function value, e.g., 1e-10
// 	int maxevals = 1e+8;		// maximum number of function evaluations allowed, e.g., 1e+6
// 	FunctionId FuncId = (FunctionId)int(2);	//	fitness function ID, use pointer to function if needed
// 	int inseed = 1;		// initial seed for random number generator, e.g., 1
// 	int algorithmType = 10;	// 0:LMCMA, 10:LMCMA_fixed+features, 1:sepCMA, 2:Cholesky-CMA, 3: baseline CMA-ES
// 	int printToFile = 0; // 1 or 0

// 	bool sample_symmetry = false; // use 'true' to make the algorithm 2 times faster (in CPU time) and perform better (in terms of number of evaluations)

// 	double output[2];

// 	std::clock_t start;
// 	start = std::clock();

// 	if (algorithmType == -1) // check Vector x Scalar Multiplication Cost
// 	{  //0 - VectorScalar; 1 - MatrixVector; 2 - RandomScalar;	3 - FunctionEvaluation
// 		int OperationType = 0;
// 		output[1] = time_costOfSimpleOperation(OperationType, 10, N, 0, FuncId);
// 		output[0] = maxevals;
// 	}
// 	if (algorithmType == -2) // check Vector x Scalar Multiplication Cost
// 	{ 
// 		int OperationType = 1; //0 - VectorScalar; 1 - MatrixVector; 2 - RandomScalar;	3 - FunctionEvaluation
// 		output[1] = time_costOfSimpleOperation(OperationType, 10, N, 0, FuncId);
// 		output[0] = maxevals;
// 	}
// 	if (algorithmType == -3) // check Vector x Scalar Multiplication Cost
// 	{ 
// 		int OperationType = 2; //0 - VectorScalar; 1 - MatrixVector; 2 - RandomScalar;	3 - FunctionEvaluation
// 		output[1] = time_costOfSimpleOperation(OperationType, 10, N, 0, FuncId);
// 		output[0] = maxevals;
// 	}
// 	if (algorithmType == -4) // check Vector x Scalar Multiplication Cost
// 	{ 
// 		int OperationType = 3; //0 - VectorScalar; 1 - MatrixVector; 2 - RandomScalar;	3 - FunctionEvaluation
// 		output[1] = time_costOfSimpleOperation(OperationType, 10, N, 0, FuncId);
// 		output[0] = maxevals;
// 	}
// 	if (algorithmType == 0)
// 	{
// 		LMCMA(N, lambda, mu, ccov, xmin, xmax, nvectors,
// 			maxsteps, cc, val_target, sigma, c_s, target_f, 
// 			maxevals, FuncId, inseed, output, printToFile, sample_symmetry);	
// 	}
// 	if (algorithmType == 10)
// 	{
// 		LMCMAfixed(N, lambda, mu, ccov, xmin, xmax, nvectors,
// 			maxsteps, cc, val_target, sigma, c_s, target_f, 
// 			maxevals, FuncId, inseed, output, printToFile, sample_symmetry);	
// 	}
// 	if (algorithmType == 1)
// 	{
// 		sepCMA(N, lambda, mu, ccov, xmin, xmax, nvectors,
// 			maxsteps, cc, val_target, sigma, c_s, target_f, 
// 			maxevals, FuncId, inseed, output, printToFile);	
// 	}
// 	if (algorithmType == 2)
// 	{
// 		runCMAmulambdaCholesky(N, lambda, mu, sigma, xmin, xmax, maxevals, target_f, FuncId, 0, inseed, output, printToFile);
// 	}
// 	double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
// 	printf("ievals: %d  bestfit: %g  time:%g sec.  cpu/eval: %g \n",int(output[0]), output[1], duration, duration/double(output[0]));
// 	int i;
// 	std::cin>>i;
// }
