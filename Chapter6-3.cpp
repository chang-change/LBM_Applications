#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>

#define NUM_THREADS  4

using namespace std;

const int n = 100;
const int m = 100;

double f[9][n + 1][m + 1];
double feq[9][n + 1][m + 1];
double rho[n + 1][m + 1];
double u[n + 1][m + 1];
double v[n + 1][m + 1];
double w[9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
double cx[9] = { 0,1,0,-1,0,1,-1,-1,1 };
double cy[9] = { 0,0,1,0,-1,1,1,-1,-1 };

double u0 = 0.2;
double rho0 = 5.0;
const double dx = 1.0;
const double dy = 1.0;
const double dt = 1.0;
double tw = 1.0;
double th = 0.0;
double g = 0.0;
double visco = 0.02;
double pr = 0.71;
double alpha = visco/pr;

double omega = 1.0 / (3.0*visco + 0.5);
double omegat = 1.0 / (3.0*alpha + 0.5);
const int mstep = 20000;

void init()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(rho, n, m, u, v, rho0)
	for (int j = 0; j < m + 1; j++)
	{
		for (int i = 0; i < n + 1; i++)
		{
			rho[i][j] = rho0;
			u[i][j] = 0.0;
			v[i][j] = 0.0;
		}
	}

# pragma omp parallel for shared(u, n, m, v, u0)
	for (int i = 1; i < n; i++)
	{
		u[i][m] = 0.0;
		v[i][m] = 0.0;
	}
}

void collesion()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(u, v, cx, cy, feq, rho, w, omega, f)
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < m + 1; j++)
		{
			double t1 = u[i][j] * u[i][j] + v[i][j] * v[i][j];
			for (int k = 0; k < 9; k++)
			{
				double t2 = u[i][j] * cx[k] + v[i][j] * cy[k];
				feq[k][i][j] = rho[i][j] * w[k] * (1.0 + 3.0*t2 + 4.5*t2*t2 - 1.5*t1);
				f[k][i][j] = omega * feq[k][i][j] + (1.0 - omega)*f[k][i][j];
			}
		}
	}
}