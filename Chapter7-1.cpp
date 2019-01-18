/* This code produce results for lid driven cavity, MRT code */
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>

#define NUM_THREADS  8

using namespace std;

const int n = 100, m = 100;
const int mstep = 100000;

double f[9][n + 1][m + 1];
double feq[9][n + 1][m + 1];
double rho[n + 1][m + 1];
double u[n + 1][m + 1];
double v[n + 1][m + 1];
double w[9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
double cx[9] = { 0,1,0,-1,0,1,-1,-1,1 };
double cy[9] = { 0,0,1,0,-1,1,1,-1,-1 };

double a = 1. / 36;
double tminv[9][9] = { {4.*a, -4.*a,4.*a,0,0,0,0,0,0},{4.*a,-a,-2.*a,6.*a,-6.*a,0,0,9.*a,0},{4.*a,-a,-2.*a,0,0,6.*a,-6.*a,-9.*a,0},
	{4.*a,-a,-2*a,-6*a,6*a,0,0,9*a,0},{4*a,-a,-2*a,0,0,-6*a,6*a,-9*a,0},{4*a,2*a,a,6*a,3*a,6*a,3*a,0,9*a},
	{4*a,2*a,a,-6*a,-3*a,6*a,3*a,0,-9*a},{4*a,2*a,a,-6*a,-3*a,-6*a,-3*a,0,9*a},{4*a,2*a,a,6*a,3*a,-6*a,-3*a,0,-9*a} };
double tm[9][9] = { {1,1,1,1,1,1,1,1,1},{-4,-1,-1,-1,-1,2,2,2,2},{4,-2,-2,-2,-2,1,1,1,1},{0,1,0,-1,0,1,-1,-1,1},
	{0,-2,0,2,0,1,-1,-1,1},{0,0,1,0,-1,1,1,-1,-1},{0,0,-2,0,2,1,1,-1,-1},{0,1,-1,1,-1,0,0,0,0},{0,0,0,0,0,1,1,1,-1} };

double stmiv[9][9];
double ev[9][9];

double u0 = 0.05;
double rho0 = 1.0;
double dx = 1.0;
double dy = 1.0;
double dt = 1.0;

double alpha = 0.001;
double omega = 1.0 / (3.0*alpha + 0.5);
double tau = 1. / omega;
double sm[9] = { 1, 1.4, 1.4, 1, 1.2, 1, 1.2, tau, tau };

void init()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(rho, rho0, u, v)
	for (int j = 0; j < m + 1; j++)
	{
		for (int i = 0; i < n + 1; i++)
		{
			rho[i][j] = rho0;
			v[i][j] = 0.0;
			u[i][j] = 0.0;
		}
	}

# pragma omp parallel for shared(u, v, u0)
	for (int i = 1; i <= n - 1; i++)
	{
		u[i][m] = u0;
		v[i][m] = 0.0;
	}

# pragma omp parallel for shared(tminv, sm, tm, ev)
	for (int i = 0; i <= 8; i++)
	{
		for (int j = 0; j <= 8; j++)
		{
			double sumcc = 0.0;
			for (int k = 0; k <= 8; k++)
				sumcc += tminv[i][1] * tm[1][j];
			ev[i][j] = sumcc;
			stmiv[i][j] = tminv[i][j] * sm[j];
		}
	}
}


void collesion()
{
	double fmom[9][n + 1][m + 1];
	double fmeq[9][n + 1][m + 1];

	omp_set_num_threads(NUM_THREADS);
	/// calculate equilibrium moments
# pragma omp parallel for shared(rho, u, v)
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < m + 1; j++)
		{
			fmeq[0][i][j] = rho[i][j];
			fmeq[1][i][j] = rho[i][j] * (-2.0 + 3.0*rho[i][j] *(u[i][j]* u[i][j] + v[i][j]*v[i][j]) );
			fmeq[2][i][j] = rho[i][j] * (1.0 - 3.0*rho[i][j] * (u[i][j] * u[i][j] + v[i][j] * v[i][j]));
			fmeq[3][i][j] = rho[i][j] * u[i][j];
			fmeq[4][i][j] = -rho[i][j] * u[i][j];
			fmeq[5][i][j] = rho[i][j] * v[i][j];
			fmeq[6][i][j] = -rho[i][j] * v[i][j];
			fmeq[7][i][j] = rho[i][j] * (u[i][j] * u[i][j] - v[i][j] * v[i][j]);
			fmeq[8][i][j] = rho[i][j] * u[i][j] * v[i][j];
		}
	}

	/// calculate moments
# pragma omp parallel for shared(tm, f)
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < m + 1; j++)
		{
			for (int k = 0; k < 9; k++)
			{
				double suma = 0.0;
				for (int L = 0; L < 9; L++)
					suma += tm[k][L] * f[L][i][j];
				fmom[k][i][j] = suma;
			}
		}
	}

	/// calculate moments
# pragma omp parallel for shared(stmiv, fmom, fmeq)
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < m + 1; j++)
		{
			for (int k = 0; k < 9; k++)
			{
				double sumb = 0.0;
				for (int L = 0; L < 9; L++)
					sumb += stmiv[k][L] * (fmom[L][i][j] - fmeq[L][i][j]);
				f[k][i][j] -= sumb;
			}
		}
	}
}


void streaming()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(f)
	for (int j = 0; j < m + 1; j++)
	{
		/// right to left
		for (int i = n; i > 0; i--)
		{
			f[1][i][j] = f[1][i - 1][j];
		}

		/// left to right
		for (int i = 0; i < n; i++)
		{
			f[3][i][j] = f[3][i + 1][j];
		}
	}

# pragma omp parallel for shared(f)
	/// top to bottom
	for (int j = m; j > 0; j--)
	{
		for (int i = 0; i < n + 1; i++)
		{
			f[2][i][j] = f[2][i][j - 1];
		}
		for (int i = n; i > 0; i--)
		{
			f[5][i][j] = f[5][i - 1][j - 1];
		}
		for (int i = 0; i < n; i++)
		{
			f[6][i][j] = f[6][i + 1][j - 1];
		}
	}

# pragma omp parallel for shared(f)
	/// bottom to top
	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n + 1; i++)
		{
			f[4][i][j] = f[4][i][j + 1];
		}
		for (int i = 0; i < n; i++)
		{
			f[7][i][j] = f[7][i + 1][j + 1];
		}
		for (int i = n; i > 0; i--)
		{
			f[8][i][j] = f[8][i - 1][j + 1];
		}
	}
}


void sfbound()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(f)
	for (int j = 0; j < m + 1; j++)
	{
		///  west boundary
		f[1][0][j] = f[3][0][j];
		f[5][0][j] = f[7][0][j];
		f[8][0][j] = f[6][0][j];

		/// east boundary
		f[3][n][j] = f[1][n][j];
		f[7][n][j] = f[5][n][j];
		f[6][n][j] = f[8][n][j];
	}

# pragma omp parallel for shared(f)
	for (int i = 0; i < n + 1; i++)
	{
		/// south boundary
		f[2][i][0] = f[4][i][0];
		f[5][i][0] = f[7][i][0];
		f[6][i][0] = f[8][i][0];
	}

	/// moving lid, north boundary
# pragma omp parallel for shared(f)
	for (int i = 1; i < n; i++)
	{
		double rhon = f[0][i][m] + f[1][i][m] + f[3][i][m] + 2 * (f[2][i][m] + f[6][i][m] + f[5][i][m]);
		f[4][i][m] = f[2][i][m];
		f[8][i][m] = f[6][i][m] + rhon * u0 / 6.0;
		f[7][i][m] = f[5][i][m] - rhon * u0 / 6.0;
	}

}


void rhouv()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(f, rho)
	for (int j = 0; j < m + 1; j++)
	{
		for (int i = 0; i < n + 1; i++)
		{
			double ssum = 0.0;
			for (int k = 0; k < 9; k++)
				ssum += f[k][i][j];
			rho[i][j] = ssum;
		}
	}

# pragma omp parallel for shared(f, rho)
	for (int i = 1; i < n + 1; i++)
	{
		rho[i][m] = f[0][i][m] + f[1][i][m] + f[3][i][m] + 2 * (f[2][i][m] + f[6][i][m] + f[5][i][m]);
	}

# pragma omp parallel for shared(f, cx, cy, u, v, rho)
	for (int j = 1; j < m; j++)
	{
		for (int i = 1; i < n + 1; i++)
		{
			double usum = 0.0;
			double vsum = 0.0;
			for (int k = 0; k < 9; k++)
			{
				usum += f[k][i][j] * cx[k];
				vsum += f[k][i][j] * cy[k];
			}
			u[i][j] = usum / rho[i][j];
			v[i][j] = vsum / rho[i][j];
		}
	}
}


void output()
{
	ofstream outfile;

	/// user define filename and title
	string filename = "rho.dat";
	string title = "2D-velocity";

	/// start write data into tecplot dat format file
	outfile.open(filename);
	outfile << "TITLE = \"" << title << "\"" << endl;
	outfile << "VARIABLES = \"X\", \"Y\", \"Z\", \"U\", \"V\" " << endl;
	outfile << "ZONE I = " << n << ", J = " << m << ", K = 1, F = point" << endl;
	for (int j = m; j >= 0; j--)
	{
		for (int i = 0; i < n + 1; i++)
		{
			outfile << i << "," << j << "," << 0 << "," << u[i][j] << "," << v[i][j] << endl;
		}
	}
	outfile.close();
}

int main()
{
	init();
	for (int kk = 1; kk <= mstep; kk++)
	{
		collesion();
		streaming();
		sfbound();
		rhouv();
	}
	output();
	return 0;
}