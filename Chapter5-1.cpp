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

double u0 = 0.1;	/// u������ӳ�ʼ�ٶ�
double rho0 = 5.0;
const double dx = 1.0;
const double dy = 1.0;
const double dt = 1.0;
double alpha = 0.01;
double omega = 1.0 / (3.0*alpha + 0.5);
const int mstep = 40000;

void init()
{
	omp_set_num_threads(NUM_THREADS);
	# pragma omp parallel for shared(rho, n, m, rho0)
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
		u[i][m] = u0;
		v[i][m] = 0.0;
	}
}

void collesion()
{
	omp_set_num_threads(NUM_THREADS);
	# pragma omp parallel for shared(u, v, cx, cy, rho, w, omega, feq, f, n, m)
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

void streaming()
{
	omp_set_num_threads(NUM_THREADS);
	# pragma omp parallel for shared(f, n, m)
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

	# pragma omp parallel for shared(f, n, m)
	/// top to bottom
	for (int j = m; j > 0; j--)
	{
		for (int i = 0; i < n+1; i++)
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

	# pragma omp parallel for shared(f, n, m)
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
	# pragma omp parallel for shared(f, m, u0)
	
	for (int j = 0; j < m + 1; j++)
	{
		/// bounce back on west boundary
		f[1][0][j] = f[3][0][j];
		f[5][0][j] = f[7][0][j];
		f[8][0][j] = f[6][0][j];

		/// bounce back on east boundary
		f[3][n][j] = f[1][n][j];
		f[7][n][j] = f[5][n][j];
		f[6][n][j] = f[8][n][j];
	}
	
	# pragma omp parallel for shared(f, n)
	/// bounce back on south boundary
	for (int i = 0; i < n + 1; i++)
	{
		f[2][i][0] = f[4][i][0];
		f[5][i][0] = f[7][i][0];
		f[6][i][0] = f[8][i][0];
	}

	# pragma omp parallel for shared(f, m, n)
	/// moving lid, north boundary
	for (int i = 1; i < n; i++)
	{
		double rhow = f[0][i][m] + f[1][i][m] + f[3][i][m] + 2.0 * (f[2][i][m] + f[6][i][m] + f[5][i][m]);
		f[4][i][m] = f[2][i][m];
		f[7][i][m] = f[5][i][m] - rhow * u0 / 6.0;
		f[8][i][m] = f[6][i][m] + rhow * u0 / 6.0;
	}

}

void rhouv()
{
	omp_set_num_threads(NUM_THREADS);
	# pragma omp parallel for shared(f, m, n, rho)
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

	# pragma omp parallel for shared(f, m, n, rho)
	for (int i = 1; i < n + 1; i++)
		rho[i][m] = f[0][i][m] + f[1][i][m] + f[3][i][m] + 2.0 * (f[2][i][m] + f[6][i][m] + f[5][i][m]);

	# pragma omp parallel for shared(f, m, u0)
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
	outfile << "VARIABLES = \"X\", \"Y\", \"Z\", \"U\", \"V\"" << endl;
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
	for (int kk = 0; kk < mstep; kk++)
	{
		collesion();
		streaming();
		sfbound();
		rhouv();
	}
	output();
	return 0;
}