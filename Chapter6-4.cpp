/* The LBM code for a forced convection
Computer code for Forced Convection Through a Heated Channel */
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>

#define NUM_THREADS  8

using namespace std;

const int n = 1000, m = 50;
const int mstep = 10000;

double f[9][n + 1][m + 1];
double feq[9][n + 1][m + 1];
double rho[n + 1][m + 1];
double u[n + 1][m + 1];
double v[n + 1][m + 1];
double w[9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
double cx[9] = { 0,1,0,-1,0,1,-1,-1,1 };
double cy[9] = { 0,0,1,0,-1,1,1,-1,-1 };
double g[9][n + 1][m + 1];
double geq[9][n + 1][m + 1];
double th[n + 1][m + 1];

double u0 = 0.12;
double sumvel0 = 0.0;
double rho0 = 5.0;
double dx = 1.0;
double dy = 1.0;
double dt = 1.0;
double tw = 1.0;
double visco = 0.038;
double pr = 3.8;
double alpha = visco / pr;
double omega = 1.0 / (3.0*visco + 0.5);
double omegat = 1.0 / (3.0*alpha + 0.5);


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
			th[i][j] = 0.0;
			for (int k = 0; k <= 8; k++)
				g[k][i][j] = 0.0;
		}
	}

# pragma omp parallel for shared(u, v, u0)
	for (int j = 1; j <= m - 1; j++)
	{
		u[0][j] = u0;
		v[0][j] = 0.0;
	}
}


void collesion()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(u, v, cx, cy, rho, w, omega, feq, f)
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


void collt()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(u, v, cx, cy, w, omegat, geq, g, th)
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < m + 1; j++)
		{
			for (int k = 0; k < 9; k++)
			{
				geq[k][i][j] = th[i][j] * w[k] * (1.0 + 3.0*(u[i][j] * cx[k] + v[i][j] * cy[k]));
				g[k][i][j] = omegat * geq[k][i][j] + (1.0 - omegat)*g[k][i][j];
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
# pragma omp parallel for shared(f, u0)
	for (int j = 0; j < m + 1; j++)
	{
		/// flow in on west boundary
		double rhow = f[0][0][j] + f[2][0][j] + f[4][0][j] + 2 * (f[3][0][j] + f[6][0][j] + f[7][0][j]) / (1.0 - u0);
		f[1][0][j] = f[3][0][j] + 2*rhow*u0/3.0;
		f[5][0][j] = f[7][0][j] + rhow * u0 / 6.0;
		f[8][0][j] = f[6][0][j] + rhow * u0 / 6.0;
	}

# pragma omp parallel for shared(f)
	/// bounce back on south boundary
	for (int i = 0; i < n + 1; i++)
	{
		f[2][i][0] = f[4][i][0];
		f[5][i][0] = f[7][i][0];
		f[6][i][0] = f[8][i][0];
	}

# pragma omp parallel for shared(f)
	/// bounce back on north boundary
	for (int i = 0; i < n + 1; i++)
	{
		f[4][i][m] = f[2][i][m];
		f[7][i][m] = f[5][i][m];
		f[8][i][m] = f[6][i][m];
	}

# pragma omp parallel for shared(f)
	/// account for open boundary condition at the outlet
	for (int j = 1; j <= m; j++)
	{
		f[1][n][j] = 2 * f[1][n - 1][j] - f[1][n - 2][j];
		f[5][n][j] = 2 * f[5][n - 1][j] - f[5][n - 2][j];
		f[8][n][j] = 2 * f[8][n - 1][j] - f[8][n - 2][j];
	}
}


void gbound()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(g)
	/// west boundary condition, the temperature is given, T = 0
	for (int j = 0; j < m + 1; j++)
	{
		g[1][0][j] = -g[3][0][j];
		g[5][0][j] = -g[7][0][j];
		g[8][0][j] = -g[6][0][j];
	}

	/// east boundary condition, open
# pragma omp parallel for shared(g)
	for (int j = 0; j < m + 1; j++)
	{
		g[6][n][j] = 2 * g[6][n - 1][j] - g[6][n - 2][j];
		g[3][n][j] = 2 * g[3][n - 1][j] - g[3][n - 2][j];
		g[7][n][j] = 2 * g[7][n - 1][j] - g[7][n - 2][j];
		g[2][n][j] = 2 * g[2][n - 1][j] - g[2][n - 2][j];
		g[0][n][j] = 2 * g[0][n - 1][j] - g[0][n - 2][j];
		g[1][n][j] = 2 * g[1][n - 1][j] - g[1][n - 2][j];
		g[4][n][j] = 2 * g[4][n - 1][j] - g[4][n - 2][j];
		g[5][n][j] = 2 * g[5][n - 1][j] - g[5][n - 2][j];
		g[8][n][j] = 2 * g[8][n - 1][j] - g[8][n - 2][j];
	}

	/// top boundary condition, T = tw = 1.0
# pragma omp parallel for shared(g, tw, w)
	for (int i = 0; i < n + 1; i++)
	{
		g[8][i][m] = tw * (w[8] + w[6]) - g[6][i][m];
		g[7][i][m] = tw * (w[7] + w[5]) - g[5][i][m];
		g[4][i][m] = tw * (w[4] + w[2]) - g[2][i][m];
	}

	/// bottom boundary condition, Adiabatic
# pragma omp parallel for shared(g)
	for (int i = 0; i < n + 1; i++)
	{
		g[2][i][0] = tw * (w[2] + w[4]) - g[4][i][0];
		g[6][i][0] = tw * (w[6] + w[8]) - g[8][i][0];
		g[5][i][0] = tw * (w[5] + w[7]) - g[7][i][0];
	}
}


void tcalcu()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(th, g)
	for (int j = 1; j < m; j++)
	{
		for (int i = 1; i < n; i++)
		{
			double ssumt = 0.0;
			for (int k = 0; k < 9; k++)
			{
				ssumt += g[k][i][j];
			}
			th[i][j] = ssumt;
		}
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

# pragma omp parallel for shared(f, u0, u, v)
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

# pragma omp parallel for shared(v)
	for (int j = 1; j <= m; j++)
		v[n][j] = 0.0;
}


void output()
{
	ofstream outfile;

	/// user define filename and title
	string filename = "rho.dat";
	string title = "2D-velocity-temperature";

	/// start write data into tecplot dat format file
	outfile.open(filename);
	outfile << "TITLE = \"" << title << "\"" << endl;
	outfile << "VARIABLES = \"X\", \"Y\", \"Z\", \"U\", \"V\", \"T\"" << endl;
	outfile << "ZONE I = " << n << ", J = " << m << ", K = 1, F = point" << endl;
	for (int j = m; j >= 0; j--)
	{
		for (int i = 0; i < n + 1; i++)
		{
			outfile << i << "," << j << "," << 0 << "," << u[i][j] << "," << v[i][j] << "," << th[i][j] << endl;
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

		omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(g)
		for (int j = 0; j < m + 1; j++)
		{
			for (int i = 0; i < n + 1; i++)
			{
				double sum = 0.0;
				for (int k = 0; k < 9; k++)
				{
					sum += g[k][i][j];
				}
				th[i][j] = sum;
			}
		}

		collt();    /// collestion for scalar
		streaming();    /// streaming for scalar
		gbound();
	}
	output();
	return 0;
}