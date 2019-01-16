/* The LBM code for a natural convection
Computer code for Natural Convection in a Differentially Heated cavity */
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>

#define NUM_THREADS  8

using namespace std;

const int n = 100, m = 100;
const int mstep = 150000;

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

double u0 = 0.0;
double sumvel0 = 0.0;
double rho0 = 6.0;
double dx = 1.0;
double dy = 1.0;
double dt = 1.0;
double tw = 1.0;

double ra = 1.0E5;
double pr = 0.71;
double visco = 0.02;
double alpha = visco / pr;
double gbeta = ra * visco*alpha / (float(m*m*m));
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
	double tref = 0.5;
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
				double force = 3.0 * w[k] * gbeta * (th[i][j] - tref) * cy[k] * rho[i][j];
				if (i == 0 || i == n) force = 0.0;
				if (j == 0 || j == m) force = 0.0;
				feq[k][i][j] = rho[i][j] * w[k] * (1.0 + 3.0*t2 + 4.5*t2*t2 - 1.5*t1);
				f[k][i][j] = omega * feq[k][i][j] + (1.0 - omega)*f[k][i][j] + force;
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


void bounceb()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(f, u0)
	for (int j = 0; j < m + 1; j++)
	{
		///  west boundary
		f[1][0][j] = f[3][0][j];
		f[5][0][j] = f[7][0][j] ;
		f[8][0][j] = f[6][0][j] ;

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

		/// north boundary
		f[4][i][m] = f[2][i][m];
		f[7][i][m] = f[5][i][m];
		f[8][i][m] = f[6][i][m];
	}
}


void gbound()
{
	omp_set_num_threads(NUM_THREADS);
# pragma omp parallel for shared(g)
	/// west boundary condition, T = 1
	for (int j = 0; j < m + 1; j++)
	{
		g[1][0][j] = tw * (w[1] + w[3]) - g[3][0][j];
		g[5][0][j] = tw * (w[5] + w[7]) - g[7][0][j];
		g[8][0][j] = tw * (w[8] + w[6]) - g[6][0][j];
	}

	/// east boundary condition, T = 0
# pragma omp parallel for shared(g)
	for (int j = 0; j < m + 1; j++)
	{
		g[6][n][j] = -g[8][n][j];
		g[3][n][j] = -g[1][n][j];
		g[7][n][j] = -g[5][n][j];
	}

	/// top boundary condition, Adiabatic
# pragma omp parallel for shared(g)
	for (int i = 0; i < n + 1; i++)
	{
		g[8][i][m] = g[8][i][m - 1];
		g[7][i][m] = g[7][i][m - 1];
		g[6][i][m] = g[6][i][m - 1];
		g[5][i][m] = g[5][i][m - 1];
		g[4][i][m] = g[4][i][m - 1];
		g[3][i][m] = g[3][i][m - 1];
		g[2][i][m] = g[2][i][m - 1];
		g[1][i][m] = g[1][i][m - 1];
		g[0][i][m] = g[0][i][m - 1];
	}

	/// bottom boundary condition, Adiabatic
# pragma omp parallel for shared(g)
	for (int i = 0; i < n + 1; i++)
	{
		g[8][i][0] = g[8][i][1];
		g[7][i][0] = g[7][i][1];
		g[6][i][0] = g[6][i][1];
		g[5][i][0] = g[5][i][1];
		g[4][i][0] = g[4][i][1];
		g[3][i][0] = g[3][i][1];
		g[2][i][0] = g[2][i][1];
		g[1][i][0] = g[1][i][1];
		g[0][i][0] = g[0][i][1];
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
		bounceb();
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