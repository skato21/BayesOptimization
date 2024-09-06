#include <stdio.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

double alpha[] = {1.0, 1.2, 3.0, 3.2};
double A[] = {10, 3, 17, 3.5, 1.7, 8,
	      0.05, 10, 17, 0.1, 8, 14,
	      3, 3.5, 1.7, 10, 17, 8,
	      17, 8, 0.05, 10, 0.1, 14};
double P[] = {1312, 1696, 5569, 124, 8283, 5886,
	      2329, 4135, 8307, 3736, 1004, 9991,
	      2348, 1451, 3522, 2883, 3047, 6650,
	      4047, 8828, 8732, 5743, 1091, 381};

double hartmann6d(const double x0, const double x1, const double x2,
		  const double x3, const double x4, const double x5)
{
  const double x[6] = {x0, x1, x2, x3, x4, x5};

  gsl_matrix_view MA = gsl_matrix_view_array(A, 4, 6);
  gsl_matrix_view MP = gsl_matrix_view_array(P, 4, 6);

  double tmpi = 0.;
  for (unsigned int i=0; i<4; ++i)
    {
      double tmpj = 0.;
      for (unsigned int j=0; j<6; ++j)
	{
	  tmpj += (gsl_matrix_get(&MA.matrix, i, j) * pow(x[j] - 1.e-4 * gsl_matrix_get(&MP.matrix, i, j), 2.));
	}
      
      tmpi += (alpha[i] * exp(-tmpj));
    }

  return -tmpi;
}
