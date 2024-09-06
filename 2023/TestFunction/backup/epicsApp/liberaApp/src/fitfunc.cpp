#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_multimin.h>

#include "fitfunc.h"
#include "bobyqa.h"

const unsigned int nx = 141;
const unsigned int ny =  71;

double gridx[nx];
double gridy[ny];

double *gridv1;
double *gridv2;
double *gridv3;
double *gridv4;

gsl_spline2d *splv1;
gsl_spline2d *splv2;
gsl_spline2d *splv3;
gsl_spline2d *splv4;

gsl_interp_accel *xacc;
gsl_interp_accel *yacc;

/* MEAS class */
struct MEAS
  {
    double v1meas;
    double v2meas;
    double v3meas;
    double v4meas;
  };

static double FcnPosFit(const long n, const double *x, void *data)
{
  double v1 = gsl_interp2d_eval_extrap((gsl_interp2d*)splv1, gridx, gridy, gridv1, x[0], x[1], xacc, yacc);
  double v2 = gsl_interp2d_eval_extrap((gsl_interp2d*)splv2, gridx, gridy, gridv2, x[0], x[1], xacc, yacc);
  double v3 = gsl_interp2d_eval_extrap((gsl_interp2d*)splv3, gridx, gridy, gridv3, x[0], x[1], xacc, yacc);
  double v4 = gsl_interp2d_eval_extrap((gsl_interp2d*)splv4, gridx, gridy, gridv4, x[0], x[1], xacc, yacc);

  // normalize to unity
  const double sumv = v1 + v2 + v3 + v4;
  v1 /= sumv;
  v2 /= sumv;
  v3 /= sumv;
  v4 /= sumv;

  MEAS *meas {(MEAS*)data};
  return
    pow((v1 - meas->v1meas)/meas->v1meas, 2.) +
    pow((v2 - meas->v2meas)/meas->v2meas, 2.) +
    pow((v3 - meas->v3meas)/meas->v3meas, 2.) +
    pow((v4 - meas->v4meas)/meas->v4meas, 2.);
}

class FitFunc
  {
  public:
    FitFunc();
    ~FitFunc();

    const long npar = 2;
    const long npt  = 5;
    const int nw    = (npt+5)*(npt+npar) + 3*npar*(npar+5)/2;

    const double rhobeg = 1e-5;
    const double rhoend = 1e-9;
    const long   maxfun = 1000;

    time_t tpre;

    void init(const std::string smap)
    {
      // interpolation
      for (unsigned int ix=0; ix<nx; ++ix)
        gridx[ix] = -7.0 + 0.1*(double)ix; // [m]

      for (unsigned int iy=0; iy<ny; ++iy)
        gridy[iy] = -3.5 + 0.1*(double)iy; // [m]

      gridv1 = (double*)malloc(nx * ny * sizeof(double));
      gridv2 = (double*)malloc(nx * ny * sizeof(double));
      gridv3 = (double*)malloc(nx * ny * sizeof(double));
      gridv4 = (double*)malloc(nx * ny * sizeof(double));

      const gsl_interp2d_type *T2 = gsl_interp2d_bilinear;
      splv1 = gsl_spline2d_alloc(T2, nx, ny);
      splv2 = gsl_spline2d_alloc(T2, nx, ny);
      splv3 = gsl_spline2d_alloc(T2, nx, ny);
      splv4 = gsl_spline2d_alloc(T2, nx, ny);

      xacc = gsl_interp_accel_alloc();
      yacc = gsl_interp_accel_alloc();

      FILE *fp = fopen(smap.c_str(), "r");
      if (fp == NULL)
        assert(!"No such file");

      for (unsigned int iline=0; iline<(nx*ny); ++iline)
        {
          double xtmp, ytmp, v1tmp, v2tmp, v3tmp, v4tmp;
          fscanf(fp, "%lf %lf %lf %lf %lf %lf", &xtmp, &ytmp, &v1tmp, &v2tmp, &v3tmp, &v4tmp);

          const int ix = (int)round(xtmp*10.) + 70; // x offset = -70
          const int iy = (int)round(ytmp*10.) + 35; // y offset = -35

          gridv1[iy*nx + ix] = v1tmp;
          gridv2[iy*nx + ix] = v2tmp;
          gridv3[iy*nx + ix] = v3tmp;
          gridv4[iy*nx + ix] = v4tmp;
        }
      fclose(fp);

      gsl_spline2d_init(splv1, gridx, gridy, gridv1, nx, ny);
      gsl_spline2d_init(splv2, gridx, gridy, gridv2, nx, ny);
      gsl_spline2d_init(splv3, gridx, gridy, gridv3, nx, ny);
      gsl_spline2d_init(splv4, gridx, gridy, gridv4, nx, ny);
      std::cout << "----- Init: loading " << smap << " done. -----" << std::endl;
    }

    void fit(const double &v1meas, const double &v2meas, const double &v3meas, const double &v4meas, double *fitx, double *fity)
    {
      const double vsum = v1meas + v2meas + v3meas + v4meas;
      MEAS meas;
      meas.v1meas = v1meas/vsum;
      meas.v2meas = v2meas/vsum;
      meas.v3meas = v3meas/vsum;
      meas.v4meas = v4meas/vsum;

      double par[npar];
      double parl[npar]; // lower bounds on fit parameters
      double parh[npar]; // lower bounds on fit parameters

      par[0]  =  0.0;
      parl[0] = -7.0;
      parh[0] =  7.0;

      par[1]  =  0.0;
      parl[1] = -3.5;
      parh[1] =  3.5;

      double work[nw];

      // print every 60 seconds
      long iprint = 0;
      time_t t = time(NULL);
      if (t-tpre > 5)
        {
          iprint = 1;
          tpre = t;
        }

      bobyqa(npar, npt, FcnPosFit, (void*)&meas, par, parl, parh, rhobeg, rhoend, iprint, maxfun, work);

      *fitx = par[0];
      *fity = par[1];
    }
  };

FitFunc::FitFunc()
{
  tpre = 0;
}
;

FitFunc::~FitFunc()
{}
;

/*========== called from C codes ==========*/
void *new_object()
{
  return static_cast<void*>(new FitFunc());
}

void call_init(void *the_object, const char *smap)
{
  FitFunc *obj = static_cast<FitFunc*>(the_object);
  obj->init(smap);
}

void call_fit(void *the_object, double *v1, double *v2, double *v3, double *v4, double *fitx, double *fity)
{
  FitFunc *obj = static_cast<FitFunc*>(the_object);
  return obj->fit(*v1, *v2, *v3, *v4, fitx, fity);
}

void delete_object(void *the_object)
{
  FitFunc *obj = static_cast<FitFunc*>(the_object);
  delete obj;
}
/*=========================================*/
