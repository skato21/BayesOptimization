/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
   BayesOpt is free software: you can redistribute it and/or modify it 
   under the terms of the GNU Affero General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/

#include <cadef.h>
#include <limits>

#include "param_loader.hpp"
#include "bopt_state.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include "bayesopt/bayesopt.hpp"
#include "specialtypes.hpp"


const unsigned int npar = 1;
const unsigned int npause = 5;

class TestFunction: public bayesopt::ContinuousModel
  {
  public:
    TestFunction(bayesopt::Parameters par):
        ContinuousModel(npar, par)
    {
      chan_getinfo = NULL;
      chan_putinfo = NULL;

      create_channel();

      getinfo = (dbr_float_t*)calloc(ca_element_count(chan_getinfo), sizeof(*getinfo));
      putinfo = (dbr_float_t*)calloc(ca_element_count(chan_putinfo), sizeof(*putinfo));
    }

    ~TestFunction()
    {
      free(getinfo);
      free(putinfo);
    }

    void create_channel()
    {
      /*--- BMHXRM:VEC:HPROF_FITINFO ---*/
      SEVCHK(ca_create_channel("BMHSRM:BEAM:SIGMAX", NULL, NULL, 0, &chan_getinfo), "Create channel failed");
      SEVCHK(ca_pend_io(5.0), "Search failed");

      /*--- BMHXRM:VEC:HPROF_DATAFIT ---*/
      SEVCHK(ca_create_channel("BMHSRM:VEC:HINTERF_DATAFIT", NULL, NULL, 0, &chan_putinfo), "Create channel failed");
      SEVCHK(ca_pend_io(5.0), "Search failed");
    }

    void array_put(const float &x)
    {
      putinfo[0] = x;

      /* Write data through EPICS PV */
      SEVCHK(ca_array_put(DBR_FLOAT, ca_element_count(chan_putinfo), chan_putinfo, putinfo), "array write request failed");
      SEVCHK(ca_flush_io(), "array write failed");

      free(putinfo);
    }

    double array_get()
    {
      /* Read data through EPICS PV */
      SEVCHK(ca_array_get(DBR_FLOAT, ca_element_count(chan_getinfo), chan_getinfo, getinfo), "array read request failed");
      SEVCHK(ca_flush_io(), "array read failed");

      return (double)getinfo[0];
    }

    double evaluateSample(const vectord& xin)
    {
      const double x = xin(0) * 8.-4.;
      array_put((float)x);

      std::cout << "Press Enter to Continue";
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

      double y = 0;
      for (unsigned int i=0; i<npause; ++i)
        {
          y += array_get();
          sleep(1);
        }

      return y/(double)npause;
    }

    bool checkReachability(const vectord &query)
    {
      return true;
    };

  private:
    chid chan_getinfo; // get
    chid chan_putinfo; // put
    dbr_float_t *getinfo;
    dbr_float_t *putinfo;
  };


int main(int nargs, char *args[])
{
  bayesopt::Parameters par;
  if(nargs > 1)
    {
      if(!bayesopt::utils::ParamLoader::load(args[1], par))
        {
          std::cout << "ERROR: provided file \"" << args[1] << "\" does not exist" << std::endl;
          return -1;
        }
    }
  else
    {
      par = initialize_parameters_to_default();
      par.n_iterations = 200;
      par.random_seed = 0;
      par.verbose_level = 1;
      par.noise = 1e-1;

      par.crit_name = "cLCB";

      bayesopt::utils::ParamLoader::save("injtest2.txt", par);
    }

  TestFunction myfunc(par);
  vectord result(npar);

  myfunc.optimize(result);

  bayesopt::BOptState state;
  myfunc.saveOptimization(state);
  state.saveToFile("injtest2.log");
  std::cout << "STATE ITERS: " << state.mCurrentIter << std::endl;

  return 0;
}
