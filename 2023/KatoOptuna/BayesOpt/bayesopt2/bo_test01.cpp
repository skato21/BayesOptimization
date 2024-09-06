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

//#include "testfunctions.hpp"
#include "param_loader.hpp"
#include "bopt_state.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include "bayesopt/bayesopt.hpp"
#include "specialtypes.hpp"


boost::random::mt19937 gen;
boost::normal_distribution<double> dist;


class BraninNormalized: public bayesopt::ContinuousModel
  {
  public:
    BraninNormalized(bayesopt::Parameters par):
        ContinuousModel(1, par)
    {
      gen.seed(par.random_seed);
      dist.param(boost::normal_distribution<double>::param_type(0, par.noise));
    }

    double evaluateSample( const vectord& xin)
    {
      //double x = xin(0) * 10.;
      //return cos(2.3 * x) + 0.5 * x + dist(gen);

      //double x = xin(0) * 2.;
      //return pow(x-1.23, 2.) + dist(gen);

      double x = xin(0) * 8.-4.;
      return cos(x) + 0.01*x + dist(gen);
    }

    bool checkReachability(const vectord &query)
    {
      return true;
    };
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
      par.verbose_level = 2;
      par.noise = 1e-1;

      //par.kernel.name = "kMaternARD1";
      //par.kernel.hp_mean <<= 1.0;
      //par.kernel.hp_std  <<= 1.0;

      par.crit_name = "cLCB";

      bayesopt::utils::ParamLoader::save("bo_branin.txt", par);
    }

  BraninNormalized branin(par);
  vectord result(1);

  branin.optimize(result);
  std::cout << "Result: " << result << "->"
  << branin.evaluateSample(result) << std::endl;

  //  branin.initializeOptimization();
  //
  //  for (size_t ii = 0; ii < par.n_iterations; ++ii)
  //    {
  //      branin.stepOptimization();
  //    }

  bayesopt::BOptState state;
  branin.saveOptimization(state);
  state.saveToFile("state.dat");
  std::cout << "STATE ITERS: " << state.mCurrentIter << std::endl;

  for (size_t ii = 0; ii < par.n_iterations; ++ii)
    {
      std::cout << state.mX[ii](0) << " " << state.mY[ii] << std::endl;
    }

  return 0;
}
