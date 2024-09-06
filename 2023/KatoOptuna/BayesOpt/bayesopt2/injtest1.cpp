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

#include <fstream>
#include <vector>
#include <string>

#include <TGraph.h>
#include <TGraph2D.h>
#include <TH2D.h>
#include <TF2.h>
#include <TFile.h>

#include "param_loader.hpp"
#include "bopt_state.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/array.hpp>
#include <boost/filesystem.hpp>
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
        ContinuousModel(2, par)
    {
      gen.seed(par.random_seed);
      dist.param(boost::normal_distribution<double>::param_type(0, par.noise));

      dx   = abs(xmax-xmin);
      dy   = abs(ymax-ymin);

      sigx = dx/70.;
      sigy = dy/70.;

      h2ang_ypos0 = new TH2D("h2ang_ypos0", "h2ang_ypos0", nbin, xmin, xmax, nbin, ymin, ymax);
      h2ang_ypos1 = new TH2D("h2ang_ypos1", "h2ang_ypos1", nbin, xmin, xmax, nbin, ymin, ymax);
      h2ang_ypos2 = new TH2D("h2ang_ypos2", "h2ang_ypos2", nbin, xmin, xmax, nbin, ymin, ymax);
    }

    ~BraninNormalized()
    {
      delete h2ang_ypos0;
      delete h2ang_ypos1;
      delete h2ang_ypos2;
    }

    double evaluateSample( const vectord& xin)
    {
      const double x = xin(0) * dx + xmin;
      const double y = xin(1) * dy + ymin;

      return -h2ang_ypos2->Interpolate(x, y);
    }

    bool checkReachability(const vectord &query)
    {
      return true;
    };

    bool makemap(const std::string &file)
    {
      //      const std::string arglist = boost::str(boost::format("awk \'{print$1\"\t\"$2\"\t\"$3}\' %s | uniq > %s.uniq") %file %file);
      //      system(arglist.c_str());

      h2ang_ypos0->Sumw2();
      h2ang_ypos1->Sumw2();
      h2ang_ypos2->Sumw2();

      std::ifstream ifs(file);
      while (!ifs.eof())
        {
          double ang, ypos, yang, eff;
          ifs >> ang>>ypos>>yang>>eff;

          h2ang_ypos0->Fill(ang, ypos * 1e+6);
          h2ang_ypos1->Fill(ang, ypos * 1e+6, eff);
        }

      h2ang_ypos1->Divide(h2ang_ypos0);

      TF2 *fgauss2D = new TF2("fgauss2D", "[0]*TMath::Gaus(x, [1], [2]) * TMath::Gaus(y, [3], [4])", xmin, xmax, ymin, ymax);

      for (unsigned int ix=0; ix<nbin; ++ix)
        for (unsigned int iy=0; iy<nbin; ++iy)
          {
            const double x = h2ang_ypos1->GetXaxis()->GetBinCenter(ix+1);
            const double y = h2ang_ypos1->GetYaxis()->GetBinCenter(iy+1);
            const double z = h2ang_ypos1->GetBinContent(ix+1, iy+1);

            fgauss2D->SetParameters(z, x, sigx, y, sigy);

            for (unsigned int ix1=0; ix1<nbin; ++ix1)
              for (unsigned int iy1=0; iy1<nbin; ++iy1)
                {
                  const double x1 = h2ang_ypos1->GetXaxis()->GetBinCenter(ix1+1);
                  const double y1 = h2ang_ypos1->GetYaxis()->GetBinCenter(iy1+1);
                  const double z1 = fgauss2D->Eval(x1, y1);
                  h2ang_ypos2->Fill(x1, y1, z1);
                }
          }

      delete fgauss2D;

      return true;
    };

    const double xmin =    2.73;
    const double xmax =    2.78;
    const double ymin = -265;
    const double ymax = -230;

    double dx, dy, sigx, sigy;

    TH2D *h2ang_ypos2;

  private:
    const unsigned int nbin = 100;

    TH2D *h2ang_ypos0;
    TH2D *h2ang_ypos1;
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

      //par.crit_name = "cEI";  //**
      par.crit_name = "cLCB"; //***
      //par.crit_name = "cLCBa"; //*
      //par.crit_name = "cExpReturn"; //**

      bayesopt::utils::ParamLoader::save("bo_branin.txt", par);
    }

  BraninNormalized branin(par);
  vectord result(2);

  branin.makemap("aho");

  branin.optimize(result);
  std::cout << "Result: " << result << "->" << branin.evaluateSample(result) << std::endl;

  bayesopt::BOptState state;
  branin.saveOptimization(state);
  state.saveToFile("injtest1.log");
  std::cout << "STATE ITERS: " << state.mCurrentIter << std::endl;

  TGraph   *gresult1D = new TGraph(  par.n_iterations);
  gresult1D->SetName("gresult1D");
  gresult1D->SetMarkerColor(2);
  gresult1D->SetMarkerStyle(24);

  TGraph2D *gresult2D = new TGraph2D(par.n_iterations);
  gresult2D->SetName("gresult2D");
  gresult2D->SetMarkerColor(2);
  gresult2D->SetMarkerStyle(24);

  std::ofstream ofs("injtest1.dat");
  for (size_t ii = 0; ii < par.n_iterations; ++ii)
    {
      const double x = state.mX[ii](0)*branin.dx+branin.xmin;
      const double y = state.mX[ii](1)*branin.dy+branin.ymin;
      const double z = state.mY[ii];
      gresult1D->SetPoint(ii, x, y);
      gresult2D->SetPoint(ii, x, y, -z);
      ofs << boost::format("%10.3e %10.3e %10.3e") %x %y %z << std::endl;
    }

  TFile f("test.root", "recreate");
  branin.h2ang_ypos2->Write();
  gresult1D->Write();
  gresult2D->Write();
  f.Close();

  return 0;
}
