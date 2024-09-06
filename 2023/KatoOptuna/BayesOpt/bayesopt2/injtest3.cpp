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

#include <limits>

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/thread.hpp>

#include <TCanvas.h>
#include <TGraph.h>
#include <TH1.h>
#include <TH2.h>
#include <TRint.h>
#include <TStyle.h>

#include <cadef.h>

#include <bayesopt/bayesopt.hpp>
#include <bopt_state.hpp>
#include <dataset.hpp>
#include <param_loader.hpp>
#include <prob_distribution.hpp>
#include <specialtypes.hpp>

#include "tool_lib.h"
#include "onlinemean.h"

using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using boost::program_options::notify;

class TestFunction: public bayesopt::ContinuousModel
  {
  public:
    TestFunction(boost::property_tree::ptree &pt, const unsigned int &nx_, bayesopt::Parameters par_):
        ContinuousModel(nx_, par_)
    {
      nx = nx_;
      chan_putinfo.resize(nx);
      putinfo.resize(nx);
      for (unsigned int i=0; i<nx; ++i)
        {
          const std::string xname = pt.get<std::string>(boost::str(boost::format("PV_X%d.name") %i));
          const double      xmin  = pt.get<double>(     boost::str(boost::format("PV_X%d.min")  %i));
          const double      xmax  = pt.get<double>(     boost::str(boost::format("PV_X%d.max")  %i));
          const double      xstep = pt.get<double>(     boost::str(boost::format("PV_X%d.step") %i));

          SEVCHK(ca_create_channel(xname.c_str(), NULL, NULL, 0, &chan_putinfo[i]), "Create channel failed");
          putinfo[i] = (dbr_double_t*)calloc(ca_element_count(chan_putinfo[i]), sizeof(*putinfo[i]));

          vxmin.push_back(xmin);
          vxmax.push_back(xmax);
          vxstep.push_back(xstep);
        }

      ny = pt.get<unsigned int>("PV.ny");
      for (unsigned int i=0; i<ny; ++i)
        {
          const std::string yname = pt.get<std::string>(boost::str(boost::format("PV_Y%d.name")   %i));
          const double      ywgt  = pt.get<double>(     boost::str(boost::format("PV_Y%d.weight") %i));

          vyname.push_back(yname);
          vywgt.push_back(ywgt);
        }

      usekalman = pt.get<unsigned int>("PV.usekalman");
      stepsleep = pt.get<unsigned int>("PV.stepsleep");
      evalsleep = pt.get<unsigned int>("PV.evalsleep");

      n_iterations = par_.n_iterations;

      OnlineMean_Init(&oMean);

      ofsExpect.open("Expect.dat");
      ofsObtain.open("Obtain.dat");

      pHandler = new EventHandler();

      /* ROOT */
      h10 = new TH1D("h10", "h10", 100, 0, 1);
      h11 = new TH1D("h11", "h11", 100, 0, 1);
      c1 = new TCanvas("c1", "c1", 600, 600);
      c1->Divide(1, 2);

      h20 = new TH2D("Mean",         "Mean",         100, 0, 1, 100, 0, 1);
      h21 = new TH2D("2#sigma std.", "2#sigma std.", 100, 0, 1, 100, 0, 1);
      gStyle->SetPalette(kBird);
      gStyle->SetNumberContours(256);
      c0 = new TCanvas("c0", "c0", 1000, 500);
      c0->Divide(2, 1);

      g2 = new TGraph(nx);
      g2->SetMarkerStyle(24);
      g2->SetMarkerColor(6);
    }

    ~TestFunction()
    {
      for (unsigned int i=0; i<putinfo.size(); ++i)
        free(putinfo[i]);

      delete pHandler;
    }

    void run()
    {
      initializeOptimization();

      for (size_t i = 0; i < n_iterations; ++i)
        {
          stepOptimization();

          ofsExpect.close();
          ofsExpect.open("Expect.dat", std::ios::trunc);

          h10->Reset();
          h11->Reset();
          h20->Reset();
          h21->Reset();

          if (nx == 1)
            {
              vectord q(1);
              for (unsigned int ix0 = 0; ix0 <= nxmap; ++ix0)
                {
                  const double x0 = (double)ix0 * dxmap;

                  q(0) = x0;
                  bayesopt::ProbabilityDistribution *db = this->getPrediction(q);
                  const double y = db->getMean();
                  const double su = y + 2. * db->getStd();
                  const double sl = y - 2. * db->getStd();

                  ofsExpect << boost::format("%10.3e, %10.3e, %10.3e, %10.3e") %x0 %y %su %sl << std::endl;

                  h10->Fill(x0 + 0.5 * dxmap, y);
                }
              c1->cd(1);
              h10->Draw("C hist");
              c1->cd(2);
              h11->Draw("C hist");
              c1->Update();
            }
          else if (nx == 2)
            {
              vectord q(2);
              for (unsigned int ix0 = 0; ix0 <= nxmap; ++ix0)
                for (unsigned int ix1 = 0; ix1 <= nxmap; ++ix1)
                  {
                    const double x0 = (double) ix0 * dxmap;
                    const double x1 = (double) ix1 * dxmap;

                    q(0) = x0;
                    q(1) = x1;
                    bayesopt::ProbabilityDistribution *db = this->getPrediction(q);
                    const double y = db->getMean();
                    const double su = y + 2. * db->getStd();
                    const double sl = y - 2. * db->getStd();

                    ofsExpect << boost::format("%10.3e, %10.3e, %10.3e, %10.3e, %10.3e") %x0 %x1 %y %su %sl << std::endl;

                    h20->Fill(x0 + 0.5 * dxmap, x1 + 0.5 * dxmap, y);
                    h21->Fill(x0 + 0.5 * dxmap, x1 + 0.5 * dxmap, 2.*db->getStd());
                  }

              c0->cd(1);
              h20->Draw("colz");
              g2->Draw("P, same");
              c0->cd(2);
              h21->Draw("colz");
              h21->SetMinimum(0);
              c0->Update();

              c1->cd(1);
              h10 = (TH1D*)h20->ProjectionX();
              h10->SetTitle("X0");
              h10->Draw("C hist");
              c1->cd(2);
              h11 = (TH1D*)h20->ProjectionY();
              h11->SetTitle("X1");
              h11->Draw("C hist");
              c1->Update();
            }
          else
            {
              std::cout << "nx==3 not yet implemented." << std::endl;
            }
        }
    }

    void array_put(const std::vector<double> &vx)
    {
      // Get the current PV values
      std::vector<double> vxold;
      for (unsigned int i=0; i<vx.size(); ++i)
        {
          /* Read data through EPICS PV */
          SEVCHK(ca_array_get(DBR_DOUBLE, ca_element_count(chan_putinfo[i]), chan_putinfo[i], putinfo[i]), "array write request failed");
          SEVCHK(ca_pend_io(5), "ca_pend_io failure");

          if ((putinfo[i])[0]==0.)
            vxold.push_back((vxmax[i]+vxmin[i])/2.);
          else
            vxold.push_back((putinfo[i])[0]);
        }

      // Put new PV values step by step
      for (unsigned int i=0; i<vx.size(); ++i)
        {
          const int nstep0   = int(fabs(vx[i] - vxold[i])/vxstep[i]);
          const int nstep1   = nstep0 + 1; // plus one more step for safety
          const double dx    = vx[i] - vxold[i];
          const double dstep = dx/(double)nstep1;

          for (int istep=1; istep<=nstep1; ++istep)
            {
              (putinfo[i])[0] = dstep * (double)istep + vxold[i];

              /* Write data through EPICS PV */
              SEVCHK(ca_array_put(DBR_DOUBLE, ca_element_count(chan_putinfo[i]), chan_putinfo[i], putinfo[i]), "array write request failed");
              SEVCHK(ca_flush_io(), "array write failed");

              boost::this_thread::sleep(boost::posix_time::milliseconds(stepsleep));
            }
        }
    }

    void Update(const double &ysum)
    {
      // Update the online mean
      OnlineMean_Update(&oMean, (float)ysum);
    }

    void PVMonitor()
    {
      /* Start up Channel Access */
      int result = ca_context_create(ca_disable_preemptive_callback);
      if (result != ECA_NORMAL)
        {
          fprintf(stderr, "CA error %s occurred while trying to start channel access.\n", ca_message(result));
          assert(!"Error.");
        }

      /*--- Allocate PV structure array (camonitor) ---*/
      pv *pvsmon = (pv*)calloc(ny, sizeof(pv));
      if (!pvsmon)
        assert(!"Memory allocation for channel structures failed.");

      // Connect channels
      for (unsigned int i=0; i<ny; ++i)
        {
          pvsmon[i].name   = new char[vyname[i].size() + 1];
          std::strcpy(pvsmon[i].name, vyname[i].c_str());

          pvsmon[i].ywgt   = vywgt[i];
          pvsmon[i].Update = std::bind(&TestFunction::Update, this, std::placeholders::_1);
        }

      /*--- Create CA connections ---*/
      if (pHandler->create_pvs(pvsmon, ny, pHandler->pvget_connection_handler))
        {
          std::cout << "Return from create_pvs." << std::endl;
          abort();
        }

      /*--- Check for channels that didn't connect ---*/
      ca_pend_event(DEFAULT_TIMEOUT);
      for (unsigned int m = 0; m < ny; ++m)
        if (!pvsmon[m].onceConnected)
          pHandler->print_time_val_sts(&pvsmon[m], reqElems);

      /* Read and print data forever */
      ca_pend_event(0);

      /* Shut down Channel Access */
      ca_context_destroy();

      for (unsigned int i=0; i<ny; ++i)
        free(&pvsmon[i]);
    }

    double evaluateSample(const vectord& xin)
    {
      const unsigned int nx = xin.size();

      std::vector<double> vx(nx);
      for (unsigned int i=0; i<nx; ++i)
        vx[i] = (double)(xin(i) * (vxmax[i] - vxmin[i]) + vxmin[i]);
      array_put(vx);

      std::cout << "X:";
      for (unsigned int i=0; i<nx-1; ++i)
        std::cout << boost::format(" %10.5f") %vx[i];
      std::cout << boost::format(" %10.5f") %vx[nx-1] << std::endl;

      OnlineMean_Reset(&oMean);
      boost::this_thread::sleep(boost::posix_time::milliseconds(evalsleep));

      double ysum = oMean.mean;
      std::cout << ysum << std::endl;

      for (unsigned int i=0; i<nx-1; ++i)
        ofsObtain << boost::format("%10.3e") %vx[i];
      ofsObtain << boost::format(" %10.3e %10.3e") %vx[nx-1] %ysum << std::endl;

      g2->SetPoint(mCurrentIter, xin(0), xin(1));

      return ysum;
    }

    void printBestpoints(const vectord& xin)
    {
      std::cout << "Best values: " << this->getValueAtMinimum() << std::endl;
      std::cout << "Best points:" << std::endl;
      for (unsigned int i=0; i<xin.size(); ++i)
        std::cout << boost::format("X%d %10.5f") %i %(xin(i) * (vxmax[i] - vxmin[i]) + vxmin[i]) << std::endl;
    }

    bool checkReachability(const vectord &query)
    {
      return true;
    };

  private:
    std::vector<chid>     chan_putinfo;
    std::vector<dbr_double_t*> putinfo;

    unsigned int nx, ny, usekalman, n_iterations, stepsleep, evalsleep;

    std::vector<double>      vxmin, vxmax, vxstep, vywgt;
    std::vector<std::string> vyname;

    EventHandler *pHandler;

    const unsigned int nxmap = 100;
    const double       dxmap = 1./(double)nxmap;

    std::ofstream ofsObtain;
    std::ofstream ofsExpect;

    TH1D *h10;
    TH1D *h11;
    TH2D *h20;
    TH2D *h21;

    TGraph *g2;

    TCanvas *c0;
    TCanvas *c1;

    OnlineMean oMean;
  };


int main(int argc, char *argv[])
{
  /* ROOT initialization */
  int tmp = 1;
  TApplication theApp ("App", &tmp, argv);
  gStyle->SetOptStat (0);

  /* Command line option */
  options_description opt("Option");
  // define format
  opt.add_options()
  ("help,h", "display help")
  ("input,i", value<std::string>(), "ini file");

  // analyze command line
  variables_map argmap;
  store(parse_command_line(argc, argv, opt), argmap);
  notify(argmap);

  // if no matching option, show help
  if (argmap.count("help") || !argmap.count("input"))
    {
      std::cerr << opt << std::endl;
      return 1;
    }

  /* Configuration */
  const std::string input = argmap["input"].as<std::string>();

  boost::property_tree::ptree pt;
  try
    {
      boost::property_tree::read_ini(input.c_str(), pt);
    }
  catch(boost::property_tree::ptree_error& e)
    {
      std::cout << "ptree_error " << e.what() << std::endl;
      exit(-1);
    }

  const unsigned int nx = pt.get<unsigned int>("PV.nx");

  bayesopt::Parameters par;
  par = initialize_parameters_to_default();
  par.n_iterations  = pt.get<size_t>("PAR.n_iterations");
  par.random_seed   = pt.get<int>("PAR.random_seed")==0 ? (int)time(0) : pt.get<int>("PAR.random_seed");
  par.verbose_level = pt.get<int>("PAR.verbose_level");
  par.noise         = pt.get<double>("PAR.noise");
  par.crit_name     = pt.get<std::string>("PAR.crit_name");

  // Define format
  auto format = new boost::posix_time::time_facet("%Y%m%d_%H.%M.%S");
  std::stringstream ss;
  ss.imbue(std::locale(std::cout.getloc(), format));

  // fetch the current time
  auto now = boost::posix_time::second_clock::local_time();
  ss << now;

  bayesopt::utils::ParamLoader::save(ss.str()+".txt", par);

  TestFunction *myfunc = new TestFunction(pt, nx, par);
  boost::thread th1(&TestFunction::PVMonitor, myfunc);

  myfunc->run();
  vectord result = myfunc->getFinalResult();

  bayesopt::BOptState state;
  myfunc->saveOptimization(state);
  state.saveToFile(ss.str()+".log");
  std::cout << "STATE ITERS: " << state.mCurrentIter << std::endl;

  myfunc->printBestpoints(result);

  theApp.Run();

  return 0;
}
