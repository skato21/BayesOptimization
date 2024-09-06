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
#include <utility>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/histogram.hpp>
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

using boost::program_options::notify;
using boost::program_options::options_description;
using boost::program_options::parse_command_line;
using boost::program_options::store;
using boost::program_options::value;
using boost::program_options::variables_map;

using reg = boost::histogram::axis::any_std;

struct HolderOfStaticHistogram
{
  // put axis types here
  using hist_t = boost::histogram::histogram<std::vector<reg>>;
  hist_t hist_;
};

class TestFunction : public bayesopt::ContinuousModel
{
public:
  TestFunction(boost::property_tree::ptree &pt, const unsigned int &nx_, bayesopt::Parameters par_) : ContinuousModel(nx_, par_)
  {
    nx = nx_;
    chan_putinfo.resize(nx);
    putinfo.resize(nx);

    std::vector<reg> axes;

    for (unsigned int i = 0; i < nx; ++i)
    {
      const std::string xname = pt.get<std::string>(boost::str(boost::format("PV_X%d.name") % i));
      const double xrmin = pt.get<double>(boost::str(boost::format("PV_X%d.rmin") % i));
      const double xrmax = pt.get<double>(boost::str(boost::format("PV_X%d.rmax") % i));
      const double xstep = pt.get<double>(boost::str(boost::format("PV_X%d.step") % i));

      SEVCHK(ca_create_channel(xname.c_str(), NULL, NULL, 0, &chan_putinfo[i]), "Create channel failed");
      putinfo[i] = (dbr_double_t *)calloc(ca_element_count(chan_putinfo[i]), sizeof(*putinfo[i]));

      vxrmin.push_back(xrmin);
      vxrmax.push_back(xrmax);
      vxstep.push_back(xstep);

      axes.emplace_back(boost::histogram::axis::regular<>(nxmap, 0, 1, Form("x%d", nx)));
    }

    hmap.hist_ = boost::histogram::make_dynamic_histogram(axes.begin(), axes.end());

    ny = pt.get<unsigned int>("PV.ny");
    for (unsigned int i = 0; i < ny; ++i)
    {
      const std::string yname = pt.get<std::string>(boost::str(boost::format("PV_Y%d.name") % i));
      const double ywght = pt.get<double>(boost::str(boost::format("PV_Y%d.weight") % i));

      vyname.push_back(yname);
      vywght.push_back(ywght);
    }

    stepsleep = pt.get<unsigned int>("PV.stepsleep");
    evalsleep = pt.get<unsigned int>("PV.evalsleep");
    n_iterations = par_.n_iterations;

    pHandler = new EventHandler();

    OnlineMean_Init(&oMean);

    /* ROOT */
    gStyle->SetPalette(kBird);
    gStyle->SetNumberContours(256);
    c2 = new TCanvas("c2", "c2", 800, 800);
    c2->Divide(nx - 1, nx - 1);

    for (unsigned int ix0 = 0; ix0 < nx - 1; ++ix0)
      for (unsigned int ix1 = ix0 + 1; ix1 < nx; ++ix1)
      {
        h2[ix0][ix1] = new TH2D(Form("h2_%d_%d", ix0, ix1), Form("h2_%d_%d", ix0, ix1), nxmap, 0, 1, nxmap, 0, 1);

        g2[ix0][ix1] = new TGraph(n_iterations);
        g2[ix0][ix1]->SetMarkerStyle(24);
        g2[ix0][ix1]->SetMarkerColor(6);
      }

    c1 = new TCanvas("c1", "c1", 800, 500);

    for (unsigned int ix0 = 0; ix0 < nx; ++ix0)
    {
      g1[ix0] = new TGraph(n_iterations);
      g1[ix0]->SetTitle("");
      g1[ix0]->SetMarkerStyle(20);
      g1[ix0]->SetMarkerColor(ix0 + 1);
      g1[ix0]->SetLineColor(ix0 + 1);
    }
  }

  ~TestFunction()
  {
    for (unsigned int i = 0; i < putinfo.size(); ++i)
      free(putinfo[i]);

    delete pHandler;
  }

  void run()
  {
    initializeOptimization();

    for (size_t i = 0; i < n_iterations; ++i)
    {
      stepOptimization();

      for (unsigned int ix0 = 0; ix0 < nx - 1; ++ix0)
        for (unsigned int ix1 = ix0 + 1; ix1 < nx; ++ix1)
          h2[ix0][ix1]->Reset();

      vectord q(nx);

      for (auto it = hmap.hist_.begin(), end = hmap.hist_.end(); it != end; ++it)
      {
        for (unsigned int ix = 0; ix < nx; ++ix)
          q(ix) = double(it.idx(ix)) * dxmap;

        const double mean = this->getPrediction(q)->getMean();
        *it = mean;

        // projection to 2D
        for (unsigned int ix0 = 0; ix0 < nx - 1; ++ix0)
          for (unsigned int ix1 = ix0 + 1; ix1 < nx; ++ix1)
            h2[ix0][ix1]->Fill(q(ix0), q(ix1), mean);
      }

      for (unsigned int ix0 = 0; ix0 < nx - 1; ++ix0)
        for (unsigned int ix1 = ix0 + 1; ix1 < nx; ++ix1)
        {
          c2->cd(ix0 * (nx - 1) + ix1);
          h2[ix0][ix1]->Draw("colz");
          g2[ix0][ix1]->Draw("P,same");
        }
      c2->Update();

      c1->cd();
      g1[0]->Draw("ALP");
      g1[0]->GetYaxis()->SetRangeUser(-0.05, 1.05);
      for (unsigned int ix0 = 1; ix0 < nx; ++ix0)
        g1[ix0]->Draw("LP,same");
      c1->Update();
    }
  }

  void array_put(const std::vector<double> &vx)
  {
    // Get the current PV values
    std::vector<double> vxold;
    for (unsigned int i = 0; i < vx.size(); ++i)
    {
      /* Read data through EPICS PV */
      SEVCHK(ca_array_get(DBR_DOUBLE, ca_element_count(chan_putinfo[i]), chan_putinfo[i], putinfo[i]), "array write request failed");
      SEVCHK(ca_pend_io(5), "ca_pend_io failure");

      if ((putinfo[i])[0] == 0.)
        vxold.push_back((vxrmax[i] + vxrmin[i]) / 2.);
      else
        vxold.push_back((putinfo[i])[0]);
    }

    // Put new PV values step by step
    for (unsigned int i = 0; i < vx.size(); ++i)
    {
      const double dx = vx[i] - vxold[i];
      const int nstep = int(fabs(dx) / vxstep[i]) + 1; // plus one more step for safety
      const double dstep = dx / (double)nstep;

      for (int istep = 1; istep <= nstep; ++istep)
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
    pv *pvmon = (pv *)calloc(ny, sizeof(pv));
    if (!pvmon)
      assert(!"Memory allocation for channel structures failed.");

    // Connect channels
    for (unsigned int i = 0; i < ny; ++i)
    {
      pvmon[i].name = new char[vyname[i].size() + 1];
      std::strcpy(pvmon[i].name, vyname[i].c_str());

      pvmon[i].ywght = vywght[i];
      pvmon[i].Update = std::bind(&TestFunction::Update, this, std::placeholders::_1);
    }

    /*--- Create CA connections ---*/
    if (pHandler->create_pvs(pvmon, ny, pHandler->pvget_connection_handler))
    {
      std::cout << "Return from create_pvs." << std::endl;
      abort();
    }

    /*--- Check for channels that didn't connect ---*/
    ca_pend_event(DEFAULT_TIMEOUT);
    for (unsigned int m = 0; m < ny; ++m)
      if (!pvmon[m].onceConnected)
        pHandler->print_time_val_sts(&pvmon[m], reqElems);

    /* Read and print data forever */
    ca_pend_event(0);

    /* Shut down Channel Access */
    ca_context_destroy();

    for (unsigned int i = 0; i < ny; ++i)
      free(&pvmon[i]);
  }

  double evaluateSample(const vectord &xin)
  {
    std::vector<double> vx(nx);
    for (unsigned int i = 0; i < nx; ++i)
      vx[i] = (double)(xin(i) * (vxrmax[i] - vxrmin[i]) + vxrmin[i]);
    array_put(vx);

    std::cout << "X:";
    for (unsigned int i = 0; i < nx - 1; ++i)
      std::cout << boost::format(" %10.5f") % vx[i];
    std::cout << boost::format(" %10.5f") % vx[nx - 1] << std::endl;

    OnlineMean_Reset(&oMean);
    boost::this_thread::sleep(boost::posix_time::milliseconds(evalsleep));

    double ysum = oMean.mean;

    for (unsigned int ix0 = 0; ix0 < nx - 1; ++ix0)
      for (unsigned int ix1 = ix0 + 1; ix1 < nx; ++ix1)
        g2[ix0][ix1]->SetPoint(mCurrentIter, xin(ix0), xin(ix1));

    for (unsigned int ix0 = 0; ix0 < nx; ++ix0)
      for (unsigned int i = mCurrentIter; i < n_iterations; ++i)
        g1[ix0]->SetPoint(i, mCurrentIter, xin(ix0));

    return ysum;
  }

  void printBestpoints(const vectord &xin)
  {
    std::cout << "--------------------" << std::endl;
    std::cout << "Best values: " << this->getValueAtMinimum() << std::endl;
    std::cout << "Best points:" << std::endl;
    for (unsigned int i = 0; i < xin.size(); ++i)
      std::cout << boost::format("X%d %10.5f") % i % (xin(i) * (vxrmax[i] - vxrmin[i]) + vxrmin[i]) << std::endl;
    std::cout << "--------------------" << std::endl;
  }

  bool checkReachability(const vectord &query)
  {
    return true;
  };

private:
  std::vector<chid> chan_putinfo;
  std::vector<dbr_double_t *> putinfo;

  unsigned int nx, ny, n_iterations, stepsleep, evalsleep;

  std::vector<double> vxrmin, vxrmax, vxstep, vywght;
  std::vector<std::string> vyname;

  EventHandler *pHandler;

  const unsigned int nxmap = 20;
  const double dxmap = 1. / (double)nxmap;

  HolderOfStaticHistogram hmap;

  static const unsigned int N = 10;
  TH2D *h2[N][N];
  TGraph *g2[N][N];
  TGraph *g1[N];

  TCanvas *c2;
  TCanvas *c1;

  OnlineMean oMean;
};

int main(int argc, char *argv[])
{
  /* ROOT initialization */
  int tmp = 1;
  TApplication theApp("App", &tmp, argv);
  gStyle->SetOptStat(0);

  /* Command line option */
  options_description opt("Option");
  // define format
  opt.add_options()("help,h", "display help")("input,i", value<std::string>(), "ini file");

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
  catch (boost::property_tree::ptree_error &e)
  {
    std::cout << "ptree_error " << e.what() << std::endl;
    exit(-1);
  }

  const unsigned int nx = pt.get<unsigned int>("PV.nx");

  bayesopt::Parameters par;
  par = initialize_parameters_to_default();
  par.n_iterations = pt.get<size_t>("PAR.n_iterations");
  par.random_seed = pt.get<int>("PAR.random_seed") == 0 ? (int)time(0) : pt.get<int>("PAR.random_seed");
  par.verbose_level = pt.get<int>("PAR.verbose_level");
  par.noise = pt.get<double>("PAR.noise");
  par.crit_name = pt.get<std::string>("PAR.crit_name");

  par.surr_name = "sGaussianProcess";
  // par.surr_name = "sStudentTProcessJef";
  // par.surr_name = "sStudentTProcessNIG";
  // par.n_iter_relearn = 1;

  // Define format
  auto format = new boost::posix_time::time_facet("%Y%m%d_%H.%M.%S");
  std::stringstream ss;
  ss.imbue(std::locale(std::cout.getloc(), format));

  // fetch the current time
  auto now = boost::posix_time::second_clock::local_time();
  ss << now;

  bayesopt::utils::ParamLoader::save(ss.str() + ".txt", par);

  TestFunction *myfunc = new TestFunction(pt, nx, par);
  boost::thread th1(&TestFunction::PVMonitor, myfunc);

  myfunc->run();
  vectord result = myfunc->getFinalResult();

  bayesopt::BOptState state;
  myfunc->saveOptimization(state);
  state.saveToFile(ss.str() + ".log");
  std::cout << "STATE ITERS: " << state.mCurrentIter << std::endl;

  myfunc->printBestpoints(result);
  myfunc->evaluateSample(result);

  theApp.Run();

  return 0;
}
