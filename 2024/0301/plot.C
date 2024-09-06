{
  gSystem->Setenv("TZ", "UTC");
  gStyle->SetTimeOffset(0);

  TFile *fin = new TFile("fout.root", "read");
  TGraph *g0 = (TGraph*)fin->Get("BMLDCCT:RATE");    
  TGraph *g1 = (TGraph*)fin->Get("BMLDCCT:CURRENT");    
  TGraph *g2 = (TGraph*)fin->Get("CG_BTP:BPM:CHARGE1");    
  TGraph *g3 = (TGraph*)fin->Get("CGLINJ:EFFICIENCY");

  // const int t0 = 1709164800 - 788918400; // Feb. 29, 00:00:00
  // const int t1 = 1709169600 - 788918400; // Feb. 29, 01:20:00

  // const int t0 = 1709287500 - 788918400; // Mar. 1, 10:05:00
  const int t0 = 1709289900 - 788918400; // Mar. 1, 10:45:00

  const int t1 = 1709291700 - 788918400; // Mar. 1, 11:15:00
  // const int t1 = 1709289120 - 788918400; // Mar. 1, 11:32:00
  // const int t1 = 1709298000 - 788918400; // Mar. 1, 13:00:00
  
  std::vector<double> vdcct, vrate;
  for (int i=0; i<=(t1-t0); ++i)
    {
      const double sec  = double(t0+i);
      const double dcct = g1->Eval(sec); // mA
      const double rate = g0->Eval(sec); // mA/s
      
      if (rate > 0. || rate < -1.)
	continue;
      
      vdcct.push_back(dcct);
      vrate.push_back(rate);
    }

  TGraph *gdcctvsrate = new TGraph(vdcct.size(), &vdcct[0], &vrate[0]);
  gdcctvsrate->SetTitle("");
  gdcctvsrate->SetMarkerStyle(20+4);
  gdcctvsrate->GetHistogram()->SetMinimum(-0.3);
  gdcctvsrate->GetHistogram()->SetMaximum(+0.04);
  gdcctvsrate->GetXaxis()->SetTitle("DCCT (mA)");
  gdcctvsrate->GetYaxis()->SetTitle("dI_{DCCT}/dt (mA/s)");
  gdcctvsrate->GetYaxis()->SetTitleOffset(1.25);

  TF1 *fdcctvsrate = new TF1("fdcctvsrate", "[0]*log(1+exp([1]*x+[2]))", 0, 100);
  fdcctvsrate->SetParameter(0, -0.02);
  fdcctvsrate->SetParameter(1, +0.10);
  fdcctvsrate->SetParameter(2, -2.00);
  fdcctvsrate->SetLineColor(2);
  
  TCanvas *cdcctvsrate = new TCanvas("cdcctvsrate", "cdcctvsrate", 600, 600);
  cdcctvsrate->cd(1);
  gdcctvsrate->Fit(fdcctvsrate, "NOME");
  gdcctvsrate->Draw("AP");
  fdcctvsrate->Draw("same");
  cdcctvsrate->Update();

  const int t3 = 1709287500 - 788918400; // Mar. 1, 10:05:00
  const int t4 = 1709289120 - 788918400; // Mar. 1, 11:32:00
  
  std::vector<double> vsec, veff, vcor;
  for (unsigned int i=0; i<g0->GetMaxSize(); ++i)
    {
      double sec, rate;
      g0->GetPoint(i, sec, rate);
            
      if (sec < t3)
	continue;

      const double btma = (g2->Eval(sec)-0.1) * 0.1; // -0.1 is the charge offset, then convert nC -> mA

      vsec.push_back(sec);
      veff.push_back(rate/btma * 100); // %

      const double dcct = g1->Eval(sec); // mA
      vcor.push_back((rate - fdcctvsrate->Eval(dcct))/btma * 100.); // %
    }

  TGraph *gsecvseff = new TGraph(vsec.size(), &vsec[0], &veff[0]);
  gsecvseff->SetTitle("");
  gsecvseff->SetMarkerStyle(20+5);
  gsecvseff->SetMarkerColor(6);
  gsecvseff->SetLineColor(6);
  gsecvseff->GetHistogram()->SetMinimum(0);
  gsecvseff->GetHistogram()->SetMaximum(150);
  gsecvseff->GetYaxis()->SetTitle("Efficiency (%)");
  gsecvseff->GetYaxis()->SetTitleOffset(1.25);

  TGraph *gsecvscor = new TGraph(vsec.size(), &vsec[0], &vcor[0]);
  gsecvscor->SetMarkerStyle(20+6);
  gsecvscor->SetMarkerColor(7);
  gsecvscor->SetLineColor(7);
  
  TCanvas *csecvseff = new TCanvas("csecvseff", "csecvseff", 600, 600);
  csecvseff->cd(1);
  TAxis *axis = gsecvseff->GetXaxis();
  axis->SetTimeDisplay(1);
  axis->SetTimeFormat("%H:%M");
  gsecvseff->Draw("AP");
  gsecvscor->Draw("P,same");
  g3->Draw("P,same");
  csecvseff->Update();
}
