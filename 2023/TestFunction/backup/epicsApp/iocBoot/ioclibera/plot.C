{
  const int norg = 4469;
  TGraph2D *grorg = new TGraph2D(norg);

  std::ifstream ifsorg("/home/mitsuka/workspace/StriplineBPM/Debug/fort.26_ler_asym_ver2");
  int ip = 0;
  while (!ifsorg.eof())
    {
      double x, y, v1, v2, v3, v4;
      ifsorg >> x >> y >> v1 >> v2 >> v3 >> v4;

      grorg->SetPoint(ip, x, y, v1);
      
      ip++;
      if (ip==norg)
	return;
    }

  const int nint = 10000;
  TGraph2D *grint = new TGraph2D(nint);

  std::ifstream ifsint("interpol.dat");
  ip = 0;
  while (!ifsint.eof())
    {
      double x, y, v1, v2, v3, v4;
      ifsint >> x >> y >> v1 >> v2 >> v3 >> v4;

      grint->SetPoint(ip, x, y, v1);
      
      ip++;
      if (ip==nint)
	return;
    }
  
  TCanvas *c0 = new TCanvas("c0", "HER", 800, 400);
  c0->Divide(2, 1);
  c0->cd(1);
  grorg->Draw("PZ");

  c0->cd(2);
  grint->Draw("PZ");
  c0->Update();
}
