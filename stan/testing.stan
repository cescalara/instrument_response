
functions {

  real bounded_power_law(real E, real alpha, real min_energy, real max_energy) {

    real norm = (1-alpha) * ( pow(max_energy, 1-alpha) - pow(min_energy, 1-alpha) ); 

    if (E < min_energy || E > max_energy) {
      return 0;
    }
    else {
      return norm * pow(E, -alpha);
    }
  }

  real differential_flux(real E, real N, real alpha, real min_energy, real max_energy) {

    return N * bounded_power_law(E, alpha, min_energy, max_energy);
  }

  vector integral(vector Ebins, real N, real alpha, real min_energy, real max_energy) {

    int len = num_elements(Ebins);
    vector[len-1] output;

    for (i in 1:len-1) {
      output[i] = ((Ebins[i+1] - Ebins[i]) / 6.0) * (differential_flux(Ebins[i], N, alpha, min_energy, max_energy)
						     + (4*differential_flux((Ebins[i]+Ebins[i+1])/2, N, alpha, min_energy, max_energy))
						     + differential_flux(Ebins[i+1], N, alpha, min_energy, max_energy));

    }
    return output;
  }

}

data {

  int Nbins_true;
  int Nbins_detected;

  vector[Nbins_true+1] true_energy_bins;
  vector[Nbins_detected+1] detected_energy_bins;

  real alpha;
  real N;

  real min_energy;
  real max_energy;
  
}

generated quantities {


  vector[Nbins_true] spectrum;
  vector[Nbins_true] model_flux;
  
  for (i in 1:Nbins_true) {

    spectrum[i] = differential_flux(true_energy_bins[i], N, alpha, min_energy, max_energy); 
    model_flux = integral(true_energy_bins, N, alpha, min_energy, max_energy);
      
  }
  
  
}
