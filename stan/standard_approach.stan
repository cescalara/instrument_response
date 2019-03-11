/**
 * Fit detected data by forward folding through
 * a response matrix.
 *
 * The model is a power law.
 * @author Francesca Capel
 * @date March 2019
 */

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

  real differential_flux(real E, real F_N, real alpha, real min_energy, real max_energy) {

    return F_N * bounded_power_law(E, alpha, min_energy, max_energy);
  }

  vector integral(vector Ebins, real F_N, real alpha, real min_energy, real max_energy) {

    int len = num_elements(Ebins);
    vector[len-1] output;

    for (i in 1:len-1) {
      output[i] = ((Ebins[i+1] - Ebins[i]) / 6.0) * (differential_flux(Ebins[i], F_N, alpha, min_energy, max_energy)
						     + (4*differential_flux((Ebins[i]+Ebins[i+1])/2, F_N, alpha, min_energy, max_energy))
						     + differential_flux(Ebins[i+1], F_N, alpha, min_energy, max_energy));

    }
    return output;
  }

}

data {

  int Nbins_detected;
  int Nbins_true;

  real min_energy;
  real max_energy;
  
  int n[Nbins_detected];

  matrix[Nbins_true, Nbins_detected] response_matrix;
  vector[Nbins_true+1] true_energy_bins;
  vector[Nbins_detected+1] detected_energy_bins;
  
}

transformed data {

  matrix[Nbins_detected, Nbins_true] transposed_matrix;

  transposed_matrix = response_matrix';

}

parameters {

  real<lower=0, upper=5> F_N;
  real<lower=1, upper=5> alpha;

}

transformed parameters {

  vector[Nbins_true] model_flux;
  vector[Nbins_detected] s;

  /* model */
  for (i in 1:Nbins_detected) {
    
    model_flux = integral(true_energy_bins, F_N, alpha, min_energy, max_energy);

  }
  
  /* forward folding */
  for (i in 1:Nbins_detected) {
    s[i] = dot_product(model_flux, transposed_matrix[i]);
  }

}

model {

  for (i in 1:Nbins_detected) {

    if (s[i] != 0) {
      target += poisson_lpmf(n[i] | s[i]);
    }
    else {
      target += log(1.0);
    }
    
  }
  
}


