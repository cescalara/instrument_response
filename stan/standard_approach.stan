/**
 * Fit detected data by forward folding through
 * a response matrix.
 *
 * The model is a power law.
 * @author Francesca Capel
 * @date March 2019
 */

functions {

  /* Define functions to match implementation in response_demo.ipynb */

  /**
   * Bounded power law to match the implementation in instrument_response/toy_simulation.py
   */
  real bounded_power_law(real E, real alpha, real min_energy, real max_energy) {

    real norm = (1-alpha) * ( pow(max_energy, 1-alpha) - pow(min_energy, 1-alpha) ); 

    if (E < min_energy || E > max_energy) {
      return 0;
    }
    else {
      return norm * pow(E, -alpha);
    }
  }

  /**
   * F_N * power law 
   */
  real differential_flux(real E, real F_N, real alpha, real min_energy, real max_energy) {

    return F_N * bounded_power_law(E, alpha, min_energy, max_energy);
  }

  /**
   * Integrate the differential flux using Simpson's rule
   */
  row_vector integral(vector Ebins, real F_N, real alpha, real min_energy, real max_energy) {

    int len = num_elements(Ebins);
    row_vector[len-1] output;

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


parameters {

  real<lower=0, upper=5> F_N;
  real<lower=1, upper=5> alpha;

}

transformed parameters {

  row_vector[Nbins_true] model_flux;
  row_vector[Nbins_detected] s;

  /* forward folding */
  for (i in 1:Nbins_detected) {

    model_flux = integral(true_energy_bins, F_N, alpha, min_energy, max_energy);

  }

  s = model_flux * response_matrix;

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


