/**
 * Fit detected data by treating the 
 * problem hierarchically.
 *
 * The model is a power law.
 * @author Francesca Capel
 * @date March 2019
 */

functions {

#include bspline_ev.stan
#include interpolate.stan

  /**
   * Bounded power law to match the implementation in instrument_response/toy_simulation.py
   */
  real bounded_power_law_lpdf(real E, real alpha, real F_N, real min_energy, real max_energy) {

    real norm = (1-alpha) * ( pow(max_energy, 1-alpha) - pow(min_energy, 1-alpha) ); 

    if (E < min_energy || E > max_energy) {
      return negative_infinity();
    }
    else {
      return log(F_N * norm * pow(E, -alpha));
    }

  }

  /**
   * The number of expected events is the convolution of the effective
   * area and the power law flux.
   *
   * expected_Nevents = \int dE (A_eff(E) dN/dEdtdA)
   *
   * For this simple toy problem, we can do this analytically.
   */
  real get_expected_Nevents(real alpha, real F_N, real min_energy, real max_energy) {

   real norm = (1-alpha) * ( pow(max_energy, 1-alpha) - pow(min_energy, 1-alpha) ); 
   real gamma_fac = (gamma_q(3-alpha, min_energy/10.0) - gamma_q(3-alpha, max_energy/10.0)) * tgamma(3-alpha);
   
   return (10 * pow(10, 3-alpha) * norm * F_N * gamma_fac);
   
  }

  /**
   * Effective area as a function of energy [m^2].
   * Matches the implementation in instrument_response/toy_simulation.py.
   */
  real get_effective_area(real E) {

    return (10 * pow(E, 2) * exp(-E/10));

  }
  
}

data {

  int Nevents;
  vector[Nevents] Edet;

  real min_energy;
  real max_energy;
  
  /* interpolation */
  int  Ngrid;
  vector[Ngrid] Etrue_grid;
  vector[Ngrid] cond_log_prob[Nevents];
  
}


parameters {

  real<lower=0, upper=5> F_N;
  real<lower=1, upper=3> alpha;

  /* latent true energies */
  vector<lower=min_energy, upper=max_energy>[Nevents] Etrue;

}


model {

  vector[Nevents] lp;
  real log_p_det_given_true; 
  real Nex;
  
  for (i in 1:Nevents) {
    
    lp[i] = 0;
    
    /* spectrum */
    lp[i] += bounded_power_law_lpdf(Etrue[i] | alpha, F_N, min_energy, max_energy);


    /* P(Edet | Etrue) */
    log_p_det_given_true = interpolate(Etrue_grid, cond_log_prob[i], Etrue[i]);
    lp[i] += log_p_det_given_true;

    /* effective area */
    lp[i] += log(get_effective_area(Etrue[i]));
    
    target += lp[i];
  }


  /* expected number of events */
  Nex = get_expected_Nevents(alpha, F_N, min_energy, max_energy);
  target += -Nex;

}
