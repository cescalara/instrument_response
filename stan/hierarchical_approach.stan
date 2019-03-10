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

  real bounded_power_law_lpdf(real E, real alpha, real N, real min_energy, real max_energy) {

    real norm = (1-alpha) * ( pow(max_energy, 1-alpha) - pow(min_energy, 1-alpha) ); 

    if (E < min_energy || E > max_energy) {
      return negative_infinity();
    }
    else {
      return log(norm * pow(E, -alpha));
    }

  }

}

data {

  int Nevents;
  vector[Nevents] Edet;

  real min_energy;
  real max_energy;
  
  /* spline */
  int p; // spline degree
  int Lknots_x; // length of knot vector
  int Lknots_y; // length of knot vector

  vector[Lknots_x] xknots; // knot sequence - needs to be a monotonic sequence
  vector[Lknots_y] yknots; // knot sequence - needs to be a monotonic sequence
 
  matrix[Lknots_x+p-1, Lknots_y+p-1] c; // spline coefficients 

}

transformed data {
  
  real epsilon = 1.0e-5;
  
}

parameters {

  real<lower=0, upper=5> N;
  real<lower=1.1, upper=5> alpha;

  /* latent true energies */
  vector<lower=min_energy, upper=max_energy>[Nevents] Etrue;

}

transformed parameters {

  vector[Nevents] P_Edet_given_Etrue;
  real Et;
  real Ed;
  
  /* evaluate the spline */
  for (i in 1:Nevents) {

    /* deal with boundary conditions */
    if (Etrue[i] < xknots[1]+epsilon) {
      Et = xknots[1]+epsilon;
    }
    else if (Etrue[i] > xknots[Lknots_x]-epsilon) {
      Et = xknots[Lknots_x]-epsilon;
    }
    else {
      Et = Etrue[i];
    }

    if (Edet[i] < yknots[1]+epsilon) {
      Ed = yknots[1]+epsilon;
    }
    else if (Edet[i] > yknots[Lknots_y]-epsilon) {
      Ed = yknots[Lknots_y]-epsilon;
    }
    else{
      Ed = Edet[i];
    }

    
    P_Edet_given_Etrue[i] = bspline_func_2d(xknots, yknots, p, c, Et, Ed);

    
  }

}

model {

  vector[Nevents] lp;
  real log_prob_E;
  
  for (i in 1:Nevents) {

    lp[i] = 0;
    /* spectrum */
    lp[i] += bounded_power_law_lpdf(Etrue[i] | alpha, N, min_energy, max_energy);


    /* P(Edet | Etrue) */
    log_prob_E = log(P_Edet_given_Etrue[i]);
    if (log_prob_E != negative_infinity()) {
      if (!is_nan(log_prob_E)) {
	lp[i] += log(P_Edet_given_Etrue[i]);
      }
    }


  }


  target += log_sum_exp(lp);

}
