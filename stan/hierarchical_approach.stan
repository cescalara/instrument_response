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
  
  real bounded_power_law_lpdf(real E, real alpha, real N, real min_energy, real max_energy) {

    real norm = (1-alpha) * ( pow(max_energy, 1-alpha) - pow(min_energy, 1-alpha) ); 

    if (E < min_energy || E > max_energy) {
      return negative_infinity();
    }
    else {
      return log(N * norm * pow(E, -alpha));
    }

  }

  real get_Nex(real alpha, real N, real min_energy, real max_energy) {

   real norm = (1-alpha) * ( pow(max_energy, 1-alpha) - pow(min_energy, 1-alpha) ); 

   real gamma_fac = (gamma_q(3-alpha, min_energy/10.0) - gamma_q(3-alpha, max_energy/10.0)) * tgamma(3-alpha);
   
   return (10 * pow(10, 3-alpha) * norm * N * gamma_fac);
   
  }

  real get_Aeff(real E, real maximum) {

    return (10 * pow(E, 2) * exp(-E/10)) / maximum;

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

  /* interpolation */
  int  Ngrid;
  vector[Ngrid] cond_prob[Nevents];
  vector[Ngrid] Etrue_grid;
  
}

transformed data {
  
  real epsilon = 1.0e-5;
  //real N = 0.9236320123663313;
  //real effective_area_max = 541.3411329464508;
  real effective_area_max = 1.0;

}

parameters {

  real<lower=0, upper=5> N;
  real<lower=1, upper=3> alpha;

  /* latent true energies */
  vector<lower=min_energy, upper=max_energy>[Nevents] Etrue;

}


model {

  vector[Nevents] lp;
  vector[Nevents] P_Edet_given_Etrue;
  real log_prob_E;
  real Et;
  real Ed;
  real Nex;
  
  for (i in 1:Nevents) {
    
    lp[i] = 0;
    /* spectrum */
    lp[i] += bounded_power_law_lpdf(Etrue[i] | alpha, N, min_energy, max_energy);

    /* deal with boundary conditions */
    /*
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
    */

    //P_Edet_given_Etrue[i] = bspline_func_2d(xknots, yknots, p, c, Etrue[i], Edet[i]);
    P_Edet_given_Etrue[i] = interpolate(Etrue_grid, cond_prob[i], Etrue[i]);

    /* P(Edet | Etrue) */
    log_prob_E = P_Edet_given_Etrue[i];
    if (log_prob_E != negative_infinity() && !is_nan(log_prob_E)) {
	lp[i] += log_prob_E;
    }
    else{
      lp[i] += negative_infinity();
    }

    /* Aeff */
    lp[i] += log(get_Aeff(Etrue[i], effective_area_max));
    
    target += lp[i];
  }


  Nex = get_Nex(alpha, N, min_energy, max_energy);
  print("Nex: ", Nex);
  target += -Nex;

  /* prior */
  alpha ~ normal(2, 1);
  N ~ normal(1, 1);

}

