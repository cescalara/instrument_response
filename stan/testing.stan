
functions {

#include bspline_ev.stan

}

data {

 /* spline */
  int p; // spline degree
  int Lknots_x; // length of knot vector
  int Lknots_y; // length of knot vector

  vector[Lknots_x] xknots; // knot sequence - needs to be a monotonic sequence
  vector[Lknots_y] yknots; // knot sequence - needs to be a monotonic sequence
 
  matrix[Lknots_x+p-1, Lknots_y+p-1] c; // spline coefficients 

  int Nevals; // length of vector to evaluate
  vector<lower=xknots[1], upper=xknots[Lknots_x]>[Nevals] xvals; // vector to evaluate
  vector<lower=yknots[1], upper=yknots[Lknots_y]>[Nevals] yvals; // vector to evaluate
  
}

generated quantities {

  matrix[Nevals, Nevals] zvals;
  matrix[Nevals, Nevals] log_zvals;

  for (idx_x in 1:Nevals) {
    for (idx_y in 1:Nevals) {

      zvals[idx_x, idx_y] = bspline_func_2d(xknots, yknots, p, c, xvals[idx_x], yvals[idx_y]);
      log_zvals[idx_x, idx_y] = log(zvals[idx_x, idx_y]);
      
    } 
  } 
  
  
}
