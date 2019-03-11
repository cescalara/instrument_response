real interpolate(vector x_values, vector y_values, real x) {
  real x_left;
  real y_left;
  real x_right;
  real y_right;
  real dydx;
  
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];
  int i = 1;
  
  if (x > xmax || x < xmin) {
    print("Error, x is outside of interpolation range!");
    print("x: ", x);
    return 1;
  }
  
  if( x >= x_values[Nx - 1] ) {
    i = Nx - 1;
  }
  else {
    while( x > x_values[i + 1] ) { i = i+1; }
  }
  
  x_left = x_values[i];
  y_left = y_values[i];
  x_right = x_values[i + 1];
  y_right = y_values[i + 1];
  
  dydx = (y_right - y_left) / (x_right - x_left);
  
  return y_left + dydx * (x - x_left);
}
