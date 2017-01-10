#include <cmath>
#include <ODESolvers.hpp>
#include "../model/model.hpp"

/** \file modelJin96.cpp
 *  \brief Definitions for Timmerman and Jin (2002) ENSO model.
 *   
 *  Definitions for Timmerman and Jin (2002) ENSO model.
 */

/**
 * Return the parameters of the model.
 */
void
Jin96::getParameters(double *rho_, double *delta_, double *a_,
		    double *c_, double *k_)
{
  *rho_ = rho;
  *delta_ = delta;
  *a_ = a;
  *c_ = c;
  *k_ = k;
  
  return;
}

/**
 * Set parameters of the model.
 */
void
Jin96::setParameters(const double rho_, const double delta_,
		    const double a_, const double c_, const double k_)
{
  rho = rho_;
  delta = delta_;
  a = a_;
  c = c_;
  k = k_;
  
  return;
}

/** 
 * Evaluate the vector field of the Timmerman and Jin (2002) ENSO model
 * at a given state.
 * \param[in]  state State at which to evaluate the vector field.
 * \param[out] field Vector resulting from the evaluation of the vector field.
 */
void
Jin96::evalField(const gsl_vector *state, gsl_vector *field)
{
  double x, y, z;

  /** Get x, y and z from state */
  x = gsl_vector_get(state, 0);
  y = gsl_vector_get(state, 1);
  z = gsl_vector_get(state, 2);

  /** Tendency of temperature gradient */
  gsl_vector_set(field, 0,
		 rho * delta * (gsl_pow_2(x) - a * x)
		 + x * (x + y + c * (1 - tanh(x + z))));
		 
  /** Tendency of western temperature deviation from radiative equilibrium */
  gsl_vector_set(field, 1,
		 -rho * delta * (a * y + gsl_pow_2(x)));

  /** Tendency of the western thermocline depth anomaly */
  gsl_vector_set(field, 2,
		 delta * (k - z - x / 2));

  return;
}


// /**
//  * Return the parameters of the model.
//  */
// void
// Jin96ContEps::getParameters(double *rho_, double *delta_, double *a_,
// 			    double *c_, double *k_, double *epsFact_)
// {
//   *rho_ = rho;
//   *delta_ = delta;
//   *a_ = a;
//   *c_ = c;
//   *k_ = k;
//   eps
  
//   return;
// }

// /**
//  * Set parameters of the model.
//  */
// void
// Jin96ContEps::setParameters(const double rho_, const double delta_,
// 		    const double a_, const double c_, const double k_)
// {
//   rho = rho_;
//   delta = delta_;
//   a = a_;
//   c = c_;
//   k = k_;
  
//   return;
// }

// /** 
//  * Evaluate the vector field of the Timmerman and Jin (2002) ENSO model
//  * at a given state.
//  * \param[in]  state State at which to evaluate the vector field.
//  * \param[out] field Vector resulting from the evaluation of the vector field.
//  */
// void
// Jin96ContEps::evalField(const gsl_vector *state, gsl_vector *field)
// {
//   double x, y, z;

//   /** Get x, y and z from state */
//   x = gsl_vector_get(state, 0);
//   y = gsl_vector_get(state, 1);
//   z = gsl_vector_get(state, 2);

//   /** Tendency of temperature gradient */
//   gsl_vector_set(field, 0,
// 		 rho * delta * (gsl_pow_2(x) - a * x)
// 		 + x * (x + y + c * (1 - tanh(x + z))));
		 
//   /** Tendency of western temperature deviation from radiative equilibrium */
//   gsl_vector_set(field, 1,
// 		 -rho * delta * (a * y + gsl_pow_2(x)));

//   /** Tendency of the western thermocline depth anomaly */
//   gsl_vector_set(field, 2,
// 		 delta * (k - z - x / 2));

//   return;
// }


