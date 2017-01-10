#ifndef MODELTJ02_HPP
#define MODELTJ02_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <ODESolvers.hpp>

/** \file modelTJ02.hpp
 *  \brief Vector field for the Jin 96 model of ENSO.
 *   
 *  Vector field for the Jin 96 (Roberts et al 2016 implementation) model of
 *  El Nino Southern Oscillation.
 */

/** \brief Vector field for the Jin 96 model of ENSO.
 *   
 *  Vector field for the Jin 96 (Roberts et al 2016 implementation) model of
 *  El Nino Southern Oscillation.
 */
class Jin96 : public vectorField {
  double rho;
  double delta;
  double a;
  double c;
  double k;
  
  
public:
  /** \brief Constructor defining the model parameters. */
  Jin96(const double rho_, const double delta_, const double a_,
       const double c_, const double k_)
    : vectorField(), rho(rho_), delta(delta_), a(a_), c(c_), k(k_) {}
  
  /** \brief Destructor. */
  ~Jin96() { }

  /** \brief Return the parameters of the model. */
  void getParameters(double *rho_, double *delta_, double *a_,
		     double *c_, double *k_);
		     
  /** \brief Set parameters of the model. */
  void setParameters(const double rho_, const double delta_,
		     const double a_, const double c_, const double k_);

  /** \brief  Evaluate the vector field of the Jin 96
   * ENSO model at a given state. */
  void evalField(const gsl_vector *state, gsl_vector *field);
  
};

// /** \brief Vector field for the Jin 96 model of ENSO for continuation.
//  *   
//  *  Vector field for the Jin 96 (Roberts et al 2016 implementation) model of
//  *  El Nino Southern Oscillation for continuation.
//  */
// class Jin96ContEps : public vectorField {
//   double rho;
//   double delta;
//   double a;
//   double c;
//   double k;
  
  
// public:
//   /** \brief Constructor defining the model parameters. */
//   Jin96ContEps(const double rho_, const double delta_, const double a_,
//        const double c_, const double k_)
//     : vectorField(), rho(rho_), delta(delta_), a(a_), c(c_), k(k_) {}
  
//   /** \brief Destructor. */
//   ~Jin96ContEps() { }

//   /** \brief Return the parameters of the model. */
//   void getParameters(double *rho_, double *delta_, double *a_,
// 		     double *c_, double *k_);
		     
//   /** \brief Set parameters of the model. */
//   void setParameters(const double rho_, const double delta_,
// 		     const double a_, const double c_, const double k_);

//   /** \brief  Evaluate the vector field of the Jin 96
//    * ENSO model at a given state. */
//   void evalField(const gsl_vector *state, gsl_vector *field);
  
// };

#endif
