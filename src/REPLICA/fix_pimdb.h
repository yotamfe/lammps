/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(pimdb,FixPIMDB)

#else

#ifndef FIX_PIMDB_H
#define FIX_PIMDB_H

#include "fix_pimd.h"
#include <vector>

namespace LAMMPS_NS {

class FixPIMDB : public FixPIMD {
 public:
  FixPIMDB(class LAMMPS *, int, char **);

  int setmask() override;

  void end_of_step() override;

  void spring_force() override;
  double Evaluate_Ekn(const int n, const int k);
  std::vector<double> Evaluate_dEkn_on_atom(const int n, const int k, const int atomnum);
  std::vector<double> Evaluate_VBn(std::vector <double>& V, const int n);
  std::vector<std::vector<double>> Evaluate_dVBn(const std::vector <double>& V, const std::vector <double>& save_E_kn, const int n);

  std::vector<double> E_kn;
  std::vector<double> V;
  int nbosons;
};


}

#endif
#endif
