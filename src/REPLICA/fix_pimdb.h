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
  ~FixPIMDB();

  int setmask() override;

  void setup(int) override;
  void end_of_step() override;

  void spring_force() override;

 private:
  void evaluate_cycle_energies();
  void diff_two_beads(const double* x1, int l1, const double* x2, int l2, double diff[3]);
  double spring_energy_two_beads(const double* x1, int l1, const double* x2, int l2);
  double get_Enk(int m, int k);
  double set_Enk(int m, int k, double val);
  void evaluate_connection_probabilities(const std::vector<double>& V, const double* V_backwards,
                                         double* connection_probabilities);
  void spring_force_last_bead(const double* connection_probabilities);
  void spring_force_first_bead(const double* connection_probabilities);
  void Evaluate_VBn(std::vector <double>& V, const int n);
  void Evaluate_V_backwards(double* V_backwards);

  int nbosons;

  double* intra_atom_spring_local;
  double* separate_atom_spring;
  std::vector<double> E_kn;
  std::vector<double> V;
  double* V_backwards;
  double* connection_probabilities;
};


}

#endif
#endif
