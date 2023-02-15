/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Package      FixPIMDB
   Purpose      Quantum Path Integral Algorithm for Quantum Chemistry
   Copyright    Voth Group @ University of Chicago
   Authors      Chris Knight & Yuxing Peng (yuxing at uchicago.edu)

   Updated      Oct-01-2011
   Version      1.0
------------------------------------------------------------------------- */

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include "fix_pimdb.h"
#include "universe.h"
#include "comm.h"
#include "force.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include <algorithm> 

using namespace LAMMPS_NS;
using namespace FixConst;

enum{PIMD,NMPIMD,CMD};

/* ---------------------------------------------------------------------- */

FixPIMDB::FixPIMDB(LAMMPS *lmp, int narg, char **arg) : FixPIMD(lmp, narg, arg)
{
  nbosons    = atom->nlocal;
  nevery     = 100; // TODO: make configurable (thermo_style?)

  E_kn = std::vector<double>((nbosons * (nbosons + 1) / 2),0.0);
  V = std::vector<double>((nbosons + 1),0.0);

  memory->create(intra_atom_spring_local, nbosons, "FixPIMDB: intra_atom_spring_local");
  memory->create(separate_atom_spring, nbosons, "FixPIMDB: separate_atom_spring");
  memory->create(V_backwards, nbosons + 1, "FixPIMDB: V_backwards");
  memory->create(connection_probabilities, nbosons * nbosons, "FixPIMDB: connection probabilities");
}

/* ---------------------------------------------------------------------- */

FixPIMDB::~FixPIMDB() {
  memory->destroy(connection_probabilities);
  memory->destroy(V_backwards);
  memory->destroy(separate_atom_spring);
  memory->destroy(intra_atom_spring_local);
}

int FixPIMDB::setmask()
{
  int mask = FixPIMD::setmask();
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::setup(int vflag)
{
  FixPIMD::setup(vflag);
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::diff_two_beads(const double* x1, int l1, const double* x2, int l2,
                                double diff[3]) {
  l1 = l1 % np;
  l2 = l2 % np;
  double delx2 = x2[3 * l2 + 0] - x1[3 * l1 + 0];
  double dely2 = x2[3 * l2 + 1] - x1[3 * l1 + 1];
  double delz2 = x2[3 * l2 + 2] - x1[3 * l1 + 2];
  domain->minimum_image(delx2, dely2, delz2);

  diff[0] = delx2;
  diff[1] = dely2;
  diff[2] = delz2;
}

/* ---------------------------------------------------------------------- */

double FixPIMDB::distance_squared_two_beads(const double* x1, int l1, const double* x2, int l2) {
  double diff[3];
  diff_two_beads(x1, l1, x2, l2, diff);
  return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::evaluate_cycle_energies()
{
  double **x = atom->x;
  for (int i = 0; i < nbosons; i++) {
    intra_atom_spring_local[i] = distance_squared_two_beads(*x, i, buf_beads[x_next], i);
  }

  // TODO: enough to communicate to replicas 0,np-1
  MPI_Allreduce(intra_atom_spring_local, separate_atom_spring, nbosons,
                MPI_DOUBLE, MPI_SUM, universe->uworld);

  if (universe->me == 0 || universe->me == np - 1) {

    double* x_first_bead;
    double* x_last_bead;
    if (universe->me == 0) {
      x_first_bead = *x;
      x_last_bead = buf_beads[x_last];
    } else {
      x_first_bead = buf_beads[x_next];
      x_last_bead = *x;
    }

    double ff = fbond * atom->mass[atom->type[0]]; // TODO: ensure they're all the same

    for (int v = 0; v < nbosons; v++) {
      set_Enk(v + 1, 1,
              -0.5 * ff * separate_atom_spring[v]);

      for (int u = v - 1; u >= 0; u--) {
        double val = get_Enk(v + 1, v - u) +
            -0.5 * ff * (
              // Eint(u)
              separate_atom_spring[u] - distance_squared_two_beads(x_first_bead, u, x_last_bead, u)
              // connect u to u+1
              + distance_squared_two_beads(x_last_bead, u, x_first_bead, u + 1)
              // break cycle [u+1,v]
              - distance_squared_two_beads(x_first_bead, u + 1, x_last_bead, v)
              // close cycle from v to u
              + distance_squared_two_beads(x_first_bead, u, x_last_bead, v));

        set_Enk(v + 1, v - u + 1, val);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixPIMDB::get_Enk(int m, int k) {
  int end_of_m = m * (m + 1) / 2;
  return E_kn.at(end_of_m - k);
}

/* ---------------------------------------------------------------------- */

double FixPIMDB::set_Enk(int m, int k, double val) {
  int end_of_m = m * (m + 1) / 2;
  return E_kn.at(end_of_m - k) = val;
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::Evaluate_VBn(std::vector <double>& V, const int n)
{
  const double Boltzmann = force->boltz;
  double beta   = 1.0 / (Boltzmann * nhc_temp);

  for (int m = 1; m < n+1; ++m) {
    double Elongest = std::min((get_Enk(m,1)+V.at(m-1)), (get_Enk(m,m)+V.at(0)));

    double sig_denom = 0.0;
    for (int k = m; k > 0; --k) {
          sig_denom += exp(-beta*(get_Enk(m,k) + V.at(m-k)-Elongest));
    }

    V.at(m) = Elongest-1.0/beta*log(sig_denom / (double)m);
    if(std::isinf(V.at(m)) || std::isnan(V.at(m))) {
	if (universe->iworld ==0){
          std::cout << "sig_denom is: " << sig_denom << " Elongest is: " << Elongest
                    << std::endl;}
          exit(0);
    }
  }
}

void FixPIMDB::Evaluate_V_backwards(double* V_backwards) {
  const double Boltzmann = force->boltz;
  double beta   = 1.0 / (Boltzmann * nhc_temp);

  V_backwards[nbosons] = 0.0;

  for (int l = nbosons - 1; l > 0; l--) {
    double Elongest = std::min(get_Enk(l + 1, 1) + V_backwards[l+1],
                               get_Enk(nbosons, nbosons - l));

    double sig_denom = 0.0;
    for (int p = l; p < nbosons; p++) {
          sig_denom += 1.0 / (p + 1) * exp(-beta *
                                           (get_Enk(p + 1, p - l + 1) + V_backwards[p + 1]
                                            - Elongest)
                                           );
    }

    V_backwards[l] = Elongest - log(sig_denom) / beta;

    if(std::isinf(V_backwards[l]) || std::isnan(V_backwards[l])) {
          if (universe->iworld ==0){
          // TODO: put backwards in the error message
          std::cout << "sig_denom is: " << sig_denom << " Elongest is: " << Elongest
                    << std::endl;}
          exit(0);
    }

    V_backwards[0] = V.at(nbosons);
  }
}


/* ---------------------------------------------------------------------- */

void FixPIMDB::spring_force() {

    evaluate_cycle_energies();

    if (universe->me != 0 && universe->me != np - 1) {
      // interior beads
      FixPIMD::spring_force();
    } else {
      // exterior beads
      V.at(0) = 0.0;

      Evaluate_VBn(V, nbosons);

      Evaluate_V_backwards(V_backwards);
      evaluate_connection_probabilities(V, V_backwards, connection_probabilities);

      // TODO: spring_force output

      if (universe->me == np - 1) {
          spring_force_last_bead(connection_probabilities);
      } else {
          spring_force_first_bead(connection_probabilities);
      }
    }
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::evaluate_connection_probabilities(const std::vector<double>& V,
                                                 const double* V_backwards,
                                                 double* connection_probabilities) {
    const double Boltzmann = force->boltz;
    double beta   = 1.0 / (Boltzmann * nhc_temp);

    for (int l = 0; l < nbosons - 1; l++) {
      double direct_link_probability = 1.0 - (exp(-beta *
                                                (V.at(l + 1) + V_backwards[l + 1] -
                                                 V.at(nbosons))));
      connection_probabilities[nbosons * l + (l + 1)] = direct_link_probability;
    }
    for (int u = 0; u < nbosons; u++) {
      for (int l = u; l < nbosons; l++) {
          double close_cycle_probability = 1.0 / (l + 1) *
              exp(-beta * (V.at(u) + get_Enk(l + 1, l - u + 1) + V_backwards[l + 1]
                         - V.at(nbosons)));
          connection_probabilities[nbosons * l + u] = close_cycle_probability;
      }
    }
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::spring_force_last_bead(const double* connection_probabilities)
{
    double** x = atom->x;
    double** f = atom->f;

    double* x_first_bead = buf_beads[x_next];
    double* x_last_bead = *x;

    virial = 0.0;

    for (int l = 0; l < nbosons; l++) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        for (int next_l = 0; next_l <= l + 1 && next_l < nbosons; next_l++) {
          double diff_next[3];

          diff_two_beads(x_last_bead, l, x_first_bead, next_l, diff_next);

          double prob = connection_probabilities[nbosons * l + next_l];

          sum_x += prob * diff_next[0];
          sum_y += prob * diff_next[1];
          sum_z += prob * diff_next[2];
        }

        double diff_prev[3];
        diff_two_beads(x_last_bead, l, buf_beads[x_last], l, diff_prev);
        sum_x += diff_prev[0];
        sum_y += diff_prev[1];
        sum_z += diff_prev[2];

        double ff = fbond * atom->mass[atom->type[l]];

        // TODO: why does this happen before updating the force?
        virial += -0.5 * (x[l][0] * f[l][0] + x[l][1] * f[l][1] + x[l][2] * f[l][2]);

        f[l][0] -= sum_x * ff;
        f[l][1] -= sum_y * ff;
        f[l][2] -= sum_z * ff;
    }
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::spring_force_first_bead(const double* connection_probabilities)
{
    double** x = atom->x;
    double** f = atom->f;

    double* x_first_bead = *x;
    double* x_last_bead = buf_beads[x_last];

    virial = 0.0;

    for (int l = 0; l < nbosons; l++) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        for (int prev_l = std::max(0, l - 1); prev_l < nbosons; prev_l++) {
          double diff_prev[3];

          diff_two_beads(x_first_bead, l, x_last_bead, prev_l, diff_prev);

          double prob = connection_probabilities[nbosons * prev_l + l];

          sum_x += prob * diff_prev[0];
          sum_y += prob * diff_prev[1];
          sum_z += prob * diff_prev[2];
        }

        double diff_next[3];
        diff_two_beads(x_first_bead, l, buf_beads[x_next], l, diff_next);
        sum_x += diff_next[0];
        sum_y += diff_next[1];
        sum_z += diff_next[2];

        double ff = fbond * atom->mass[atom->type[l]];

        // TODO: why does this happen before updating the force?
        virial += -0.5 * (x[l][0] * f[l][0] + x[l][1] * f[l][1] + x[l][2] * f[l][2]);

        f[l][0] -= sum_x * ff;
        f[l][1] -= sum_y * ff;
        f[l][2] -= sum_z * ff;
    }
}

/* ---------------------------------------------------------------------- */

//FOR PRINTING ENERGIES AND POTENTIALS FOR PIMD-B
void FixPIMDB::end_of_step() {

    if (universe->iworld == 0) {
    //std::cout << "E1\tE12\tE2\tE123\tE23\tE3\tVB0\tVB1\tVB2\tVB3" <<std::endl;
    //#! FIELDS time E1 E2 E12 E3 E23 E123 VB1 VB2 VB3 E_ox3 Vr.bias
      std::ofstream myfile;
      // TODO: make sure that the file is created the first time, not appending to old results
      myfile.open ("pimdb.log", std::ios::out | std::ios::app);

      for (double val: E_kn)
        myfile << val << " ";
      for (double val: V)
        myfile << val << " "; // mult by 2625.499638 to go from Ha to kcal/mol
      myfile << std::endl;

      myfile.close();

    }

}