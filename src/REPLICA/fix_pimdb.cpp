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
  if (method == CMD) {
    error->universe_all(FLERR, "Method cmd not supported in fix pimdb");
  }

  nbosons    = atom->nlocal;
  nevery     = 100; // TODO: make configurable (thermo_style?)
  
  memory->create(multiplex_atom_indices, nbosons, "FixPIMDB::multiplex_atom_indices");
  memory->create(temp_nbosons_array, nbosons, "FixPIMDB: temp_nbosons_array");
  memory->create(separate_atom_spring, nbosons, "FixPIMDB: separate_atom_spring");
  memory->create(E_kn, (nbosons * (nbosons + 1) / 2), "FixPIMDB: E_kn");
  memory->create(V, nbosons + 1, "FixPIMDB: V");
  memory->create(V_backwards, nbosons + 1, "FixPIMDB: V_backwards");
  memory->create(connection_probabilities, nbosons * nbosons, "FixPIMDB: connection probabilities");

  for(int i = 0; i < nbosons; i++) {
    multiplex_atom_indices[i] = i;
  }
}

/* ---------------------------------------------------------------------- */

FixPIMDB::~FixPIMDB() {
  memory->destroy(connection_probabilities);
  memory->destroy(V_backwards);
  memory->destroy(V);
  memory->destroy(E_kn);
  memory->destroy(separate_atom_spring);
  memory->destroy(temp_nbosons_array);
  memory->destroy(multiplex_atom_indices);
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
  l1 = multiplex_atom_indices[l1 % nbosons];
  l2 = multiplex_atom_indices[l2 % nbosons];
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
    temp_nbosons_array[i] = distance_squared_two_beads(*x, i, buf_beads[x_next], i);
  }

  // TODO: enough to communicate to replicas 0,np-1
  MPI_Allreduce(temp_nbosons_array, separate_atom_spring, nbosons,
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
  return E_kn[end_of_m - k];
}

/* ---------------------------------------------------------------------- */

double FixPIMDB::set_Enk(int m, int k, double val) {
  int end_of_m = m * (m + 1) / 2;
  return E_kn[end_of_m - k] = val;
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::Evaluate_VBn()
{
  const double Boltzmann = force->boltz;
  double beta   = 1.0 / (Boltzmann * nhc_temp);

  V[0] = 0.0;

  for (int m = 1; m < nbosons + 1; m++) {
    double Elongest = std::numeric_limits<double>::max();

    for (int k = m; k > 0; k--) {
    	double val = get_Enk(m,k) + V[m-k];
    	Elongest = std::min(Elongest, val);
      temp_nbosons_array[k] = val;
    }

    double sig_denom = 0.0;
    for (int k = m; k > 0; k--) {
          sig_denom += exp(-beta * (temp_nbosons_array[k] - Elongest));
    }

    V[m] = Elongest - (1.0 / beta) * log(sig_denom / (double)m);

    if (!std::isfinite(V[m])) {
          error->universe_one(
              FLERR,
              fmt::format("Invalid sig_denom {} with Elongest {} in fix pimdb potential",
                          sig_denom, Elongest));
    }
  }
}

void FixPIMDB::Evaluate_V_backwards() {
  const double Boltzmann = force->boltz;
  double beta   = 1.0 / (Boltzmann * nhc_temp);

  V_backwards[nbosons] = 0.0;

  for (int l = nbosons - 1; l > 0; l--) {
    double Elongest = std::numeric_limits<double>::max();
    for (int p = l; p < nbosons; p++) {
      double val = get_Enk(p + 1, p - l + 1) + V_backwards[p + 1];
      Elongest = std::min(Elongest, val);
      temp_nbosons_array[p] = val;
    }

    double sig_denom = 0.0;
    for (int p = l; p < nbosons; p++) {
          sig_denom += 1.0 / (p + 1) * exp(-beta *
                                           (temp_nbosons_array[p]
                                            - Elongest)
                                           );
    }

    V_backwards[l] = Elongest - log(sig_denom) / beta;

    if (!std::isfinite(V_backwards[l])) {
          error->universe_one(
              FLERR,
              fmt::format("Invalid sig_denom {} with Elongest {} in fix pimdb potential backwards",
                          sig_denom, Elongest));
    }
  }

  V_backwards[0] = V[nbosons];
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::possibly_shuffle_atoms_indices() {
  if (shuffle_indices_every > 0 && update->ntimestep % shuffle_indices_every == 0) { 
    std::random_shuffle(&multiplex_atom_indices[0], &multiplex_atom_indices[atom->nlocal]);
  }
}


/* ---------------------------------------------------------------------- */

void FixPIMDB::spring_force() {

    possibly_shuffle_atoms_indices();

    evaluate_cycle_energies();

    if (universe->me != 0 && universe->me != np - 1) {
      // interior beads
      FixPIMD::spring_force();
      spring_energy = 0.0;
    } else {
      // exterior beads
      Evaluate_VBn();

      Evaluate_V_backwards();
      evaluate_connection_probabilities();

      if (universe->me == np - 1) {
          spring_force_last_bead();
      } else {
          spring_force_first_bead();
      }
    }
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::evaluate_connection_probabilities() {
    const double Boltzmann = force->boltz;
    double beta   = 1.0 / (Boltzmann * nhc_temp);

    for (int l = 0; l < nbosons - 1; l++) {
      double direct_link_probability = 1.0 - (exp(-beta *
                                                (V[l + 1] + V_backwards[l + 1] -
                                                 V[nbosons])));
      connection_probabilities[nbosons * l + (l + 1)] = direct_link_probability;
    }
    for (int u = 0; u < nbosons; u++) {
      for (int l = u; l < nbosons; l++) {
          double close_cycle_probability = 1.0 / (l + 1) *
              exp(-beta * (V[u] + get_Enk(l + 1, l - u + 1) + V_backwards[l + 1]
                         - V[nbosons]));
          connection_probabilities[nbosons * l + u] = close_cycle_probability;
      }
    }
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::spring_force_last_bead()
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

        int l_actual = multiplex_atom_indices[l];

        virial += -0.5 * (x[l_actual][0] * f[l_actual][0] + x[l_actual][1] * f[l_actual][1] + x[l_actual][2] * f[l_actual][2]);

        f[l_actual][0] -= sum_x * ff;
        f[l_actual][1] -= sum_y * ff;
        f[l_actual][2] -= sum_z * ff;
    }

    spring_energy = V[nbosons];
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::spring_force_first_bead()
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
        int l_actual = multiplex_atom_indices[l];

        virial += -0.5 * (x[l_actual][0] * f[l_actual][0] + x[l_actual][1] * f[l_actual][1] + x[l_actual][2] * f[l_actual][2]);

        f[l_actual][0] -= sum_x * ff;
        f[l_actual][1] -= sum_y * ff;
        f[l_actual][2] -= sum_z * ff;
    }

    spring_energy = 0.0;
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

      for (int i = 0; i < nbosons * (nbosons + 1) / 2; i++) {
          myfile << E_kn[i] << " ";
      }
      for (int i = 0; i < nbosons + 1; i++) {
          myfile << V[i] << " "; // mult by 2625.499638 to go from Ha to kcal/mol
      }
      myfile << std::endl;

      myfile.close();

    }

}