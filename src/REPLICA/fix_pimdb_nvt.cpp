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
#include "fix_pimdb_nvt.h"
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

enum { PIMD, NMPIMD, CMD };

/* ---------------------------------------------------------------------- */

FixPIMDBNVT::FixPIMDBNVT(LAMMPS *lmp, int narg, char **arg) : FixPIMDNVT(lmp, narg, arg)
{
    if (method == CMD) {
        error->universe_all(FLERR, "Method cmd not supported in fix pimdb");
    }

    nbosons = atom->nlocal;
    nevery = 100; // TODO: make configurable (thermo_style?)

    memory->create(temp_nbosons_array, nbosons, "FixPIMDBNVT: temp_nbosons_array");
    memory->create(separate_atom_spring, nbosons, "FixPIMDBNVT: separate_atom_spring");
    memory->create(E_kn, (nbosons * (nbosons + 1) / 2), "FixPIMDBNVT: E_kn");
    memory->create(V, nbosons + 1, "FixPIMDBNVT: V");
    memory->create(V_backwards, nbosons + 1, "FixPIMDBNVT: V_backwards");
    memory->create(connection_probabilities, nbosons * nbosons, "FixPIMDBNVT: connection probabilities");

    if (est_options[PRIMITIVE]) memory->create(prim_est, nbosons + 1, "FixPIMDBNVT: prim_est");
}

/* ---------------------------------------------------------------------- */

FixPIMDBNVT::~FixPIMDBNVT()
{
  if (est_options[PRIMITIVE]) memory->destroy(prim_est);

    memory->destroy(connection_probabilities);
    memory->destroy(V_backwards);
    memory->destroy(V);
    memory->destroy(E_kn);
    memory->destroy(separate_atom_spring);
    memory->destroy(temp_nbosons_array);
}

int FixPIMDBNVT::setmask()
{
    int mask = FixPIMDNVT::setmask();
    mask |= END_OF_STEP;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixPIMDBNVT::setup(int vflag)
{
    FixPIMDNVT::setup(vflag);
    end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixPIMDBNVT::diff_two_beads(const double *x1, int l1, const double *x2, int l2,
    double diff[3]) {
    l1 = l1 % nbosons;
    l2 = l2 % nbosons;
    double delx2 = x2[3 * l2 + 0] - x1[3 * l1 + 0];
    double dely2 = x2[3 * l2 + 1] - x1[3 * l1 + 1];
    double delz2 = x2[3 * l2 + 2] - x1[3 * l1 + 2];
    domain->minimum_image(delx2, dely2, delz2);

    diff[0] = delx2;
    diff[1] = dely2;
    diff[2] = delz2;
}

/* ---------------------------------------------------------------------- */

double FixPIMDBNVT::distance_squared_two_beads(const double *x1, int l1, const double *x2, int l2)
{
    double diff[3];
    diff_two_beads(x1, l1, x2, l2, diff);
    return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
}

/* ---------------------------------------------------------------------- */

void FixPIMDBNVT::evaluate_cycle_energies()
{
    double** x = atom->x;
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
        }
        else {
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

double FixPIMDBNVT::get_Enk(int m, int k)
{
    int end_of_m = m * (m + 1) / 2;
    return E_kn[end_of_m - k];
}

/* ---------------------------------------------------------------------- */

double FixPIMDBNVT::set_Enk(int m, int k, double val)
{
    int end_of_m = m * (m + 1) / 2;
    return E_kn[end_of_m - k] = val;
}

/* ---------------------------------------------------------------------- */

void FixPIMDBNVT::Evaluate_VBn()
{
    const double Boltzmann = force->boltz;
    double beta = 1.0 / (Boltzmann * nhc_temp);

    V[0] = 0.0;

    for (int m = 1; m < nbosons + 1; m++) {
        double Elongest = std::numeric_limits<double>::max();

        for (int k = m; k > 0; k--) {
            double val = get_Enk(m, k) + V[m - k];
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

void FixPIMDBNVT::Evaluate_V_backwards()
{
    const double Boltzmann = force->boltz;
    double beta = 1.0 / (Boltzmann * nhc_temp);

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

void FixPIMDBNVT::spring_force()
{

    evaluate_cycle_energies();

    if (universe->me != 0 && universe->me != np - 1) {
        // interior beads
        FixPIMDNVT::spring_force();
        spring_energy = 0.0;
        if (est_options[PRIMITIVE]) primitive = 0.0;
    }
    else {
        // exterior beads
        Evaluate_VBn();

        Evaluate_V_backwards();
        evaluate_connection_probabilities();

        if (universe->me == np - 1) {
            spring_force_last_bead();
        }
        else {
            spring_force_first_bead();
        }
    }
}

/* ---------------------------------------------------------------------- */

void FixPIMDBNVT::evaluate_connection_probabilities()
{
    const double Boltzmann = force->boltz;
    double beta = 1.0 / (Boltzmann * nhc_temp);

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

void FixPIMDBNVT::spring_force_last_bead()
{
    double** x = atom->x;
    double** f = atom->f;

    double* x_first_bead = buf_beads[x_next];
    double* x_last_bead = *x;

    if (est_options[VIRIAL]) virial = 0.0;

    // Prepare the images for potential unwrapping (in the case of the centroid estimator).
    double unwrap[3];
    imageint *image = atom->image;

    if (est_options[CENTROID_VIR]) centroid_vir = 0.0;
    if (est_options[GLOB_CENTROID_VIR]) glob_centroid_vir = 0.0;

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

        /*
        // Energy estimators
        if (est_options[GLOB_CENTROID_VIR]) {
          // Global centroid-virial kinetic energy estimator calculation (without the constant and the potential energy terms).
          // Useful for translationally-invariant (periodic) bosonic systems.

          domain->unmap(x[l], image[l], unwrap);

          double diff_x = unwrap[0] - glob_centroid[0];
          double diff_y = unwrap[1] - glob_centroid[1];
          double diff_z = unwrap[2] - glob_centroid[2];

          // domain->minimum_image(diff_x, diff_y, diff_z);  // Probably not needed?

          glob_centroid_vir += -0.5 * (diff_x * f[l][0] + diff_y * f[l][1] + diff_z * f[l][2]);
        }
        */

        // Energy estimators
        if (est_options[CENTROID_VIR] || est_options[GLOB_CENTROID_VIR]) {
            domain->unmap(x[l], image[l], unwrap);

            if (est_options[CENTROID_VIR]) {
                // Standard centroid-virial kinetic energy estimator calculation (without the constant and the potential energy terms).
                // Useful for translationally-invariant (periodic) distinguishable systems.

                double diff_x = unwrap[0] - centroids[3 * l];
                double diff_y = unwrap[1] - centroids[3 * l + 1];
                double diff_z = unwrap[2] - centroids[3 * l + 2];

                // domain->minimum_image(diff_x, diff_y, diff_z);  // Probably not needed if we use unmap?

                centroid_vir += -0.5 * (diff_x * f[l][0] + diff_y * f[l][1] + diff_z * f[l][2]);
            }

            if (est_options[GLOB_CENTROID_VIR]) {
                // Global centroid-virial kinetic energy estimator calculation (without the constant and the potential energy terms).

                double diff_x = unwrap[0] - glob_centroid[0];
                double diff_y = unwrap[1] - glob_centroid[1];
                double diff_z = unwrap[2] - glob_centroid[2];

                // domain->minimum_image(diff_x, diff_y, diff_z);  // Probably not needed if we use unmap?

                glob_centroid_vir += -0.5 * (diff_x * f[l][0] + diff_y * f[l][1] + diff_z * f[l][2]);
            }
        }

        if (est_options[VIRIAL]) {
          // Virial kinetic energy estimator.
          virial += -0.5 * (x[l][0] * f[l][0] + x[l][1] * f[l][1] + x[l][2] * f[l][2]);
        }

        f[l][0] -= sum_x * ff;
        f[l][1] -= sum_y * ff;
        f[l][2] -= sum_z * ff;
    }

    if (est_options[PRIMITIVE]) prim_estimator();

    spring_energy = V[nbosons];
}

/* ---------------------------------------------------------------------- */

void FixPIMDBNVT::spring_force_first_bead()
{
    double** x = atom->x;
    double** f = atom->f;

    double* x_first_bead = *x;
    double* x_last_bead = buf_beads[x_last];

    if (est_options[VIRIAL]) virial = 0.0;

    // Prepare the images for potential unwrapping (for the centroid estimator).
    double unwrap[3];
    imageint* image = atom->image;
   
    if (est_options[CENTROID_VIR]) centroid_vir = 0.0;
    if (est_options[GLOB_CENTROID_VIR]) glob_centroid_vir = 0.0;

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

        /*
        // Energy estimators
        if (est_options[GLOB_CENTROID_VIR]) {
            // Global centroid-virial kinetic energy estimator calculation (without the constant and the potential energy terms).
            // Useful for translationally-invariant (periodic) bosonic systems.
            
            domain->unmap(x[l], image[l], unwrap);

            double diff_x = unwrap[0] - glob_centroid[0];
            double diff_y = unwrap[1] - glob_centroid[1];
            double diff_z = unwrap[2] - glob_centroid[2];

            // domain->minimum_image(diff_x, diff_y, diff_z);  // Probably not needed?

            glob_centroid_vir += -0.5 * (diff_x * f[l][0] + diff_y * f[l][1] + diff_z * f[l][2]);
        }
        */

        // Energy estimators
        if (est_options[CENTROID_VIR] || est_options[GLOB_CENTROID_VIR]) {
            domain->unmap(x[l], image[l], unwrap);

            if (est_options[CENTROID_VIR]) {
                // Standard centroid-virial kinetic energy estimator calculation (without the constant and the potential energy terms).
                // Useful for translationally-invariant (periodic) distinguishable systems.

                double diff_x = unwrap[0] - centroids[3 * l];
                double diff_y = unwrap[1] - centroids[3 * l + 1];
                double diff_z = unwrap[2] - centroids[3 * l + 2];

                // domain->minimum_image(diff_x, diff_y, diff_z);  // Probably not needed if we use unmap?

                centroid_vir += -0.5 * (diff_x * f[l][0] + diff_y * f[l][1] + diff_z * f[l][2]);
            }

            if (est_options[GLOB_CENTROID_VIR]) {
                // Global centroid-virial kinetic energy estimator calculation (without the constant and the potential energy terms).

                double diff_x = unwrap[0] - glob_centroid[0];
                double diff_y = unwrap[1] - glob_centroid[1];
                double diff_z = unwrap[2] - glob_centroid[2];

                // domain->minimum_image(diff_x, diff_y, diff_z);  // Probably not needed if we use unmap?

                glob_centroid_vir += -0.5 * (diff_x * f[l][0] + diff_y * f[l][1] + diff_z * f[l][2]);
            }
        }

        if (est_options[VIRIAL]) {
            // Virial kinetic energy estimator.
            virial += -0.5 * (x[l][0] * f[l][0] + x[l][1] * f[l][1] + x[l][2] * f[l][2]);
        }

        f[l][0] -= sum_x * ff;
        f[l][1] -= sum_y * ff;
        f[l][2] -= sum_z * ff;
    }

    spring_energy = 0.0;
    
    if (est_options[PRIMITIVE]) primitive = 0.0;
}

/* ---------------------------------------------------------------------- */

// Primitive kinetic energy estimator for bosons.
// Corresponds to Eqns. (4)-(5) in SI of pnas.1913365116
void FixPIMDBNVT::prim_estimator()
{
  const double Boltzmann = force->boltz;
  double beta = 1.0 / (Boltzmann * nhc_temp);

  prim_est[0] = 0.0;

  for (int m = 1; m < nbosons + 1; ++m) {
    double sig = 0.0;

    // Xiong & Xiong method.
    double Elongest = std::numeric_limits<double>::max();

    for (int k = m; k > 0; k--) {
      Elongest = std::min(Elongest, get_Enk(m, k) + V[m - k]);
    }

    // double Elongest = std::min(get_Enk(m, 1) + V[m - 1], get_Enk(m, m) + V[0]); // Barak's method

    for (int k = m; k > 0; --k) {
      double E_kn_val = get_Enk(m, k);

      sig += (prim_est[m - k] - E_kn_val) * exp(-beta * (E_kn_val + V[m - k] - Elongest));
    }

    double sig_denom_m = m * exp(-beta * (V[m] - Elongest));

    prim_est[m] = sig / sig_denom_m;
  }
  
  // primitive = prim_est[nbosons];
  primitive = 0.5 * np * domain->dimension * nbosons / beta + prim_est[nbosons];
}

/* ---------------------------------------------------------------------- */

//FOR PRINTING ENERGIES AND POTENTIALS FOR PIMD-B
void FixPIMDBNVT::end_of_step()
{
  /*
    if (universe->iworld == 0) {
        //std::cout << "E1\tE12\tE2\tE123\tE23\tE3\tVB0\tVB1\tVB2\tVB3" <<std::endl;
        //#! FIELDS time E1 E2 E12 E3 E23 E123 VB1 VB2 VB3 E_ox3 Vr.bias
        std::ofstream myfile;
        // TODO: make sure that the file is created the first time, not appending to old results
        myfile.open("pimdb.log", std::ios::out | std::ios::app);

        for (int i = 0; i < nbosons * (nbosons + 1) / 2; i++) {
            myfile << E_kn[i] << " ";
        }
        for (int i = 0; i < nbosons + 1; i++) {
            myfile << V[i] << " "; // mult by 2625.499638 to go from Ha to kcal/mol
        }
        myfile << std::endl;

        myfile.close();

    }
    */
}
