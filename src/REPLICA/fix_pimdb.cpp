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
}

/* ---------------------------------------------------------------------- */

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

//dE_n^(k) is a function of k atoms (R_n-k+1,...,R_n) for a given n and k.
std::vector<double> FixPIMDB::Evaluate_dEkn_on_atom(const int n, const int k, const int atomnum)
{
  //dE_n^(k)(R_n-k+1,...,R_n) is a function of k atoms
  if (atomnum < n-k or atomnum > n-1 ) { return std::vector<double>(3, 0.0); }
  else {

    //bead is the bead number of current replica. bead = 0,...,np-1.
    int bead = universe->iworld;

    double **x = atom->x;
    double *_mass = atom->mass;
    int *type = atom->type;

    //xnext is a pointer to first element of buf_beads[x_next].
    //See in FixPIMDB::comm_init() for the definition of x_next.
    //x_next is basically (bead + 1) for bead in (0,...,np-2) and 0 for bead = np-1.
    //buf_beads[j] is a 1-D array of length 3*nlocal x0^j,y0^j,z0^j,...,x_(nlocal-1)^j,y_(nlocal-1)^j,z_(nlocal-1)^j.
    double *xnext = buf_beads[x_next];
    double *xlast = buf_beads[x_last];

    //omega^2, could use fbond instead?
    double omega_sq = omega_np * omega_np;

    //dE_n^(k)(R_n-k+1,...,R_n) is a function of k atoms
    //But derivative if for atom atomnum
    xnext += 3 * (atomnum);
    xlast += 3 * (atomnum);

    //np is total number of beads
    if (bead == np - 1 && k > 1){
      atomnum == n - 1 ? (xnext-= 3*(k - 1)) : (xnext += 3);
    }

    if (bead == 0 && k > 1){
      atomnum == n-k ? (xlast+= 3*(k - 1)) : (xlast -= 3);
    }

    std::vector<double> res(3);

    double delx1 = xnext[0] - x[atomnum][0];
    double dely1 = xnext[1] - x[atomnum][1];
    double delz1 = xnext[2] - x[atomnum][2];
    domain->minimum_image(delx1, dely1, delz1);

    double delx2 = xlast[0] - x[atomnum][0];
    double dely2 = xlast[1] - x[atomnum][1];
    double delz2 = xlast[2] - x[atomnum][2];
    domain->minimum_image(delx2, dely2, delz2);

    double dx = -1.0*(delx1 + delx2);
    double dy = -1.0*(dely1 + dely2);
    double dz = -1.0*(delz1 + delz2);

    res.at(0) = _mass[type[atomnum]] * omega_sq * dx;
    res.at(1) = _mass[type[atomnum]] * omega_sq * dy;
    res.at(2) = _mass[type[atomnum]] * omega_sq * dz;

    return res;
  }

}

double FixPIMDB::spring_energy_two_beads(double* x1, int l1, double* x2, int l2) {
  l1 = l1 % np;
  l2 = l2 % np;
  double delx2 = x2[3 * l2 + 0] - x1[3 * l1 + 0];
  double dely2 = x2[3 * l2 + 1] - x1[3 * l1 + 1];
  double delz2 = x2[3 * l2 + 2] - x1[3 * l1 + 2];
  domain->minimum_image(delx2, dely2, delz2);

  double ff = fbond * atom->mass[atom->type[l1]]; // TODO: compare to l2
  return -0.5 * ff * (delx2 * delx2 + dely2 * dely2 + delz2 * delz2);
}

/* ---------------------------------------------------------------------- */

void FixPIMDB::evaluate_cycle_energies()
{
  double* intra_atom_spring_local;
  double* separate_atom_spring;

  memory->create(intra_atom_spring_local, nbosons, "FixPIMDB::evaluate_cycle_energies");
  memory->create(separate_atom_spring, nbosons, "FixPIMDB::evaluate_cycle_energies");

  double **x = atom->x;
  for (int i = 0; i < nbosons; i++) {
    intra_atom_spring_local[i] = spring_energy_two_beads(*x, i, buf_beads[x_next], i);
  }

  // TODO: enough to communicate to replicas 0,np-1
  MPI_Allreduce(intra_atom_spring_local, separate_atom_spring, nbosons,
                MPI_DOUBLE, MPI_SUM, universe->uworld);

  memory->destroy(intra_atom_spring_local);

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

    for (int v = 0; v < nbosons; v++) {
      set_Enk(v + 1, 1, separate_atom_spring[v]);

      for (int u = v - 1; u >= 0; u--) {
        double val = get_Enk(v + 1, v - u) +
            // Eint(u)
            separate_atom_spring[u] - spring_energy_two_beads(x_first_bead, u, x_last_bead, u)
            // connect u to u+1
            + spring_energy_two_beads(x_last_bead, u, x_first_bead, u + 1)
            // break cycle [u+1,v]
            - spring_energy_two_beads(x_first_bead, u + 1, x_last_bead, v)
            // close cycle from v to u
            + spring_energy_two_beads(x_first_bead, u, x_last_bead, v);

        set_Enk(v + 1, v - u + 1, val);
      }
    }
  }

  memory->destroy(separate_atom_spring);
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

std::vector<std::vector<double>>
FixPIMDB::Evaluate_dVBn(const std::vector<double> &V, const int n) {

  const double Boltzmann = force->boltz;
  double beta   = 1.0 / (Boltzmann * nhc_temp);
  int bead = universe->iworld;
  double **f = atom->f;
  double **x = atom->x;

  std::vector<std::vector<double>> dV_all(n, std::vector<double>(3,0.0));

  virial = 0.0;
  for (int atomnum = 0; atomnum < nbosons; ++atomnum) {

      std::vector<std::vector<double>> dV(n+1, std::vector<double>(3,0.0));
      dV.at(0) = {0.0,0.0,0.0};

      for (int m = 1; m < n + 1; ++m) {

        std::vector<double> sig(3,0.0);

        if (atomnum > m-1) {
          dV.at(m) = {0.0,0.0,0.0};
        }else{

	  double Elongest = std::min((get_Enk(m,1)+V.at(m-1)), (get_Enk(m,m)+V.at(0)));

            for (int k = m; k > 0; --k) {
                std::vector<double> dE_kn(3,0.0);

                dE_kn = Evaluate_dEkn_on_atom(m,k,atomnum);
                
                sig.at(0) += (dE_kn.at(0) + dV.at(m - k).at(0)) * exp(-beta * (get_Enk(m, k) + V.at(m - k)-Elongest));
                sig.at(1) += (dE_kn.at(1) + dV.at(m - k).at(1)) * exp(-beta * (get_Enk(m, k) + V.at(m - k)-Elongest));
                sig.at(2) += (dE_kn.at(2) + dV.at(m - k).at(2)) * exp(-beta * (get_Enk(m, k) + V.at(m - k)-Elongest));
            }

            double  sig_denom_m = (double)m*exp(-beta*(V.at(m)-Elongest));

            dV.at(m).at(0) = sig.at(0) / sig_denom_m;
            dV.at(m).at(1) = sig.at(1) / sig_denom_m;
            dV.at(m).at(2) = sig.at(2) / sig_denom_m;

	    if(std::isinf(dV.at(m).at(0)) || std::isnan(dV.at(m).at(0))) {
	      if (universe->iworld ==0){
		std::cout << "sig_denom_m is: " << sig_denom_m << " Elongest is: " << Elongest
			  << " V.at(m) is " << V.at(m) << " beta is " << beta << std::endl;}
	      exit(0);
	    }

        }


      }

      dV_all.at((atomnum)).at(0) = dV.at(n).at(0);
      dV_all.at((atomnum)).at(1) = dV.at(n).at(1);
      dV_all.at((atomnum)).at(2) = dV.at(n).at(2);

      virial = virial -0.5*(x[atomnum][0]*f[atomnum][0] + x[atomnum][1]*f[atomnum][1] + x[atomnum][2]*f[atomnum][2]);

      f[atomnum][0] -= dV.at(n).at(0);
      f[atomnum][1] -= dV.at(n).at(1);
      f[atomnum][2] -= dV.at(n).at(2);

  }

  return dV_all;

}

void FixPIMDB::Evaluate_VBn(std::vector <double>& V, const int n)
{
  const double Boltzmann = force->boltz;
  double beta   = 1.0 / (Boltzmann * nhc_temp);

  for (int m = 1; m < n+1; ++m) {
    double sig_denom = 0.0;
    double Elongest=0.0;

    Elongest = std::min((get_Enk(m,1)+V.at(m-1)), (get_Enk(m,m)+V.at(0)));
    for (int k = m; k > 0; --k) {
          double E_kn;

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

  for (int l = nbosons - 1; l >= 0; l--) { // TODO: case of 0.0 can be taken from V forward
    double Elongest = std::min(get_Enk(l + 1, 1) + V_backwards[l+1], get_Enk(nbosons, nbosons - l));

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
      std::vector<std::vector<double>> dV(nbosons * universe->nworlds, std::vector<double>(3, 0.0));

      Evaluate_VBn(V, nbosons);

      double* V_backwards;
      memory->create(V_backwards, nbosons + 1, "FixPIMDB::spring_force");
      Evaluate_V_backwards(V_backwards);

      dV = Evaluate_dVBn(V, nbosons);

      memory->destroy(V_backwards);
    }
}

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