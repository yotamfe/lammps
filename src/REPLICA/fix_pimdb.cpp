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

FixPIMDB::FixPIMDB(LAMMPS *lmp, int narg, char **arg) :
    FixPIMD(lmp, narg, arg),
    bosonic_exchange(lmp, atom->nlocal, np, universe->me,
                     1.0 / (force->boltz * nhc_temp))
{
  if (method == CMD) {
    error->universe_all(FLERR, "Method cmd not supported in fix pimdb");
  }

  nbosons    = atom->nlocal;
  nevery     = 100; // TODO: make configurable (thermo_style?)
}

/* ---------------------------------------------------------------------- */

FixPIMDB::~FixPIMDB() {
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

void FixPIMDB::spring_force() {
    double ff = fbond * atom->mass[atom->type[0]]; // TODO: ensure that all masses are the same
    bosonic_exchange.init(*(atom->x), buf_beads[x_last], buf_beads[x_next], ff);

    if (universe->me != 0 && universe->me != np - 1) {
        // interior beads
        FixPIMD::spring_force();
        spring_energy = 0.0;
    } else {
        // exterior beads
        virial = bosonic_exchange.spring_force(atom->f);
        if (universe->me == np - 1) {
            spring_energy = bosonic_exchange.get_potential();
        } else {
            spring_energy = 0.0;
        }
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

      for (int i = 0; i < nbosons * (nbosons + 1) / 2; i++) {
          myfile << bosonic_exchange.get_E_kn_serial_order(i) << " ";
      }
      for (int i = 0; i < nbosons + 1; i++) {
          myfile << bosonic_exchange.get_Vn(i) << " "; // mult by 2625.499638 to go from Ha to kcal/mol
      }
      myfile << std::endl;

      myfile.close();

    }

}