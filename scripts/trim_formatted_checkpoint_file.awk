#!/usr/bin/awk -f
# reduce size of .fchk file
#
# Formatted checkpoint files contain lots of information, but not all of it
# is necessary as input for `semi`.
# This script extracts only the relevant fields and writes them to stdout.
#
# Usage:
#   $ trim_formatted_checkpoint_file.awk   fchk_large.fchk   >  fchk_small.fchk
#

NR <= 2 {print}
/^[a-zA-Z0-9]/                  {if (s == 1) {s=0}} s
/Number of atoms/               {s=1; print; next} 
/Atomic numbers/                {s=1; print; next}
/Real atomic weights/           {s=1; print; next}
/Total Energy/                  {s=1; print; next}
/Current cartesian coordinates/ {s=1; print; next}
/Cartesian Gradient/            {s=1; print; next}
/Cartesian Force Constants/     {s=1; print; next}
/Nonadiabatic coupling/         {s=1; print; next}
/Gaussian Version/              {s=1; print; next}
