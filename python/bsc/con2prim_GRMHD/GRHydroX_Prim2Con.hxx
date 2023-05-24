#ifndef GRHYDROX_PRIM2CON_HXX
#define GRHYDROX_PRIM2CON_HXX

#include "GRHydroX_var_groups.hxx"

#include <cmath>

namespace GRHydroX {
using namespace std;

// Total flops = 206
// prim2con on a single point (MHD case)
static inline CCTK_DEVICE CCTK_HOST cons_point
prim2con(const metric_point &metric_point, const prim_point &prim_point) {
  // local variable assigmments from struct prim_point
  const CCTK_REAL rho = prim_point.rho;
  const CCTK_REAL vx = prim_point.velx;
  const CCTK_REAL vy = prim_point.vely;
  const CCTK_REAL vz = prim_point.velz;
  const CCTK_REAL eps = prim_point.eps;
  const CCTK_REAL press = prim_point.press;
  const CCTK_REAL Bx = prim_point.Bvecx;
  const CCTK_REAL By = prim_point.Bvecy;
  const CCTK_REAL Bz = prim_point.Bvecz;
  const CCTK_REAL Ye = prim_point.Ye;

  // local variable assigmments from struct metric_point
  const CCTK_REAL gxx = metric_point.gxx;
  const CCTK_REAL gxy = metric_point.gxy;
  const CCTK_REAL gxz = metric_point.gxz;
  const CCTK_REAL gyy = metric_point.gyy;
  const CCTK_REAL gyz = metric_point.gyz;
  const CCTK_REAL gzz = metric_point.gzz;
  const CCTK_REAL sdet = sqrt(calculate_detg(metric_point));  // flops = 16+10 = 26

  // local helpers
  // TODO: use vlow*v
  const CCTK_REAL wtemp =
      1.0 / sqrt(1.0 - (gxx * vx * vx + gyy * vy * vy + gzz * vz * vz +
                        2.0 * gxy * vx * vy + 2.0 * gxz * vx * vz +
                        2.0 * gyz * vy * vz)); // flops = 22 + 10 + 10 = 42

  const CCTK_REAL vlowx = gxx * vx + gxy * vy + gxz * vz;
  const CCTK_REAL vlowy = gxy * vx + gyy * vy + gyz * vz;
  const CCTK_REAL vlowz = gxz * vx + gyz * vy + gzz * vz;

  const CCTK_REAL Bxlow = gxx * Bx + gxy * By + gxz * Bz;
  const CCTK_REAL Bylow = gxy * Bx + gyy * By + gyz * Bz;
  const CCTK_REAL Bzlow = gxz * Bx + gyz * By + gzz * Bz;

  const CCTK_REAL B2 = Bxlow * Bx + Bylow * By + Bzlow * Bz; // flops = 35

  const CCTK_REAL Bdotv = Bxlow * vx + Bylow * vy + Bzlow * vz;
  const CCTK_REAL Bdotv2 = Bdotv * Bdotv;
  const CCTK_REAL wtemp2 = wtemp * wtemp;
  const CCTK_REAL b2 = B2 / wtemp2 + Bdotv2;
  const CCTK_REAL ab0 = wtemp * Bdotv; // flops = 19
  const CCTK_REAL blowx =
      (gxx * Bx + gxy * By + gxz * Bz) / wtemp + wtemp * Bdotv * vlowx; //TODO:vlowx doesn't reconcile with eq(21). Missing factor of betax/alp?
  const CCTK_REAL blowy =
      (gxy * Bx + gyy * By + gyz * Bz) / wtemp + wtemp * Bdotv * vlowy;
  const CCTK_REAL blowz = (gxz * Bx + gyz * By + gzz * Bz) / wtemp +
                          wtemp * Bdotv * vlowz; // flops = 18*3 = 54

  const CCTK_REAL hrhow2 = (rho * (1.0 + eps) + press + b2) * (wtemp) * (wtemp);
  const CCTK_REAL denstemp = sdet * rho * (wtemp); // flops = 8

  cons_point cons;
  cons.dens = denstemp;
  cons.sx = sdet * (hrhow2 * vlowx - ab0 * blowx);
  cons.sy = sdet * (hrhow2 * vlowy - ab0 * blowy);
  cons.sz = sdet * (hrhow2 * vlowz - ab0 * blowz);
  cons.tau = sdet * (hrhow2 - press - b2 / 2.0 - ab0 * ab0) - denstemp;
  cons.Bconsx = sdet * Bx;
  cons.Bconsy = sdet * By;
  cons.Bconsz = sdet * Bz; // flops = 22
  cons.Ye_cons = Ye * denstemp; 
  return cons;
}

} // namespace GRHydroX

#endif // #ifndef GRHYDROX_PRIM2CON_HXX
