"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from functools import partial
from math import isclose
import sys
from io import StringIO

# Third party imports
import pandas as pd
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt, arcsinh
from toolz import pipe, compose, first, valmap
from myastro.util import pow, k_gauss, GM, c_light
import  myastro.coord as co
import  myastro.timeutil as tc
from math import isclose, fmod
from myastro.coord import polarFcartesian, make_ra, make_lat, mtx_eclip_prec, mtx_eclipFequat, cartesianFpolar
from myastro.timeutil import PI, reduce_rad, TWOPI, PI_HALF, epochformat2jd, jd2mjd, MDJ_J2000, CENTURY, T_J2000, mjd2jd
from myastro.planets import g_xyz_equat_sun_j2000, g_rlb_eclip_sun_eqxdate
from myastro.gauss_lagrange_eq import solve_equation
from myastro.orbit import calc_orb_elms_from_2rs, ElemsData, find_eta, show_orbital_elements

from numpy.linalg import multi_dot, norm

from myastro.cluegen import Datum

from itertools import count



from myastro.log import get_logger
logger = get_logger(__file__.split('/')[-1])


def select (rho_a, rho_b, rho_c, n_sol, n ):
    """
    Selection of a single solution rho of the Gauss-Lagrangian equation
    
    Args:
        rho_a : First solution
        rho_b : Second solution
        rho_c : Third solution
        n_sol : Total number of solutions
        n : Number of selected solution
    Returns:
        The selected soluton rho
    """

    if (n < 1) or (n_sol < n):
        logger.error(f"Error in select n={n} and n_sol = {n_sol}")
        return
    else :
        if n == 1 :
            return rho_a 
        elif n == 2 :
            return rho_b
        elif n == 3 :
            return rho_c
        else :
            return 0.0
    

def retard (t0_mjds,  rhos) :
    """
    Light time correction and computation of time differences

    Args:
        t0_mjds : List of three time of observations (t1', t2', t3') in Modified Julian Date
        rhos : np.array with 3 elements where each element is the geocentric distance in [AU]
    
    Returns:
        t_mjds : List of times of light emittance (t1, t2, t3) in Modified Julian Date
        taus : np.array with 3 elements where each element is the scales time differences.

    """
    t_mjds = [0.0, 0.0, 0.0]
    for i in range(0,3) :
        t_mjds[i] = t0_mjds[i] - rhos[i]/c_light

    tau = np.zeros(3)
    tau[0] = k_gauss * (t_mjds[2] - t_mjds[1])
    tau[1] = k_gauss * (t_mjds[2] - t_mjds[0])
    tau[2] = k_gauss * (t_mjds[1] - t_mjds[0])

    return t_mjds, tau

    
def gauss_method(R_suns, es, t0_mjds, n_run, max_iter=100, epsilon=1e-10):

    """
    Orbit determination using Gauss's method
    Args:
        R_suns : List of three sun positions vectors in ecliptic-cartesian coordinates
                 It is a numpy 3x3 where each row is a r vector, e.g. for ith row, R_sun[i,:]
            e  : List of three observation direction unit vectors. It is a numpy 3x3 where each row
                 a e vector , e.g. for ith row, e[i,:]
       t0_mjds : List of three observation times (Modified Julian Day), It is a numpy of 3 where 
                 each value is a time reference
    Returns:
        A tuple with:
            n_sol : Number of solutions found so that the caller method can iterate.
            ElemsData  : A data class with the Orbital elements.
    """    
    # d vectors (pg 232)
    ds = [np.cross(es[1],es[2]),
          np.cross(es[2],es[0]), 
          np.cross(es[0],es[1])]

    # D matrix
    D = np.zeros((3,3))
    for i in range(0,3) :
        for j in range (0,3) :
            D[i,j] = np.dot(ds[i], R_suns[j])
    
    # Det
    det = np.dot(es[2], ds[2])

    # Direction cosine of observation unit vector with respect to the Sun
    # direction at time of second observation
    gamma = np.dot(es[1],R_suns[1])/norm(R_suns[1])

    # Time differences tau[i] and initial approximations of mu[0] and mu[2]
    tau = np.zeros(3)
    tau[0] = k_gauss * (t0_mjds[2] - t0_mjds[1])
    tau[1] = k_gauss * (t0_mjds[2] - t0_mjds[0])
    tau[2] = k_gauss * (t0_mjds[1] - t0_mjds[0])

    mu = np.zeros(3)
    mu[0] = (1.0/6.0) * tau[0]*tau[2] * (1.0+tau[0]/tau[1])
    mu[2] = (1.0/6.0) * tau[0]*tau[2] * (1.0+tau[2]/tau[1])

    rho = np.zeros(3)    
    n0 = np.zeros(3)
    n = np.zeros(3)
    eta = np.zeros(3)
    r = np.zeros((3,3))

    for i in count(0) :
        rho_old = rho[1]
        #Determine geocentric distance rho at time of second observation
        # from the Gauss-Lagrangian equation
        n0[0] = tau[0]/tau[1] 
        n0[2] = tau[2]/tau[1]
        L = - ( n0[0]*D[1,0]-D[1,1]+n0[2]*D[1,2] ) / det
        l = ( mu[0]*D[1,0] + mu[2]*D[1,2] ) / det        
        rho_min, rho_mean, rho_max, n_sol = solve_equation(gamma, norm(R_suns[1]), L, l )
        print(f" Iter:{i}, n_sol: {n_sol}, {rho_min}  {rho_mean}  {rho_max}")

        rho[1] = select(rho_min, rho_mean, rho_max, n_sol, n_run)
        r[1,:] = rho[1]*es[1] - R_suns[1]

        # Compute n1 and n3
        n[0] = n0[0] + mu[0]/pow(norm(r[1,:]),3)
        n[2] = n0[2] + mu[2]/pow(norm(r[1,:]),3)

        # Geocentric distances rho_1 and rho_3 from n_1 and n_3
        rho[0] = ( n[0]*D[0,0] - D[0,1] + n[2]*D[0,2] ) / (n[0]*det)
        rho[2] = ( n[0]*D[2,0] - D[2,1] + n[2]*D[2,2] ) / (n[2]*det)

        # Apply light time corrections and compute scaled time differences
        # Retard
        t_mjds, tau = retard(t0_mjds, rho)

        # Heliocentric coordinate vector
        for j in range(3) :
            r[j,:] = rho[j]*es[j] - R_suns[j]
        
        # Sector/triangle ratios eta_i
        eta[0] = find_eta ( r[1,:], r[2,:], tau[0] )
        eta[1] = find_eta ( r[0,:], r[2,:], tau[1] )
        eta[2] = find_eta ( r[0,:], r[1,:], tau[2] )

        # Improved values of mu_1, mu_3
        mu[0] = ( eta[1]/eta[0] - 1.0 ) * (tau[0]/tau[1]) * pow(norm(r[1,:]),3)
        mu[2] = ( eta[1]/eta[2] - 1.0 ) * (tau[2]/tau[1]) * pow(norm(r[1,:]),3)    

        if isclose(rho[1], rho_old, abs_tol=epsilon) :  
            break

    if i == max_iter :
        logger.error(f"Not converged after {max_iter} iterations and epsilon {epsilon}")
        return 
    
    # Because the distances has been calculated, they are printed 
    rows=[]
    for j in range(3) :
        row = {}
        row['Geocentric rho [AU]'] = f"{rho[j]}"
        row['Heliocentric r [AU]'] = f"{norm(r[j,:])}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.T
    df.columns=['1 Obs.','2 Obs.','3 Obs.']
    print (df)
    print ()

    return n_sol, calc_orb_elms_from_2rs(GM, r[0,:], t_mjds[0], r[2,:], t_mjds[2])

EXP_DT = {
          # "year" : object,
          # "month"  : object,
          # "day"  : object,
          # "fh" : object,
           "ra_h" : object,
           "ra_m" : object,
           "ra_s" : object,
           "dec_dg" : object,
           "dec_min" : object,
           "dec_sec" : object }    

def read_body_points(body_observs):
    df = pd.read_csv(body_observs, sep="\s+", dtype= EXP_DT) 
    df['jd'] = df.apply(lambda row: tc.datetime2jd(row['year'], row['month'], row['day'], hour=row['fh']), axis=1)
    df['mjd'] = df['jd'].map(tc.jd2mjd)
    df['ra'] = df['ra_h']+'h'+df['ra_m']+'m'+df['ra_s']+'s'
    df['ra'] = df['ra'].map(co.make_ra)
    df['dec'] = df['dec_dg']+'°'+df['dec_min']+'m'+df['dec_sec']+'s'
    df['dec'] = df['dec'].map(co.make_lat)
    cols =['jd','mjd','ra','dec']
    return df[cols].copy()
    

CERES_OBSERVS = StringIO("""year month day fh ra_h ra_m ra_s dec_dg dec_min dec_sec
1805 09 05 24.165 6 23 57.54 22 21 27.08 
1806 01 17 22.095 6 45 14.69 30 21 24.20 
1806 05 23 20.399 8 07 44.60 28 02 47.04""")

ORKISZ_OBSERVS = StringIO("""year month day fh ra_h ra_m ra_s dec_dg dec_min dec_sec
1925 04 05 2.786 22 26 43.51 16 37 16.00 
1925 04 08 2.731 22 29 42.90 19 46 25.10 
1925 04 11 2.614 22 32 55.00 23 04 52.30""")


def main (observs, T_eqx0=0, T_eqx=0):
    """
    Print the obtital elements obtained from the observations

    Args:
        observs :  Datafrane with the 3 observations
                 
        T_eqx0  : Equinox in centuries of the observations
                 
          T_eqx : Equinox in centuries when we need the prediction
                 
    Returns:
        None            
    """        
    MTX_Teqx0_Teqx = mtx_eclip_prec(T_eqx0,T_eqx)
    g_rlb_eclip_bodys = []
    t_jds = []
    g_xyz_eclip_suns = []
    df = read_body_points(observs)

    for row in df.itertuples(index=False):
        g_rlb_eclip_bodys.append(pipe(np.array([1,row.ra,row.dec]),
                            cartesianFpolar,
                            MTX_Teqx0_Teqx.dot(mtx_eclipFequat(T_eqx)).dot,
                            polarFcartesian))
        t_jds.append(row.jd)
        T = (row.mjd - MDJ_J2000)/CENTURY
        g_xyz_eclip_suns.append( pipe (g_rlb_eclip_sun_eqxdate(mjd2jd(row.mjd), tofk5=True) , 
                                       cartesianFpolar, 
                                       mtx_eclip_prec(T,T_eqx0).dot))
    
    # Print observations
    print (f"Julian Day       Solar Longitud [deg]    Body Longitude [deg]    Body Latitude [deg] ")
    for i in range(0,3):
        print (f"{t_jds[i]}   {rad2deg(polarFcartesian(g_xyz_eclip_suns[i])[1]):03.6f}            {rad2deg(g_rlb_eclip_bodys[i][1])}        {rad2deg(g_rlb_eclip_bodys[i][2])}")

    t_mjds = [jd2mjd(t) for t in t_jds]
    g_xyx_eclip_bodys = [cartesianFpolar(g_rlb) for g_rlb in g_rlb_eclip_bodys]

    for n_run in count(1):
        # In the run 1, the solution 1 will be used
        # In the run 2, the solution 2 will be used (in case it exist)
        # In the run 3, the solution 3 will be used (in case it exist)
        n_sol, elems = gauss_method(g_xyz_eclip_suns, g_xyx_eclip_bodys,  t_mjds, n_run)        
        show_orbital_elements(elems)
        if n_run >= n_sol :
            break 

def ceres():
    T_eqx0 = (1806 - 2000)/100.0
    T_eqx = (1806 - 2000)/100.0    
    main (CERES_OBSERVS, T_eqx0, T_eqx)


def orkisz():
    T_eqx0 = (1925 - 2000)/100.0
    T_eqx = (1925 - 2000)/100.0
    main (ORKISZ_OBSERVS, T_eqx0, T_eqx)



if __name__ == "__main__" :
    #orkisz()
    ceres()
    
    
