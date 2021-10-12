"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from datetime import datetime
from functools import partial
from collections import namedtuple
from math import isclose
import sys

# Third party imports
import pandas as pd
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt
from numpy.linalg import multi_dot, norm
from toolz import pipe, compose, first

# Local application imports

from myastro import timeutil as tc
from myastro  import coord as co

from myastro import orbit as ob
from myastro import data_catalog as dc
#from myastro import perturb as per
from myastro  import util as ut
from myastro.coord import as_str, Coord, cartesianFpolar, polarFcartesian,EQUAT2_TYPE


from myastro.coord import as_str, Coord, cartesianFpolar, polarFcartesian, EQUAT2_TYPE, mtx_eclip_prec, mtx_equatFeclip, format_time, format_dg

from myastro.timeutil import epochformat2jd, jd2mjd, T, mjd2jd, jd2str_date,MDJ_J2000, JD_J2000, dg2h, h2hms, dg2dgms, CENTURY, T_J2000, reduce_rad
from myastro.planets import g_xyz_equat_sun_j2000, g_rlb_eclip_sun_eqxdate, h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000, g_rlb_equat_pluto_j2000, g_rlb_equat_planet_J2000

#from myastro.orbit import h_xyz_eclip_keplerian_orbit

from myastro.keplerian import KeplerianOrbit


from myastro.log import get_logger
logger = get_logger(__file__.split('/')[-1])


def calc_eph_planet(name, eph):
    """
    Computes the ephemris for a comet

    Args:
        body : The orbital elements of the body (CometElms)
        eph : The ephemeris data

    Returns :
        A dataframe with the epehemris calculated.
    """
    rows = []
    for clock_mjd in ut.frange(eph.from_mjd, eph.to_mjd, eph.step):        
        row = {}
        clock_jd = mjd2jd(clock_mjd)
        row['date'] = jd2str_date(clock_jd)
        row['t_mjd'] = clock_mjd

        T = (clock_mjd - MDJ_J2000)/CENTURY    
        # desired equinox is J2000, so T_desired is 0
        T_desired = (JD_J2000 - JD_J2000)/CENTURY
        mtx_prec = co.mtx_eclip_prec(T, T_desired)

        if name == 'Pluto':
            h_xyz_eclipt = h_xyz_eclip_pluto_j2000(clock_jd)            
            g_rlb_equat = g_rlb_equat_pluto_j2000(clock_jd)
        else :
            # Planetary position (ecliptic and equinox of J2000)
            h_xyz_eclipt = mtx_prec.dot(h_xyz_eclip_eqxdate(name, clock_jd))
            g_rlb_equat = g_rlb_equat_planet_J2000(name,clock_jd)
        
        row['h_x'] = h_xyz_eclipt[0]
        row['h_y'] = h_xyz_eclipt[1]
        row['h_z'] = h_xyz_eclipt[2]

        row['ra'] = format_time(pipe(g_rlb_equat[1], rad2deg,  dg2h, h2hms))
        row['dec'] = format_dg(*pipe(g_rlb_equat[2], rad2deg, dg2dgms))
        row['r [AU]'] = norm(h_xyz_eclipt)
 
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(by='t_mjd')
    

def calc_eph_twobodys(body, eph, type='comet'):
    """
    Computes the ephemris for a comet

    Args:
        body : The orbital elements of the body (CometElms)
        eph : The ephemeris data

    Returns :
        A dataframe with the epehemris calculated.
    """

    # Normally, the equinox of the data of the body will be J2000  and the equinox of the 
    # ephemeris will also be J2000, so the precesion matrix will be the identity matrix
    # Just in the case of book of Personal Astronomy with your computer pag 81 is used   
    MTX_Teqx_PQR = mtx_eclip_prec(body.T_eqx0, eph.T_eqx).dot(body.mtx_PQR)  

    # Transform from ecliptic to equatorial just depend on desired equinox
    MTX_equatFecli = mtx_equatFeclip(eph.T_eqx)

    if type == 'comet' :
        k_orbit = KeplerianOrbit.for_comet(body)
    else :
        k_orbit = KeplerianOrbit.for_body(body)
     
    result = dict()
    for clock_mjd in ut.frange(eph.from_mjd, eph.to_mjd, eph.step):        
        r , v = k_orbit.calc_rv(clock_mjd)
        result[clock_mjd] = (MTX_Teqx_PQR.dot(r), MTX_Teqx_PQR.dot(v))
    return ob.process_solution(result, np.identity(3), MTX_equatFecli)
    
def test_1():
    HALLEY_1950 = dc.CometElms(name="1P/Halley",
                epoch_name=None ,
                q =  0.5870992 ,
                e = 0.9672725 ,
                i_dg = 162.23932 ,
                Node_dg = 58.14397 ,
                w_dg = 111.84658 ,
                tp_str = "19860209.44",
                equinox_name = "B1950")

    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)                

    eph_halley = ob.EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")

    df = calc_eph_twobodys(HALLEY_J2000, eph_halley)
    print (df[df.columns[0:8]])


def test_2():
    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODYS)
    eph = ob.EphemrisInput(from_date="2020.05.15.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    df = calc_eph_twobodys(CERES, eph, type='body')        


if __name__ == "__main__":
    test_2()
    #test_calc_eph_comet()
    #test_perturbed()
    #test_planets()
