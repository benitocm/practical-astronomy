"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from functools import partial
from collections import namedtuple
from math import isclose
import sys

# Third party imports
import pandas as pd
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt
from numpy.linalg import norm
import toolz as tz
# Using Newton-Ramson method
from scipy.optimize import newton, bisect
from scipy.integrate import solve_ivp    
from toolz import pipe, compose, keyfilter, merge
from myastro.cluegen import Datum


# Local application imports
from myastro import timeutil as tc
from myastro import coord as co
from myastro import orbit

from myastro import orbit as ob
from myastro import data_catalog as dc
from myastro import perturb as per
from myastro.coord import as_str, Coord, cartesianFpolar, polarFcartesian,EQUAT2_TYPE

from myastro  import util as ut
from myastro.coord import as_str, Coord, cartesianFpolar, polarFcartesian, EQUAT2_TYPE, mtx_eclip_prec, mtx_equatFeclip, format_time, format_dg
from myastro.data_catalog import CometElms
from myastro.timeutil import epochformat2jd, jd2mjd, T, mjd2jd, jd2str_date,MDJ_J2000, JD_J2000, dg2h, h2hms, dg2dgms, CENTURY, T_J2000, reduce_rad
from myastro.planets import g_xyz_equat_sun_j2000, g_rlb_eclip_sun_eqxdate, h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000, g_rlb_equat_pluto_j2000, g_rlb_equat_planet_J2000

from myastro.orbit import calc_M, elliptic_orbit, h_xyz_eclip_keplerian_orbit, next_E, parabolic_orbit, hyperpolic_orbit, calc_semimajor_axis, calc_perturbed_accelaration
from myastro.data_catalog import DF_BODYS, DF_COMETS, DF_PLANETS, read_body_elms_for, read_comet_elms_for, BodyElems
from myastro.orbital_mech import rv_from_r0v0, PLANETS_NAMES, mu_by_name, mu_Sun
from myastro.util import pow
from myastro.ephem import EphemrisInput

from myastro.timeutil import  PI_HALF, PI, TWOPI



from myastro.log import get_logger
logger = get_logger(__file__.split('/')[-1])

def calc_M_for_body(t_mjd, epoch_mjd, period_in_days, M_at_epoch) :
    """ 
    Computes the mean anomaly based on the data of BodyElms, in this case,
    uses the period (calculated) and the Mean anomaly at epoch.
    For Body Elements

    Args:
        t_mjd : time of the computation
        t0: a point in time where Mean anomaly is knwon
        M0: the mean anomaly in radians at t0
        period : period of the orbit in days.

    Returns :
        The mean anomaly in radians
    """
    M = (t_mjd - epoch_mjd)*TWOPI/period_in_days
    M += M_at_epoch
    return reduce_rad(M,to_positive=True)     


def calc_M_for_comet(t_mjd, tp_mjd, inv_a) -> float :
    """ 
    Computes the mean anomaly as a function a t, t0 and a, i.e., not depending on the
    period of the orbit=1) and semi-major axis

    Args:
        t : time of the computation in Modified Julian Dates
        t0: Time of perihelion passage 
        a: semi-major axis in AU

    Returns :
        The mean anomaly in radians
    """
    #M = sqrt(ut.GM)*(t_mjd-tp_mjd)/np.float_power(a, 1.5)
    M = sqrt(ut.GM)*(t_mjd-tp_mjd)*sqrt(ut.pow(inv_a,3))
    return reduce_rad(M, to_positive=True)
    

M_min = 0.1
class KeplerianOrbit:

    def __init__(self, epoch_mjd,  q, a, e, tp_mjd, M_at_epoch) :
        self.epoch_mjd = epoch_mjd
        self.tp_mjd = tp_mjd
        self.q = q
        self.e = e
        self.a = a
        self.M_at_epoch = M_at_epoch        

    @classmethod
    def for_body(cls, body_elms):    
        return cls(body_elms.epoch_mjd, None, body_elms.a, body_elms.e, body_elms.tp_mjd, body_elms.M0)

    @classmethod
    def for_comet(cls, comet_elms):    
        return cls( comet_elms.epoch_mjd, comet_elms.q, None, comet_elms.e, comet_elms.tp_mjd, None)

    def calc_rv(self, t_mjd) :
        if self.a is not None :
            ## This is a body
            period_in_days = TWOPI*sqrt(ut.pow(self.a,3)/ut.GM)
            M = calc_M_for_body (t_mjd, self.epoch_mjd, period_in_days, self.M_at_epoch)
        else :
            # This is a comet
            # The inv_a is calculated to avoid to divide by 0 in parabolic
            inv_a = np.abs(1.0-self.e)/self.q
            M = calc_M_for_comet(t_mjd, self.tp_mjd, inv_a)
            # This is a comet

        if ((M < M_min) and (np.abs(1.0-self.e) < 0.1)) or isclose(self.e, 1.0, abs_tol=1e-04) :
            logger.warning(f'Doing parabolic orbit for e: {self.e}')
            xyz, vxyz = parabolic_orbit(self.tp_mjd, self.q, self.e, t_mjd, 50)
        elif self.e < 1.0 :
            a = self.q/(1.0-self.e) if self.a is None else self.a
            logger.warning(f'Doing elliptic orbit for e: {self.e}')
            xyz, vxyz = elliptic_orbit(ob.next_E, M, a, self.e)
        else :
            logger.warning(f'Doing hyperbolic orbit for e: {self.e}')
            a = self.q/np.abs(1.0-self.e) if self.a is None else self.a
            xyz, vxyz =  hyperpolic_orbit (self.tp_mjd, ob.next_H, a, self.e, t_mjd)
        return xyz, vxyz
      

def planet_rv(pl_name, t_mjd) :

    """ 
    Computes position (r) and velocity (v) vectors of planet at the time 
    t in Modified Julian Day. The caller will have to convert to the desired epoch

    Args:
        pl_name : Name of the planet
        t_mjd : time in modified Julian Days
        
    Returns :
        Position (r) w.r.t. the orbital plane
        Velocity (v) w.r.t  the orbital plane
    """

    # The century T corresponding to the time t. Also used to calculate
    # precession matrix
    T = (t_mjd - MDJ_J2000)/36525.0    

    row = DF_PLANETS[DF_PLANETS.name.str.contains(pl_name)]
    # Asumed 0 because the orbital elements are the ones at 2000 January 1.5
    # i.e. T0 (centuries from J2000) is 0 
    T0 = 0 
    row = row.to_dict('records')[0]
    M = row['M']+(row['n']*(T-T0))
    M = reduce_rad(M, to_positive=True)
    return elliptic_orbit (next_E, M, row['a'], row['e'])

def f1(vector):
    return vector/pow(norm(vector),3)

def calc_perturbed_acc (t_mjd, r_sun_body) :
    """
    Computes the acceleration vector for a minor body in the solar system at one point
    time.

    Args:
        t_mjd :  point in time in Modified Julian days
        r_sun_body : radio vector of the body (w.r.t the Sun) in the orbital plane

    Returns :
        The perturbed acceleration vector
    """
    #return 0

    acc = 0
    for pl_name in PLANETS_NAMES:
        r_q , _ = planet_rv (pl_name, t_mjd) # r_sun_planet
        p_q   = r_q - r_sun_body             # r_body_planet
        acc +=  mu_by_name[pl_name] * (f1(p_q)-f1(r_q)) 
    return acc


def calc_F(a, b ,c):
    q = a * (2*b-a)/pow(b,2)
    return (pow(q,2)- 3*q +3 / (1+pow(1-q,1.5)))*q


def my_f(t, y, r0, v0, t0):
    delta_r = y[0:3]    
    r_osc, _ = rv_from_r0v0(mu_Sun, r0, v0, t-t0)    
    r_pert = r_osc + delta_r
    F = 1 - pow(norm(r_osc)/norm(r_pert),3)
    #todo review 
    #F = calc_F(norm(delta_r), norm(r_pert), norm(r_osc))
    #delta_acc = (-mu_Sun/pow(norm(r_osc),3))*(delta_r- F*r_pert)+calc_perturbed_acc(t, r_pert)
    delta_acc = (-mu_Sun/pow(norm(r_osc),3))*(delta_r- F*r_pert)+calc_perturbed_accelaration(t, r_pert)    
    return np.concatenate((y[3:6],delta_acc))



OrbitPoint = namedtuple('OrbitPoint',['t','r','v'])
"""
Named tuple to hold time, r vector and v vector
"""

def apply_enke(eph, t_range, r0, v0):
    steps = np.diff(t_range)
    result = dict()
    clock_mjd = t_range[0]
    for idx, step in enumerate(steps) :
        if (idx % 50) == 0 :
            print (f"Iteration: {idx},  Date : {jd2str_date(tc.mjd2jd(clock_mjd))}")        
        sol = solve_ivp(my_f, (clock_mjd, clock_mjd+step), np.zeros(6), args=(r0, v0, clock_mjd) , rtol = 1e-12)  
        assert sol.success, "Integration was not OK!"
        r_osc, v_osc = rv_from_r0v0 (mu_Sun, r0, v0, step)
        # The last integration value is taken
        r0 = r_osc + sol.y[:,-1][:3]
        v0 = v_osc + sol.y[:,-1][3:6]
        if eph.from_mjd <= clock_mjd+step <= eph.to_mjd :
            result[clock_mjd+step] = (r0, v0)
        clock_mjd += step    
    return result


def calc_eph_by_enke (body, eph, type='body'):
    """
    Computes the ephemeris for a minor body 

    Args:
        body : The orbital elements of the body (CometElms)
        eph : The ephemeris data
        include_osc : Flag to include or not the osculating 

    Returns :
        A dataframe with the epehemris calculated.
    """

    # This matrix just depends on the desired equinox to calculate the obliquity
    # to pass from ecliptic coordinate system to equatorial
    MTX_equatFeclip = co.mtx_equatFeclip(eph.T_eqx)

    T_J2000 = 0.0    
    # This is to precess from J2000 to ephemeris equinox (normally this matrix will be identity matrix)
    MTX_J2000_Teqx = co.mtx_eclip_prec(T_J2000,eph.T_eqx)

    # The PQR mtx (from orbital plane to eclipt) is preccesed from equinox of body object to the desired equinox 
    MTX_J2000_PQR = co.mtx_eclip_prec(body.T_eqx0, T_J2000).dot(body.mtx_PQR)
    
    # The initial conditions for doing the integration is calculated, i.e.,
    # the r,v of the body at its epoch (in the example of Ceres, the epoch of 
    # book that epoch  is 1983/09/23.00)
    # The integration is done in the ecliptic plane and precessed in to J2000
    # so the solution will be also ecliptic and precessed.

    initial_mjd = body.epoch_mjd  

    if type == 'body' :
        k_orbit = KeplerianOrbit.for_body(body)
    else :
        k_orbit = KeplerianOrbit.for_comet(body)

    r0, v0 = k_orbit.calc_rv(initial_mjd)

    # In the ecliptic.
    r0 = MTX_J2000_PQR.dot(r0)
    v0 = MTX_J2000_PQR.dot(v0)

    if eph.from_mjd < initial_mjd < eph.to_mjd :
        # If the epoch is in the middle, we need to integrate forward  and backwards
        t_range = list(ut.frange(initial_mjd, eph.to_mjd, eph.step))
        result_1 = apply_enke(eph, t_range, r0, v0)

        # and backwards 
        t_range = list(ut.frange(eph.from_mjd, initial_mjd, eph.step))
        if t_range[-1] != initial_mjd :
            t_range.append(initial_mjd)
        result_2 = apply_enke(eph, list(reversed(t_range)), r0, v0)
        result = merge([result_1,result_2])        
        
    elif initial_mjd < eph.from_mjd :
        # If the epoch is in the past, we need to integrate forward
        t_range_1 = list(ut.frange(initial_mjd, eph.from_mjd, eph.step))
        # The previous ensure that initial_mjd is included but the eph.from  may be not included
        # so we test the final value to know if we need to include manually
        if t_range_1[-1] != eph.from_mjd :
            t_range_1.append(eph.from_mjd)

        t_range_2 = list(ut.frange(eph.from_mjd+eph.step, eph.to_mjd, eph.step))
        if len(t_range_2) == 0 :
            t_range_2.append(eph.to_mjd)
        if t_range_2[-1] != eph.to_mjd :
            t_range_2.append(eph.to_mjd)

        t_range = t_range_1 + t_range_2 
        result = apply_enke(eph, t_range, r0, v0)
    else :
        # If the epoch is in the future, we need to integrate backwards
        # goes from the epoch backward toward the end value from 
        # the ephemeris and inital value of the ephemeris

        t_range_1 = list(ut.frange(eph.to_mjd, initial_mjd, eph.step))
        # The previous ensure that eph.to is included but the initial may be not included
        # so we test the final value to know if we need to include manually
        if t_range_1[-1] != initial_mjd :
            t_range_1.append(initial_mjd)
        
        t_range_2 = list(ut.frange(eph.from_mjd, eph.to_mjd, eph.step))
        # the previous ensure that eph.from is included but the to_mjd may be included
        # but  we include in the previous so we need to remove it . We test the last element to check
        # if we need to remove it
        if t_range_2[-1] == eph.to_mjd :
            t_range_2 = t_range_2[0:-1]
        t_range = list(reversed(t_range_1)) + list(reversed(t_range_2))
        result = apply_enke(eph, t_range, r0, v0)


    result = {t:result[t] for t in sorted(result.keys())}
        
    """
    # Once we have the t_range we calculate the steps between time samples 
    # because the number of interations is exactly the number of steps, i.e., len(trange)-1
    steps = np.diff(t_range)

    result = dict()
    clock_mjd = t_range[0]
    for idx, step in enumerate(steps) :
        if (idx % 50) == 0 :
            print (f"Iteration: {idx},  Date : {jd2str_date(tc.mjd2jd(clock_mjd))}")        
        sol = solve_ivp(my_f, (clock_mjd, clock_mjd+step), np.zeros(6), args=(r0, v0, clock_mjd) , rtol = 1e-12)  
        assert sol.success, "Integration was not OK!"
        r_osc, v_osc = rv_from_r0v0 (mu_Sun, r0, v0, step)
        # The last integration value is taken
        r0 = r_osc + sol.y[:,-1][:3]
        v0 = v_osc + sol.y[:,-1][3:6]
        if eph.from_mjd <= clock_mjd+step <= eph.to_mjd :
            result[clock_mjd+step] = (r0, v0)
        clock_mjd += step
    """
    # The result is filtered to just put ordered in time just in case of backward integration
    #result = keyfilter(lambda t: (eph.from_mjd <= t) and (t <= eph.to_mjd), result)
    # The result is ordered in time just in case of backward integration
    #result = {t:result[t] for t in sorted(result.keys())}

    return process_solution(result, MTX_J2000_Teqx, MTX_equatFeclip)

def process_solution(OrbitPoints, MTX_J2000_Teqx, MTX_equatFeclip):
    cols = ['date','Sun(dg)','h_l','h_b','h_r','ra','dec','r[AU]','h_x','h_y','h_z','t_mjd']
    rows = []
    for t, r_v in OrbitPoints.items():
        clock_mjd = t
        row = {}
        clock_jd = mjd2jd(clock_mjd)
        row['date'] = jd2str_date(clock_jd)
        row['t_mjd'] = clock_mjd
        # Because the integration is done in the ecliptic coordinate system and precessed
        # so the solution will be already ecliptic so we just need to precess it to the 
        # desired equinox before transforming it to the equatorial

        h_xyz = MTX_J2000_Teqx.dot(r_v[0])
        h_vxyz = MTX_J2000_Teqx.dot(r_v[1])

        h_xyz_equat_body = MTX_equatFeclip.dot(h_xyz)
        h_vxyz_equat_body = MTX_equatFeclip.dot(h_vxyz)

        # This is just to obtain the geo ecliptic longitud of the Sun and include in the
        # dataframe. Becasue the we have the sun position at equinox of the date,
        # we need to preccess it (in ecplitpical form) to J2000
        T = (clock_jd - JD_J2000)/CENTURY
        g_rlb_eclipt_T = pipe (g_rlb_eclip_sun_eqxdate(clock_jd, tofk5=True) , 
                               cartesianFpolar, 
                               mtx_eclip_prec(T,T_J2000).dot, 
                               polarFcartesian)
        row['Sun(dg)'] = f"{rad2deg(g_rlb_eclipt_T[1]):03.1f}"
        
        g_xyz_equat_sun = g_xyz_equat_sun_j2000(clock_jd)

        #We need to log the heliocentirc ecliptic coordinates in polar format 
        rlb = polarFcartesian(h_xyz)
        row['h_l'] = f"{rad2deg(rlb[1]):03.1f}"
        row['h_b'] = f"{rad2deg(rlb[2]):03.1f}"
        row['h_r'] = f"{rlb[0]:03.4f}"

        row['h_x'] = h_xyz[0]
        row['h_y'] = h_xyz[1]
        row['h_z'] = h_xyz[2]


        # Doing the sum of the vectors in the equaotiral planes works better for me.
        g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
        r_AU = norm(g_xyz_equat_body) 

        # The object will be closer because while the light traves to the earth
        # the object is moving. The correction for the ligth is done using
        # the aproach described in Astronomy on my computer
        # In Meeus, the methods implies to run two orbits keplerian. 
        # I prefer the first method because with perturbation we cannot execute
        # the second round needed.
        g_xyz_equat_body -= ut.INV_C*norm(g_xyz_equat_body)*h_vxyz_equat_body

        g_rlb_equat_body = polarFcartesian(g_xyz_equat_body)        

        row['ra'] = format_time(pipe(g_rlb_equat_body[1], rad2deg,  dg2h, h2hms))
        row['dec'] = format_dg(*pipe(g_rlb_equat_body[2], rad2deg, dg2dgms))
        row['r[AU]'] = r_AU
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[cols].sort_values(by='t_mjd')


def test_all():
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    for name in DF_BODYS['Name']: 
        body = read_body_elms_for(name,DF_BODYS)
        print ("Calculating for ",name)
        print (calc_eph_by_enke(body, eph)) 



def test_1():
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    CERES = read_body_elms_for("Ceres",DF_BODYS)


    result = calc_eph_by_enke(CERES, eph)

    print (result)

def test_comet():
    eph = EphemrisInput(from_date="1994.01.01.0",
                    to_date = "1994.04.01.0",
                    step_dd_hh_hhh = "3 00.0",
                    equinox_name = "J2000")

    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)


    df = calc_eph_by_enke(HALLEY_J2000, eph, 'comet')

    print (df[df.columns[0:8]])

def test_all_comets():
    """
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")
    """

    for idx, name in enumerate(dc.DF_COMETS['Name']): 
        body = read_comet_elms_for(name,dc.DF_COMETS)
        eph = EphemrisInput.from_mjds( body.epoch_mjd-50, body.epoch_mjd+50, "02 00.0", "J2000" )
        print (f"{idx+1} Calculating for {name} ")
        print (calc_eph_by_enke(body, eph, 'comet')) 

def test_several_comets():
    names = [#"C/1988 L1 (Shoemaker-Holt-Rodriquez)",   # Parabolic
             "C/1980 E1 (Bowell)"]  
             #"C/1848 P1 (Petersen)"]

    for name in names :
        body = read_comet_elms_for(name,dc.DF_COMETS)
        eph = EphemrisInput.from_mjds( body.epoch_mjd-50, body.epoch_mjd+50, "02 00.0", "J2000" )
        print (f"Calculating for {name} ")
        df = calc_eph_by_enke(body, eph, 'comet')
        print (df[df.columns[0:8]])
        #print (calc_eph_by_enke(body, eph, 'comet')) 

    

if __name__ == "__main__" :
    #test_all()
    #test_2()
    #test_all_comets()
    #test_comet()
    test_several_comets()

 



    





    

