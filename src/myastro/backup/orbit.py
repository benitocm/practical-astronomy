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
from numpy.linalg import norm
import toolz as tz
# Using Newton-Ramson method
from scipy.optimize import newton, bisect
from scipy.integrate import solve_ivp    
from toolz import pipe, compose
from myastro.cluegen import Datum


# Local application imports
from myastro import timeutil as tc
from myastro import coord as co
from myastro import planets as pl
from myastro import data_catalog as dc

from myastro.timeutil import  PI_HALF, PI, TWOPI, MDJ_J2000, JD_J2000, CENTURY, mjd2jd, reduce_rad
from myastro.coord import Coord, as_str, EQUAT2_TYPE, ECLIP_TYPE
from myastro.util import pow, GM
from myastro.data_catalog import DF_PLANETS, BodyElems
from myastro.planets import h_rlb_eclip_eqxdate, h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000


from myastro.log import get_logger
logger = get_logger(__file__.split('/')[-1])


GM_by_planet = {
    "Sun" :GM,                  
    "Mercury" : GM/6023600.0,
    "Venus" : GM/408523.5, 
    "Earth": GM/328900.5,
    "Mars": GM/3098710.0,
    "Jupiter": GM/1047.355,
    "Saturn": GM /3498.5,
    "Uranus": GM / 22869.0,
    "Neptune": GM / 19314.0,
    "Pluto": GM/3000000.0 
}

CENTENNIAL_PRECESSION_DG = 1.3970
CENTENNIAL_PRECESSION_RAD = np.deg2rad(CENTENNIAL_PRECESSION_DG)


class OrbObject:
    """Orbiting Object class

    Example class: https://github.com/jakevdp/matplotlib_pydata2013/blob/master/examples/double_pendulum.py
    init_state is [theta1, omega1, theta2, omega2] in degrees,
    where theta1, omega1 is the angular position and velocity of the first
    pendulum arm, and theta2, omega2 is that of the second pendulum arm
    """
    def __init__(self,                         
                 name,    # Name
                 e,       # excentricity 
                 a,       # semi-major axis in A.U.
                 T=0,      # A moment T where the object is in the perihelion
                 time_unit = 365
                 ): 
        self.name = name
        self.e_ = e
        self.a_ = a
        # semi-minor axis in A.U.
        self.b_ = a*np.sqrt(1 - e*e)
        # Applying the third law of Kepler, we can calculate the relative
        # orbital period of the planet relative to the period of the earth
        # so we can get it in days
        # https://pwg.gsfc.nasa.gov/stargaze/Mkepl3laws.htm
        self.P_ = np.float_power(a,1.5) * time_unit
        self.T_ = T
        # distance from the center of the elipse to the Sun (focus)
        self.f_ = a * e    
        #https://stackoverflow.com/questions/15815999/is-there-a-standard-way-to-store-xy-data-in-python
        int_p = int(self.P_)
        self.true_pos = []
        self.true_anomalies = []
        self.mean_pos= []
        self.mean_anomalies = []
        self.eccentric_pos = []
        self.eccentric_anomalies = []
        self.idx = 0
        
    def b(self) :
        return self.a_*np.sqrt(1 - self.e_*self.e_)

    def a(self) :
        return self.a_


    def step(self, t):
        """execute one time"""
        # The mean anomaly is calculate referred to the center of the elipse
        m_anomaly = calc_M_by_period(self.T_, self.P_, t)
        self.mean_anomalies.append(m_anomaly)
        xm = self.a_ * cos(m_anomaly) 
        ym = self.a_ * sin(m_anomaly)
        # Once we have mean anomaly, we calculate the eccentry anomaly
        # that is also referred to the center of the elipse so 
        # we need to refer the coordinates to the elipse
        e_anomaly = solve_ke_newton(self.e_, e_anomaly_v2, m_anomaly)
        self.eccentric_anomalies.append(e_anomaly)
        xe = self.a_ * cos(e_anomaly) 
        ye = self.a_ * sin(e_anomaly)
        # finally we calculate the true anomaly that is referred to the
        # Sun as a center, i.e, x=a and y=0 so we need also the
        # radio vector, distance of the sun to the planet
        t_anomaly = true_anomaly(self.e_,e_anomaly)
        self.true_anomalies.append(t_anomaly)
        r = radio_vector (self.a_, self.e_, t_anomaly)
        xt = r * cos(t_anomaly) + self.f_
        yt = r * sin(t_anomaly)
        #logger.info(f"For t {t:.2f}, radio vector: {r:.2f}, xt: {xt:.2f}, yt: {yt:.2f}")
        #logger.info(f"For t {t:.2f}, radio vector: {r:.2f}, xt: {xt:.2f}, yt: {yt:.2f}")
        if t  < int(self.P_) :
            self.true_pos.append((xt,yt))
            self.mean_pos.append((xm,ym))
            self.eccentric_pos.append((xe,ye))
        #logger.warning(f"For t {t:.2f}, idx: {self.idx}")
        return  xt,yt,xm,ym,xe,ye

    def get_true_pos(self):
        return self.true_pos.copy()

    def get_mean_pos(self):
        return self.mean_pos.copy()

    def get_eccentric_pos(self):
        return self.eccentric_pos.copy()

    def get_mean_anomalies(self):
        return self.mean_anomalies.copy()

    def get_true_anomalies(self):
        return self.true_anomalies.copy()

    def get_eccentric_anomalies(self):
        return self.eccentric_anomalies.copy()



def e_anomaly_v1(e, m_anomaly, old_e_anomaly):
    return m_anomaly + e*np.sin(old_e_anomaly)

def e_anomaly_v2(e,m_anomaly, old_e_anomaly):
    nu = m_anomaly + e* np.sin(old_e_anomaly) - old_e_anomaly
    de = 1 - e * np.cos(old_e_anomaly)
    return old_e_anomaly + nu/de

def e_anomaly_v3 (e, m_anomaly, old_e_anomaly):
    nu = old_e_anomaly - e*np.sin(old_e_anomaly) - m_anomaly
    de = 1 - e*np.cos(old_e_anomaly)
    return old_e_anomaly - nu/de   

# Other iteration function radians
def next_E (e:float, m_anomaly:float, E:float) -> float :
    """
    Iterative function to calculate the eccentric anomaly, i.e.,
    computes the next eccentric value for ellictical orbits.
    Used in the Newton method (Pag 65 of Astronomy of 
    the personal computer)

    Args:
        e : eccentricity 
        m_anomaly: Mean anomaly in angle units (rads)
        E : Previous value of the eccentric anomaly

    Returns :
        The eccentric anomaly in angle units (radians)
    """

    num = E - e * sin(E) - m_anomaly
    den = 1 - e * cos(E)
    return E - num/den


def next_H (e:float, mh_anomaly: float, H:float) -> float:
    """
    Iterative function to calculate the eccentric anomaly, i.e.,
    computes the next eccentric value for hyperbolic orbits.
    Used in the Newton method (Pag 65 of Astronomy of 
    the personal computer)

    Args:
        e : eccentricity 
        mh_anomaly: Mean anomaly in angle units (rads) 
        H : Previous value of the eccentric anomaly

    Returns :
        The eccentric anomaly in angle units (radians)
    """
    num = e * sinh(H) - H - mh_anomaly
    den = e * cosh(H) - 1
    return H - num/den

# General function to iterate functions of onw variable
def do_iterations(func, x_0, abs_tol = 1e-08, max_iter=50):
    x = x_0
    for n in range(max_iter):
        new_value = func(x)
        if isclose(new_value, x, abs_tol = abs_tol) :
            return (True,new_value,n)
        else :
            x = new_value
    return (False,new_value,n)

def solve_ke(e, func_e_anomaly,  m_anomaly):
    """
    Solves the kepler equation, i.e., calculates
    the escentric anomaly

    Args:
        e : eccentricity 
        func_e_anomaly: function to calculate the excentric anomaly used to iterate
        m_anomaly: Mean anomaly in angle units (rads)

    Returns :
        The eccentric anomaly in angle units (radians)
    """

    f = partial(func_e_anomaly,e,m_anomaly)
    res = do_iterations(f,m_anomaly,100)
    if not res[0] :
        logger.warning("Not converged")
        return 
    else :
        logger.debug(f"Converged in {res[2]} iterations wih result {np.rad2deg(res[1])} degrees")
        return res[1]


def solve_ke_newton(e, func_e_anomaly, m_anomaly, e_anomaly_0=None):
    """ 
    Solves the kepler equation, i.e., calculates the excentric anomaly by using
     the Newton-Ranson method

    Args:
        e : eccentricity [0,1]
        func_e_anomaly: function to calculate the excentric anomaly used to iterate
        m_anomaly: Mean anomaly in angle units (radians)
        e_anomaly_0 : Optional value to start the iteration 
        
    Returns :
        The eccentric anomaly in angle units (radians)
    """

    """
    if e_anomaly_0 is None :
        E0 = m_anomaly if (e<=0.8) else PI
    else :
        E0 = e_anomaly_0        
    """

    f = partial(func_e_anomaly , e ,m_anomaly)
    x, root = newton(lambda x : f(x) - x, e_anomaly_0, fprime=None,tol=1e-12, maxiter=50, fprime2=None, full_output=True)    
    logger.debug(root)
    if not root.converged :
        logger.warning(f"Not converged: {root}")
    return x

def solve_ke_bisect(e,m_anomaly,func_e_anomaly):
    m_rad = np.deg2rad(m_anomaly)    
    f = partial(func_e_anomaly,e,m_rad)    
    x = bisect(f, 0, 0.25, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, full_output=False, disp=True)
    return x
    #logger.info("Eccentric anomaly (deg):",np.rad2deg(x))
    #print ("Eccentric anomaly (deg):"+np.rad2deg(x))


def half_tan_t_anomaly(e:float, e_anomaly:float) -> float :
    """ 
    Computes the half of the tangent of the true anomaly 
    Args:
        e : eccentrity of the orbit
        e_anomaly : Excentric anomaly in radians

    Returns:
        The computed value
    """
    return sqrt((1+e)/(1-e))*tan(e_anomaly/2.0)


def calc_M_by_period (t0:float, T:float, t:float) ->float :
    """ 
    Computes the mean anomaly function as function of time, M(t), angular coordinate for an imaginary
    mean planet with a circular orbit of radius equal to semi-major axis  at constant speed

    Args:
        t0 : Time reference at which the object passed through perihelion
             (e.g. 1986 2 9.43867, year, month, fractional.day)
        T :  Period of the orbit in the same unit as t and t0 (normaly days)
        t :  time reference where the mean anomaly is calculated
        
    Returns :
        The mean anomaly in angle units (radians) at time t
    """
    return TWOPI*(t-t0)/T

def true_anomaly(e, e_anomaly):
    """ 
    Computes the true anomaly 

    Args:
        e : eccentricity of the orbit
        e_anomaly : eccentric anomaly in radians
        
    Returns :
        The true anomaly in angle units (radians) 
    """

    return 2 * arctan(sqrt((1+e)/(1-e))* tan(e_anomaly/2))

def radio_vector (a,e,t_anomaly):
    return a*(1-e*e)/(1+e*cos(t_anomaly))


def elliptic_orbit (next_E_func, m_anomaly, a, e):
    """ 
    Computes position (r) and velocity (v) vectors for elliptic orbits
    Pg 66 of Astronomy on the Personal computer
    The m_anomaly is the one that varies with time so the result
    of this function will also vary with time.

    An ellipse has (0<e<1)

    Args:
        next_E_func : iterating function to calculate the eccentric anomaly, used
                      for solving kepler equation with Newton 
        m_anomaly : The current Mean anomaly in rads (it will depend on time t)
        a : Semi-major axis of the orbit in [AU]
        e  : Eccentricity of the orbit (<1 for elliptical)
        
    Returns :
        Position (r) w.r.t. orbital plane in [AU] 
        Velocity (v) w.r.t orbital plane in [AU/days]

    """

    E0 = m_anomaly if (e<=0.8) else PI

    e_anomaly = solve_ke_newton(e, next_E_func, m_anomaly, E0)
    # Pg 65 Astronomy on the Personal Computer
    t_anomaly = true_anomaly(e,e_anomaly)
    
    cte = sqrt(GM/a)
    cos_E = cos(e_anomaly)
    sin_E = sin(e_anomaly)
    fac =  np.sqrt(1-e*e)
    rho = 1.0 - e*cos_E
    r =  np.array([a*(cos_E-e), a*fac*sin_E, 0.0]) #x,y at pag 62
    v =  np.array([-cte*sin_E/rho, cte*fac*cos_E/rho, 0.0])
    return r,v


def hyperpolic_orbit (tp, next_H_func, a, e, t):
    """ 
    Computes position (r) and velocity (v) vectors for hiperbolic orbits
    Pg 66 of Astronomy on the Personal computer
    The m_anomaly is the one that varies with time so the result
    of this function will also vary with time.

    An hyperbola has (e>1 e.g. e=1.5)

    Args:
        tp: Time of perihelion passage in Julian centuries since J2000
        next_H_func : iterating function to calculate the eccentric anomaly, used
                for solving kepler equation with Newton 
        a : Semi-major axis of the orbit in [AU]
        e : eccentricity of the orbit (>1 for hiperbola
        t : time of the computation in Julian centuries since J2000
        
    Returns :
        Position (r) w.r.t. orbital plane in [AU] 
        Velocity (v) w.r.t orbital plane in [AU/days]

    """
    cte = sqrt(GM/a)
    # I have done the calculation and it is right
    # 2*pi/T == cte/a  so this is equivalent to the mean calculation done previously
    # Mean anomaly for the hyperbolic orbit
    Mh = cte*(t-tp)/a 
    # Initial Eccentric anomaly for the hyperbolic orbit depends on Mh
    if Mh >=0 :
        H0 = np.log(1.8+2*Mh/e)
    else :
        H0 = -np.log(1.8-2*Mh/e)
    # Solve the eccentric anomaly
    H = solve_ke_newton(e, next_H, Mh, H0)
    cosh_H = cosh(H)
    sinh_H = sinh(H)
    fac =  sqrt((e+1.0)*(e-1.0))
    rho = e*cosh_H - 1.0
    r = np.array([a*(e-cosh_H), a*fac*sinh_H, 0.0])
    v = np.array([-cte*sinh_H/rho, cte*fac*cosh_H/rho,0.0])
    return r,v

def parabolic_orbit (tp, q, e, t, max_iters=15):
    """ 
    Computes position (r) and velocity (v) vectors for hiperbolic orbits
    Pg 66 of Astronomy on the Personal computer
    The m_anomaly is the one that varies with time so the result
    of this function will also vary with time.

    An parabolic has (e=1)

    Args:
        tp: Time of perihelion passage in Julian centuries since J2000
        next_H_func : iterating function to calculate the eccentric anomaly, used
                for solving kepler equation with Newton 
        a : Semi-major axis of the orbit in [AU]
        e : eccentricity of the orbit (>1 for hiperbola
        t : time of the computation in Julian centuries since J2000
        
    Returns :
        Position (r) w.r.t. orbital plane in [AU] 
        Velocity (v) w.r.t orbital plane in [AU/days]

    """
    E_2 = 0.0    
    factor = 0.5 * e
    cte = sqrt(GM/(q*(1.0+e)))
    tau = sqrt(GM)*(t-tp)
    epsilon = 1e-7

    for i in range(max_iters):
        E20 = E_2 
        A = 1.5 * sqrt(factor/(q*q*q))*tau
        B = np.cbrt(sqrt(A*A+1.0)+A)
        u = B - 1.0/B 
        u_2 = u*u
        E_2 = u_2*(1.0-e)/factor 
        c1, c2, c3 = calc_stumpff_values(E_2)
        factor = 3.0*e*c3 
        if isclose(E_2, E20, abs_tol=epsilon) :
        #if np.isclose(E_2,E20,atol=epsilon):
            R = q * (1.0 + u_2*c2*e/factor)
            r = np.array([q*(1.0-u_2*c2/factor), q*sqrt((1.0+e)/factor)*u*c1,0.0])
            v = np.array([-cte*r[1]/R, cte*(r[0]/R+e),0.0])
            return r,v
    #logger.warning(f"Not converged after {i} iterations")
    logger.warning(f'Not converged with q:{q},  e:{e}, t:{t}, t0:{tp} after {i} iterations')
    return r,v
    
def calc_stumpff_values(E_2, epsilon=1e-7, max_iters=100):    
    """ 
    Computes the values for the Stumpff functions C1, C2, C3 

    Args:
        E_2: Square of eccentric anomaly in rads^2
        epsilon: relative accuracy 
        max_iters: 
        
    Returns :
        A tuple with the valus of c1, c2, c3 
    """
    c1, c2, c3 = 0.0, 0.0, 0.0
    to_add = 1.0
    for n in range(1 ,max_iters):
        c1 += to_add
        to_add /= (2.0*n)
        c2 += to_add
        to_add /= (2.0*n+1.0)
        c3 += to_add
        to_add *= -E_2
        if isclose(to_add, 0, abs_tol=epsilon) :
        #if np.isclose(to_add,epsilon,atol=epsilon):
            return c1, c2, c3
    logger.warning(f"Not converged after {n} iterations")
    return c1, c2, c3

def h_xyz_eclip_keplerian_orbit (q, e, mtx_PQR, tp, t) :
    """ 
    Computes position (r) and velocity (v) vectors for keplerian orbits
    depending the eccentricy and mean_anomlay to choose which type of conic
    use.

    Args:
        q: Perihelion distance q in AU
        e: Eccentricity of the orbit
        mtx_U : Matrix to change from orbital plane to eclictic plane
        tp : Time reference at which the object passed through perihelion
        t : time reference where the r, v vector will be calculated

    Returns :
        Position (r): It is a np.array of 3 elements with the cartesian coordinates w.r.t. the ecliptic
        Velocity (v): It is a np.array of 3 elements with the cartesian coordinates w.r.t. the ecliptic
    """
    M0 = 0.1 # radians
    epsilon = 0.1
    a = calc_semimajor_axis(e,q)    
    M = pipe( calc_M(t, tp, a), tc.reduce_rad)         
    if (M < M0) and (np.abs(1.0-e)< epsilon) :
        r,v = parabolic_orbit(tp, q, e, t,50)
        #r,v = elliptic_orbit(next_E, M, a, e)
    elif (e < 1.0) :
        r,v = elliptic_orbit(next_E, M, a, e)
    else :
        r,v =  hyperpolic_orbit (tp, next_H, a, e, t)
    # The r and v are transformed from the orbital plane to the ecliptic coordinate system
    # althouth they keep being cartesian coordinates.
    return mtx_PQR.dot(r),  mtx_PQR.dot(v)


def calc_semimajor_axis(e,q):
    """ 
    Computes the semi-major axis 

    Args:
        e : Eccentricity of the orbit
        q : Perihelion distance q in AU
        
    Returns :
        The semi-major axis in AU
    """
    return q/(1.0-e)

def calc_M(t:float, tp:float, a:float) -> float :
    """ 
    Computes the mean anomaly as a function a t, t0 and a, i.e., not depending on the
    period of the orbit=1)

    Args:
        t : time of the computation 
        t0: Time of perihelion passage 
        a: semi-major axis in AU

    Returns :
        The mean anomaly in radians
    """
    M = sqrt(GM)*(t-tp)/np.float_power(a,1.5)
    return reduce_rad(M, to_positive=True)

def calc_M_by_M0(t, t0, M0, period):
    """ 
    Computes the mean anomaly as a function of a t, a mean anomaly (M0) at time t0
    and the period.

    Args:
        t : time of the computation 
        t0: a point in time where Mean anomaly is knwon
        M0: the mean anomaly in radians at t0
        period : period of the orbit in days.

    Returns :
        The mean anomaly in radians
    """
    #return pipe(((t - t0)*TWO_PI/period)+M0, tc.norm_rad)
    M = (t-t0)*TWOPI/period
    return reduce_rad(M+M0,to_positive=True)



def accel(gm, r):
    """
    Computes the acceleration based on the corresponding GM and the
    radio vector.
    Returns a vector
    """
    return gm*r/pow(norm(r),3) 

def calc_accelaration_v2(t, h_xyz_eclip_body ) :
    """
    Computes the acceleration vector for a minor body in the solar system at one point
    time.

    Args:
        t :  point in time in Modified Julian days
        h_xyz_eclip_body : Heliocentric Ecliptic Cartesian coordinates (J2000) of the minor body (as numpy vector of 3)

    Returns :
        The acceleration as J2000 ecliptic coordinate vector (numpy vector of 3) [AU/d**2]
    """
    # sun acceleration is calculated (as a vector)
    acc = accel(GM, h_xyz_eclip_body)
    # The century T corresponding to the time t. Also used to calculate
    # precession matrix
    T = (t - MDJ_J2000)/36525.0    
    # desired equinox is J2000, so T_desired is 0
    T_desired = (JD_J2000 - JD_J2000)/36525.0     
    mtx_prec = co.mtx_eclip_prec(T, T_desired)

    for pl_name in filter(lambda x : x != 'Sun', GM_by_planet.keys()) :
        # Planetary position (ecliptic and equinox of J2000)
        r_heclipt_planet = pipe(planet_heclipt_rv(pl_name, T)[0], mtx_prec.dot)
        r_planet_body = h_xyz_eclip_body - r_heclipt_planet
        # Direct accelaration
        acc += accel(GM_by_planet[pl_name],r_planet_body)
        # Indirect acceletration
        acc += accel(GM_by_planet[pl_name],r_heclipt_planet)
    return acc


def calc_perturbed_accelaration(t_mjd, h_xyz_eclip_body ) :
    """
    Computes the acceleration vector for a minor body in the solar system at one point
    time.

    Args:
        t :  point in time in Modified Julian days
        h_xyz_eclip_body : Heliocentric Ecliptic Cartesian coordinates (J2000) of the minor body (as numpy vector of 3)

    Returns :
        The acceleration as J2000 ecliptic coordinate vector (numpy vector of 3) [AU/d**2]
    """
    # Sun acceleration is calculated (as a vector)
    #acc = accel(GM, h_xyz_eclip_body)
    # The century T corresponding to the time t. Also used to calculate
    # precession matrix
    T = (t_mjd - MDJ_J2000)/CENTURY    
    # desired equinox is J2000, so T_desired is 0
    T_desired = (JD_J2000 - JD_J2000)/CENTURY
    mtx_prec = co.mtx_eclip_prec(T, T_desired)
    acc = 0
    #for pl_name in filter(lambda x : (x != 'Sun') and (x != 'Pluto'), GM_by_planet.keys()) :
    for pl_name in filter(lambda x : (x != 'Sun') , GM_by_planet.keys()) :
        if pl_name == 'Pluto':
            h_xyz_eclipt_planet = h_xyz_eclip_pluto_j2000(mjd2jd(t_mjd))
        else :
            # Planetary position (ecliptic and equinox of J2000)
            h_xyz_eclipt_planet = mtx_prec.dot(h_xyz_eclip_eqxdate(pl_name, mjd2jd(t_mjd)))

        h_xyz_planet_body = h_xyz_eclip_body - h_xyz_eclipt_planet
        # Direct accelaration
        acc += accel(GM_by_planet[pl_name], h_xyz_planet_body)
        # Indirect acceletration
        acc += accel(GM_by_planet[pl_name], h_xyz_eclipt_planet)
    return -acc    


def planet_heclipt_rv(pl_name, T) :
    """ 
    Computes position (r) and velocity (v) vectors of planet at the epoch 
    of date (T). The caller will have to convert to the desired epoch

    Args:
        pl_name : Name of the planet
        T : time in Julian centuries since J2000
        
    Returns :
        Position (r) w.r.t. in heliocentric ecliptic cartesian coordinates 
        Velocity (v) w.r.t  in heliocentric ecliptic cartesian coordinates
    """

    if pl_name == 'Sun':
        return np.array([0,0,0]), np.array([0,0,0])     

    row = DF_PLANETS[DF_PLANETS.name.str.contains(pl_name)]
    # Asumed 0 because the orbital elements are the ones at 2000 January 1.5
    # i.e. T0 (centuries from J2000) is 0 
    T0 = 0 
    row = row.to_dict('records')[0]

    # The matrix to transform from orbital plane to ecliptic coordinates
    mtx_PQR = co.mtx_gauss_vectors(row['Node'] + CENTENNIAL_PRECESSION_RAD * T,
                                   row['i'], 
                                   row['w']-row['Node'])                                   

    m_anomaly = row['M']+(row['n']*(T-T0))
    r,v = elliptic_orbit (next_E, m_anomaly, row['a'], row['e'])

    return mtx_PQR.dot(r), mtx_PQR.dot(v)

def my_f(t, Y):        
    h_xyz = Y[0:3]

    # - Sun acceleration  - perturbed acceleration
    acc = -accel(GM, h_xyz) + calc_perturbed_accelaration(t, h_xyz)

    #def calc_accelaration(t_mjd, h_xyz_eclip_body ) :
    #acc = calc_accelaration(t,h_xyz)
    return np.concatenate((Y[3:6], acc))
    

def do_integration(fun_t_y, y0 , t_begin, t_end, t_samples):
    sol = solve_ivp(fun_t_y, (t_begin,t_end), y0, t_eval=t_samples, rtol = 1e-12)  
    if sol.success :
        return sol
        #print (sol.t)
        #idx = len(sol.t) -1
        #return sol.y[:,idx][:3] , sol.y[:,idx][3:6]
    else :
        logger.warn("The integration was failed: "+sol.message)

  
def calc_osculating_orb_elmts(h_xyz, h_vxyz, epoch_mjd=0, equinox="J2000"):
    """ 
    Computes the orbital elements of an elliptical orbit from position
    and velocity vectors
    
    Args:
        r : Heliocentric ecliptic position vector  in [AU]
        v : Heliocentric ecliptic velocity vector  in [AU/d]
        epoch_mjd : time in Modified Julian Day where the orbital elements are calculalted 
                    i.e., the epoch
        
    Returns :
        An OrbElmtsData 
    """
    h = np.cross(h_xyz, h_vxyz)
    H = norm(h)
    Omega = arctan2(h[0], -h[1])
    i = arctan2(sqrt(pow(h[0],2)+pow(h[1],2)),h[2])
    u = arctan2(h_xyz[2]*H, -h_xyz[0]*h[1]+h_xyz[1]*h[0])
    R = norm(h_xyz)
    v_2 = h_vxyz.dot(h_vxyz)
    a = 1.0/(2.0/R-v_2/GM)
    e_cosE = 1.0-R/a
    e_sinE = h_xyz.dot(h_vxyz)/sqrt(GM*a)
    e_2 = pow(e_cosE,2) + pow(e_sinE,2)
    e = sqrt(e_2)
    E = arctan2(e_sinE,e_cosE)
    M = E - e_sinE
    nu = arctan2(sqrt(1.0-e_2)*e_sinE, e_cosE-e_2)
    omega = u - nu
    if Omega < 0.0 :
        Omega += 2.0*PI
    if omega < 0.0 :
        omega += 2.0*PI
    if M < 0.0:
        M += 2.0*PI    
    return dc.BodyElems.in_radians("Osculating Body",tc.mjd2epochformat(epoch_mjd),a,e,i,Omega,omega,M,equinox)


if __name__ == "__main__" :

    e, m_anomaly  = 0.6813025 , np.deg2rad(5.498078)
    # Solving with Newton
    e_funcs = [e_anomaly_v1,e_anomaly_v2,e_anomaly_v3, next_E] 
    for func in e_funcs:        
        e_anomaly = solve_ke_newton(e, func, m_anomaly)        
        print (np.rad2deg(e_anomaly))
    

  



    





    

