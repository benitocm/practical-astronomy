"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html


# Third party imports
import pandas as pd
import numpy as np
from numpy import rad2deg, deg2rad
from numpy.linalg import norm
import toolz as tz
# Using Newton-Ramson method
from scipy.integrate import solve_ivp    

from myastro import util as ut
from myastro import data_catalog as dc
from myastro import timeutil as tc
from myastro import coord as co
from myastro import orbit as ob

from myastro.orbit  import EphemrisInput

from myastro.timeutil import  PI_HALF, PI, TWOPI
from myastro.keplerian import KeplerianOrbit
from myastro.lagrange_coeff import rv_from_r0v0
from myastro.timeutil import epochformat2jd, jd2mjd, T, mjd2jd, jd2str_date, MDJ_J2000, JD_J2000
from myastro.planets import g_xyz_equat_sun_j2000, g_rlb_eclip_sun_eqxdate
from myastro.util import mu_by_name, mu_Sun
from myastro.orbit import calc_perturbed_accelaration


from myastro.log import get_logger
logger = get_logger(__file__.split('/')[-1])

def f1(vector):
    # Utility function
    return vector/pow(norm(vector),3)

def calc_F(a, b ,c):
    # Function to compute the difference between nearly equal numbers
    # Appendix F of Orbital Mechanics
    q = a * (2*b-a)/pow(b,2)
    return (pow(q,2)- 3*q +3 / (1+pow(1-q,1.5)))*q

def my_dfdt(t, y, r0, v0, t0):
    """
    Computes the time derivative of the unknown function. Integrating this function, we obtain the unknown
    function. We know the velocity and acceleration that is basically what this function returns so integrating we obtain 
    the position and velocity.

    Args:
        t : point in time (normally used in modified julian days) at which we want to calculate the derivative
        y  : The vector with the variables to solve the differential equation system
                [0..3] delta_r
                [3..6] delta_v (not used in this case)
        r0 : Radio Vector of the object w.r.t. the Sun (AUs) at time t0
        v0 : Velocity vector Elapsed time (AUs/days) at time t0
        t0 : Initial point timme

    Returns :
        A vector vector of 6 positions with delta_v and delta_acc ()
    """  
    delta_r = y[0:3]    
    # The two-bodys orbit is calculated starting at r0,v0 and t-t0 as elapsed time
    r_osc, _ = rv_from_r0v0(mu_Sun, r0, v0, t-t0)    
    # The radio vector perturbed is the two-bodys plus the delta_r
    r_pert = r_osc + delta_r
    F = 1 - pow(norm(r_osc)/norm(r_pert),3)
    #TODO Check if this works, to avoid compute the difference between nearly equal numbers 
    #F = calc_F(norm(delta_r), norm(r_pert), norm(r_osc))
    # The increment of accelration is calculated including the normal perturbed acceleartion
    delta_acc = (-mu_Sun/pow(norm(r_osc),3))*(delta_r- F*r_pert)+calc_perturbed_accelaration(t, r_pert)    
    return np.concatenate((y[3:6],delta_acc))


def apply_enckes(eph, t_range, r0, v0):
    """
    This is a utility function needed because the integration needs to be done in two intervals so this function
    is called for each of these intervals. It applies the enckles's approach, i.e. calcualate the dr and dv
    to modified the two bodys (osculating orbit)

    Args:
        eph : Ephemeris data (EphemrisInput)
        t_range : A numpy vector with the time samples where each time sample defines a time interval.
                  The enckles method is applied in each one of this interval.
                  The time samples are modified julian days.
        r0 : A numpy vector that indicates the initial radio vector (AUs)
        v0  : A numpy vector that indicates the initial velocity vector (AUs/days)
        r0 : Radio Vector of the object w.r.t. the Sun (AUs) at time t0

    Returns :
        A dictionary where the key is a time reference in days (modified julian days) and the 
        the value is the a tuple with two vectors, the radio vector r and the velocity vector at the time reference
    """  
    steps = np.diff(t_range)
    result = dict()
    clock_mjd = t_range[0]
    for idx, step in enumerate(steps) :
        #if (idx % 50) == 0 :
        #    print (f"Iteration: {idx},  Date : {jd2str_date(tc.mjd2jd(clock_mjd))}")        
        sol = solve_ivp(my_dfdt, (clock_mjd, clock_mjd+step), np.zeros(6), args=(r0, v0, clock_mjd) , rtol = 1e-12)  
        assert sol.success, "Integration was not OK!"
        r_osc, v_osc = rv_from_r0v0 (mu_Sun, r0, v0, step)
        # The last integration value is taken
        r0 = r_osc + sol.y[:,-1][:3]
        v0 = v_osc + sol.y[:,-1][3:6]
        # If the clock is in the middle of the ephemeris time, it is inserted in the solution
        if eph.from_mjd <= clock_mjd+step <= eph.to_mjd :
            result[clock_mjd+step] = (r0, v0)
        clock_mjd += step    
    return result


def calc_eph_by_enckes (body, eph, type='body'):
    """
    Computes the ephemeris for a minor body using the Enckes method. This has more precission that
    the Cowells but it takes more time to be calculated.

    Args:
        body : The orbital elements of the body, it can be an body or a comet
        eph : Ephemeris data (EphemrisInput)
        type : Indicates if the body is a asteroid ('body') or a comet ('comet')

    Returns :
        A dataframe with the result
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
        #t_range = list(ut.frange(initial_mjd, eph.to_mjd, eph.step))
        t_range = ut.my_range(initial_mjd, eph.to_mjd, eph.step)
        result_1 = apply_enckes(eph, t_range, r0, v0)

        # and backwards 
        #t_range = list(ut.frange(eph.from_mjd, initial_mjd, eph.step))
        #if t_range[-1] != initial_mjd :
        #    t_range.append(initial_mjd)
        t_range = list(reversed(ut.my_range(eph.from_mjd, initial_mjd, eph.step)))
        result_2 = apply_enckes(eph, t_range, r0, v0)
        solution = tz.merge([result_1,result_2])        
        
    elif initial_mjd < eph.from_mjd :
        """
        # If the epoch is in the past, we need to integrate forward
        t_range_1 = list(ut.frange(initial_mjd, eph.from_mjd, eph.step))
        # The previous ensure that initial_mjd is included but the eph.from  may be not included
        # so we test the final value to know if we need to include manually
        if t_range_1[-1] != eph.from_mjd :
            t_range_1.append(eph.from_mjd)
        """
        # [initial, from] 
        t_range_1 = ut.my_range(initial_mjd, eph.from_mjd, eph.step)
        # [from+step, to]
        t_range_2 = ut.my_range(eph.from_mjd+eph.step, eph.to_mjd, eph.step)

        """
        t_range_2 = list(ut.frange(eph.from_mjd+eph.step, eph.to_mjd, eph.step))
        if len(t_range_2) == 0 :
            t_range_2.append(eph.to_mjd)
        if t_range_2[-1] != eph.to_mjd :
            t_range_2.append(eph.to_mjd)
        """
        solution = apply_enckes(eph, t_range_1 + t_range_2, r0, v0)
    else :
        # If the epoch is in the future, we need to integrate backwards
        # goes from the epoch backward toward the end value from 
        # the ephemeris and inital value of the ephemeris

        #[initial_mjd ---> backwards to  --> eph.to.mjd]
        t_range_1 = ut.my_range(eph.to_mjd, initial_mjd, eph.step)

        """
        t_range_1 = list(ut.frange(eph.to_mjd, initial_mjd, eph.step))
        # The previous ensure that eph.to is included but the initial may be not included
        # so we test the final value to know if we need to include manually
        if t_range_1[-1] != initial_mjd :
            t_range_1.append(initial_mjd)
        """
        """
        t_range_2 = list(ut.frange(eph.from_mjd, eph.to_mjd, eph.step))
        # the previous ensure that eph.from is included but the to_mjd may be included
        # but  we include in the previous so we need to remove it . We test the last element to check
        # if we need to remove it
        if t_range_2[-1] == eph.to_mjd :
            t_range_2 = t_range_2[0:-1]
        t_range = list(reversed(t_range_1)) + list(reversed(t_range_2))
        """
        t_range_2 = ut.my_range(eph.from_mjd, eph.to_mjd-eph.step, eph.step)
        t_range = list(reversed(t_range_2+t_range_1))
        solution = apply_enckes(eph, t_range, r0, v0)

    solution = {t:solution[t] for t in sorted(solution.keys())}
        

    return ob.process_solution(solution, MTX_J2000_Teqx, MTX_equatFeclip)



def test_all():
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    for name in dc.DF_BODYS['Name']: 
        body = dc.read_body_elms_for(name,dc.DF_BODYS)
        print ("Calculating for ",name)
        print (calc_eph_by_enckes(body, eph)) 



def test_body():
    eph = EphemrisInput(from_date="2020.05.15.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODYS)


    df = calc_eph_by_enckes(CERES, eph)

    print (df[df.columns[0:8]])

def test_comet():

    eph = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")                    

    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)


    df = calc_eph_by_enckes(HALLEY_J2000, eph, 'comet')

    print (df[df.columns[0:8]])
    
def test_all_comets():
    """
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")
    """

    df_ = dc.DF_COMETS
    df_ = df_.query("0.999 < e < 1.001")

    for idx, name in enumerate(df_['Name']): 
        logger.warning(f"{idx+1} Calculating for {name} ")
        body = dc.read_comet_elms_for(name,dc.DF_COMETS)
        eph = EphemrisInput.from_mjds( body.epoch_mjd-25, body.epoch_mjd+25, "02 00.0", "J2000" )
        print (f"{idx+1} Calculating for {name} ")
        print (calc_eph_by_enckes(body, eph, 'comet')) 

def test_several_comets():
    names = [#"C/1988 L1 (Shoemaker-Holt-Rodriquez)",   # Parabolic
             "C/-146 P1"]  
             #"C/1848 P1 (Petersen)"]

    for name in names :
        body = dc.read_comet_elms_for(name,dc.DF_COMETS)
        eph = EphemrisInput.from_mjds( body.epoch_mjd-50, body.epoch_mjd+50, "02 00.0", "J2000" )
        print (f"Calculating for {name} ")
        df = calc_eph_by_enckes(body, eph, 'comet')
        print (df[df.columns[0:8]])
        #print (calc_eph_by_enke(body, eph, 'comet')) 

@ut.measure
def test_speed():
    t = 49400.0
    y = np.array([0., 0., 0., 0., 0., 0.]) 
    r0 = np.array([-13.94097381,  11.4769406 ,  -5.72123976])
    v0 = np.array([-0.00211453,  0.0030026 , -0.00107914])
    t0 =  49400.0
    my_dfdt(t, y, r0, v0, t0)

if __name__ == "__main__" :
    #test_all()
    #test_2()
    #test_all_comets()
    #test_comet()
    #test_several_comets()
    #test_body()
    test_comet()
    #test_speed()



 



    





    

