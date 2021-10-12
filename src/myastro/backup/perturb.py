"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

import sys

# Third party imports
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt 
from numpy.linalg import norm
import toolz as tz

# Local application imports
from myastro import timeutil as tc
from myastro import coord as co

from myastro.log import get_logger
logger = get_logger(__file__.split('/')[-1])


o = 16
dim = 2 * o + 1

def AddThe (c1, s1, c2, s2) :
    return c1 * c2 - s1 * s2, s1 * c2 + c1 * s2


class Pert :

    def __init__ (self, T, M,  I_min,  I_max,  m,  i_min,  i_max, phi=0.0):
        self.m_T = T
        self.m_cosM = cos(M)
        self.m_sinM = sin(M)
        self.m_C = np.zeros(dim)
        self.m_S = np.zeros(dim)
        self.m_c=  np.zeros(dim)
        self.m_s = np.zeros(dim)
        self.m_dl = 0.0
        self.m_db = 0.0
        self.m_dr = 0.0
        self.m_u = 0.0
        self.m_v = 0.0

        self.m_C[o] = cos(phi)
        self.m_S[o] = sin(phi)

        for i in range(I_max):
            self.m_C[o + i + 1], self.m_S[o + i + 1] = AddThe(self.m_C[o + i], self.m_S[o + i], +self.m_cosM, +self.m_sinM)
        
        for i  in range (0,I_min,-1):
            self.m_C[o + i - 1], self.m_S[o + i - 1] = AddThe(self.m_C[o + i], self.m_S[o + i], self.m_cosM, -self.m_sinM)

        self.m_c[o] = 1.0
        self.m_c[o + 1] = cos(m)
        self.m_c[o - 1] = +self.m_c[o + 1]
        self.m_s[o] = 0.0
        self.m_s[o + 1] = sin(m)
        self.m_s[o - 1] = -self.m_s[o + 1]

        for i  in range (1, i_max):
            self.m_c[o + i + 1], self.m_s[o + i + 1] = AddThe(self.m_c[o + i], self.m_s[o + i], self.m_c[o + 1], self.m_s[o + 1])

        for i  in range (-1,i_min,-1):            
            self.m_c[o + i - 1], self.m_s[o + i - 1] = AddThe(self.m_c[o + i], self.m_s[o + i], self.m_c[o - 1], self.m_s[o - 1]) 


    #Sum-up perturbations in longitude, radius and latitude
    def Term (self, I,  i, iT,  dlc,  dls,  drc, drs,  dbc,  dbs ):
        if (iT == 0) :
            self.m_u, self.m_v  = AddThe (self.m_C[o+I],self.m_S[o+I], self.m_c[o+i],self.m_s[o+i])
        else :
            self.m_u *= self.m_T
            self.m_v *= self.m_T

        self.m_dl += ( dlc*self.m_u + dls*self.m_v )
        self.m_dr += ( drc*self.m_u + drs*self.m_v )
        self.m_db += ( dbc*self.m_u + dbs*self.m_v )

    #Retrieve perturbations in longitude, radius and latitude
    def dl(self):
        return self.m_dl

    def dr(self):
        return self.m_dr

    def db(self):
        return self.m_db

pi2 = 2*np.pi
Arcs = 3600.0*180.0/np.pi

def Frac(number) :
    return tc.my_frac(number)
    
def sunPos(T):
    # Mean anomalies of planets and mean arguments of lunar orbit [rad]
    M2 = pi2 * Frac ( 0.1387306 + 162.5485917*T )
    M3 = pi2 * Frac ( 0.9931266 + 99.9973604*T )
    M4 = pi2 * Frac ( 0.0543250 + 53.1666028*T )
    M5 = pi2 * Frac ( 0.0551750 + 8.4293972*T )
    M6 = pi2 * Frac ( 0.8816500 + 3.3938722*T )
    D = pi2 * Frac ( 0.8274 + 1236.8531*T )
    A = pi2 * Frac ( 0.3749 + 1325.5524*T )
    U = pi2 * Frac ( 0.2591 + 1342.2278*T )

    #Keplerian terms and perturbations by Venus
    Ven = Pert(T, M3,0,7, M2,-6,0 )
    Ven.Term ( 1, 0,0,-0.22,6892.76,-16707.37, -0.54, 0.00, 0.00)
    Ven.Term ( 1, 0,1,-0.06, -17.35, 42.04, -0.15, 0.00, 0.00)
    Ven.Term ( 1, 0,2,-0.01, -0.05, 0.13, -0.02, 0.00, 0.00)
    Ven.Term ( 2, 0,0, 0.00, 71.98, -139.57, 0.00, 0.00, 0.00)
    Ven.Term ( 2, 0,1, 0.00, -0.36, 0.70, 0.00, 0.00, 0.00)
    Ven.Term ( 3, 0,0, 0.00, 1.04, -1.75, 0.00, 0.00, 0.00)
    Ven.Term ( 0,-1,0, 0.03, -0.07, -0.16, -0.07, 0.02,-0.02)
    Ven.Term ( 1,-1,0, 2.35, -4.23, -4.75, -2.64, 0.00, 0.00)
    Ven.Term ( 1,-2,0,-0.10, 0.06, 0.12, 0.20, 0.02, 0.00)
    Ven.Term ( 2,-1,0,-0.06, -0.03, 0.20, -0.01, 0.01,-0.09)
    Ven.Term ( 2,-2,0,-4.70, 2.90, 8.28, 13.42, 0.01,-0.01)
    Ven.Term ( 3,-2,0, 1.80, -1.74, -1.44, -1.57, 0.04,-0.06)
    Ven.Term ( 3,-3,0,-0.67, 0.03, 0.11, 2.43, 0.01, 0.00)
    Ven.Term ( 4,-2,0, 0.03, -0.03, 0.10, 0.09, 0.01,-0.01)
    Ven.Term ( 4,-3,0, 1.51, -0.40, -0.88, -3.36, 0.18,-0.10)
    Ven.Term ( 4,-4,0,-0.19, -0.09, -0.38, 0.77, 0.00, 0.00)
    Ven.Term ( 5,-3,0, 0.76, -0.68, 0.30, 0.37, 0.01, 0.00)
    Ven.Term ( 5,-4,0,-0.14, -0.04, -0.11, 0.43,-0.03, 0.00)
    Ven.Term ( 5,-5,0,-0.05, -0.07, -0.31, 0.21, 0.00, 0.00)
    Ven.Term ( 6,-4,0, 0.15, -0.04, -0.06, -0.21, 0.01, 0.00)
    Ven.Term ( 6,-5,0,-0.03, -0.03, -0.09, 0.09,-0.01, 0.00)
    Ven.Term ( 6,-6,0, 0.00, -0.04, -0.18, 0.02, 0.00, 0.00)
    Ven.Term ( 7,-5,0,-0.12, -0.03, -0.08, 0.31,-0.02,-0.01)
    dl = Ven.dl(); dr = Ven.dr(); db = Ven.db();    

    #Perturbations by Mars
    Mar = Pert( T, M3,1,5, M4,-8,-1 )
    Mar.Term ( 1,-1,0,-0.22, 0.17, -0.21, -0.27, 0.00, 0.00)
    Mar.Term ( 1,-2,0,-1.66, 0.62, 0.16, 0.28, 0.00, 0.00)
    Mar.Term ( 2,-2,0, 1.96, 0.57, -1.32, 4.55, 0.00, 0.01)
    Mar.Term ( 2,-3,0, 0.40, 0.15, -0.17, 0.46, 0.00, 0.00)
    Mar.Term ( 2,-4,0, 0.53, 0.26, 0.09, -0.22, 0.00, 0.00)
    Mar.Term ( 3,-3,0, 0.05, 0.12, -0.35, 0.15, 0.00, 0.00)
    Mar.Term ( 3,-4,0,-0.13, -0.48, 1.06, -0.29, 0.01, 0.00)
    Mar.Term ( 3,-5,0,-0.04, -0.20, 0.20, -0.04, 0.00, 0.00)
    Mar.Term ( 4,-4,0, 0.00, -0.03, 0.10, 0.04, 0.00, 0.00)
    Mar.Term ( 4,-5,0, 0.05, -0.07, 0.20, 0.14, 0.00, 0.00)
    Mar.Term ( 4,-6,0,-0.10, 0.11, -0.23, -0.22, 0.00, 0.00)
    Mar.Term ( 5,-7,0,-0.05, 0.00, 0.01, -0.14, 0.00, 0.00)
    Mar.Term ( 5,-8,0, 0.05, 0.01, -0.02, 0.10, 0.00, 0.00)
    dl += Mar.dl(); dr += Mar.dr(); db += Mar.db();  

    #Perturbations by Jupiter
    Jup = Pert( T, M3,-1,3, M5,-4,-1  )
    Jup.Term (-1,-1,0, 0.01, 0.07, 0.18, -0.02, 0.00,-0.02)
    Jup.Term ( 0,-1,0,-0.31, 2.58, 0.52, 0.34, 0.02, 0.00)
    Jup.Term ( 1,-1,0,-7.21, -0.06, 0.13,-16.27, 0.00,-0.02)
    Jup.Term ( 1,-2,0,-0.54, -1.52, 3.09, -1.12, 0.01,-0.17)
    Jup.Term ( 1,-3,0,-0.03, -0.21, 0.38, -0.06, 0.00,-0.02)
    Jup.Term ( 2,-1,0,-0.16, 0.05, -0.18, -0.31, 0.01, 0.00)
    Jup.Term ( 2,-2,0, 0.14, -2.73, 9.23, 0.48, 0.00, 0.00)
    Jup.Term ( 2,-3,0, 0.07, -0.55, 1.83, 0.25, 0.01, 0.00)
    Jup.Term ( 2,-4,0, 0.02, -0.08, 0.25, 0.06, 0.00, 0.00)
    Jup.Term ( 3,-2,0, 0.01, -0.07, 0.16, 0.04, 0.00, 0.00)
    Jup.Term ( 3,-3,0,-0.16, -0.03, 0.08, -0.64, 0.00, 0.00)
    Jup.Term ( 3,-4,0,-0.04, -0.01, 0.03, -0.17, 0.00, 0.00)
    dl += Jup.dl(); dr += Jup.dr(); db += Jup.db();  

    #Perturbations by Saturn
    Sat = Pert(T, M3,0,2, M6,-2,-1 )
    Sat.Term ( 0,-1,0, 0.00, 0.32, 0.01, 0.00, 0.00, 0.00)
    Sat.Term ( 1,-1,0,-0.08, -0.41, 0.97, -0.18, 0.00,-0.01)
    Sat.Term ( 1,-2,0, 0.04, 0.10, -0.23, 0.10, 0.00, 0.00)
    Sat.Term ( 2,-2,0, 0.04, 0.10, -0.35, 0.13, 0.00, 0.00)
    dl += Sat.dl(); dr += Sat.dr(); db += Sat.db();     

    #dl = 0.0
    #dr = 0.0
    #db = 0.0

    #Difference of Earth-Moon-barycentre and centre of the Earth
    dl += 6.45*sin(D) - 0.42*sin(D-A) + 0.18*sin(D+A)+ 0.17*sin(D-M3) - 0.06*sin(D+M3)
    dr += 30.76*cos(D) - 3.06*cos(D-A) + 0.85*cos(D+A)- 0.58*cos(D+M3) + 0.57*cos(D-M3)
    db += 0.576*sin(U)

    #Long-periodic perturbations
    dl += + 6.40 * sin ( pi2*(0.6983 + 0.0561*T) )+ 1.87 * sin ( pi2*(0.5764 + 0.4174*T) ) + 0.27 * sin ( pi2*(0.4189 + 0.3306*T) )+ 0.20 * sin ( pi2*(0.3581 + 2.4814*T) )

    #Ecliptic coordinates ([rad],[AU])
    l = pi2 * Frac ( 0.7859453 + M3/pi2 +( (6191.2+1.1*T)*T + dl ) / 1296.0e3 );    
    r = 1.0001398 - 0.0000007 * T + dr * 1.0e-6
    b = db / Arcs

    return co.Coord(np.array([r,l,b]),'date_equinox',co.ECLIP_TYPE)
    #print (r,l,b)

    #return co.cartesianFpolar(np.array([r,l,b]))

if __name__ == "__main__":
    sun = sunPos(0.5)

    print (sun)
