"""
This module contains the tests for timeconv function
"""
# Standard library imports


# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np
from numpy.random import default_rng


# Local application imports
from myastro.timeutil import *
from myastro.coord import *

def test_dms2d():
    assert dgms2dg(24,13,18) == approx(24.221666)
    assert dgms2dg(13,4,10) == approx(13.069444)
    assert dgms2dg(24,13,18) == approx(24.221666)
    assert dgms2dg(13,4,10) == approx(13.069444)
    assert dgms2dg(300,20,00) == approx(300.333333)
    assert dgms2dg(182,31,27) == approx(182.524167)

def test_d2dms():
    assert dg2dgms(-0.50833) == approx((0,30,29.9,-1),abs=1)
    assert dg2dgms(10.2958) == approx((10,17,44.88,1))
    assert dg2dgms(-0.586) == approx((0,35,9.6,-1))
    assert dg2dgms(182.5241667) == (182,31,approx(27,abs=1),1)


def test_hms2h():
    assert hms2h (10,25,11) == approx(10.419722)
    assert hms2h (9,14,55.8) == approx(9.248833)

def test_h2hms():
    assert h2hms(20.352) == approx((20, 21, 07.2))

def test_radians():
    assert np.rad2deg(2.5) == approx(143.239449)

def test_hms2deg():
    assert hms2dg(2,0,0) == 30
    
def test_dms2h():
    assert dms2h(156.3,0) == 10.42

def test_INT():
    assert np.floor(1.5) == 1
    assert np.floor(1.4) == 1
    assert np.floor(-1.5) == -2
    assert np.floor(-1.4) == -2

def test_FIX():
    assert my_fix(1.5)  == 1
    assert my_fix(1.4) == 1
    assert my_fix(-1.5) == -1
    assert my_fix(-1.4) == -1

def test_ABS():
    assert np.abs(-5.0) == 5.0
    assert np.abs(5.4) == 5.4
    assert np.abs(0) == 0

def test_FRAC():
    assert my_frac(1.5) == 0.5
    assert my_frac(-1.5) == 0.5

def test_MOD():    
    assert np.mod(-100,8) == 4
    assert np.mod(-400,360) == 320
    assert np.mod(270,180) == 90
    assert np.mod(-270.8,180) == approx(89.2)
    assert np.mod(390,360) == 30
    assert np.mod(390.5,360) == 30.5
    assert np.mod(-400,360) == 320

def test_ROUND():
    assert np.round(1.4) == 1
    assert np.round(1.8) == 2
    assert np.round(-1.4) == -1
    assert np.round(-1.8) == -2

def test_datefd2jd():
    assert datefd2jd(1957,10,4.81) ==  2436116.31
    assert datefd2jd(333,1,27.5) ==  1842713.0
    assert datefd2jd(2000,1,1.5) ==  2451545.0
    assert datefd2jd(1999,1,1) ==  2451179.5
    assert datefd2jd(1987,1,27) ==  2446822.5
            

def test_is_leap_year():
    assert is_leap_year(1984) == True
    assert is_leap_year(1974) == False
    assert is_leap_year(2000) == True
    assert is_leap_year(1900) == False

def test_datetime2jd():
    assert datetime2jd(2010,11,1,0,0,0) == 2455501.5
    assert datetime2jd(2015,5,10,6,0,0) == 2457152.75
    assert datetime2jd(2015,5,10,18,0,0) == 2457153.25
    assert datefd2jd(1957,10,4.81) == 2436116.31
      

def test_jd2date():
    assert jd2datefd(2369915.5) == (1776,7,4)
    assert jd2datefd(2455323.0) == (2010,5,6.5)
    assert jd2datetime(2456019.37) == (2012,4,1,20,52,approx(48))

def test_dayOfweek():
    assert dayOfweek(1776,7,4) == "Thursday"
    #assert dayOfweek(2011,9,11) == "Sunday"
    assert dayOfweek(1985,2,7) == 'Thursday'

def test_datet2elapsed_days():
    assert datet2elapsed_days(2009,10,30) == 303
    year = 2020
    ndays = datet2elapsed_days(year,3,1)
    assert elapsed_days2date(year,ndays) == (year,3,1)

def test_elapsed_days2date():
    assert elapsed_days2date(1900,250) == (1900,9,7)

def test_ut2gst():
    assert pipe(tc.ut2gst(1987,4,10,0,0,0),tc.h2hms) == approx((13,10,46.3668))


def test_lct2ut():    
    year, month, day, lon = 2014, 12, 12, -77
    *ut , incr = lon_lct2ut(20,0,0,lon,is_dst=False)
    ut = tuple(ut)
    assert ut == (1,0,0)
    # In this case, the incr is 1 so this is the next day
    # because ut2gst needs a date-time we need to add one day
    gst = pipe(ut2gst(year,month,day+incr,*ut),tc.h2hms)
    assert gst == (6,26,approx(34,abs=1))
    *lst, incr = gst2lst(*gst,lon)
    assert tuple(lst) == (1,18,approx(34,abs=1))


def test_lct2ut_kk():    
    year, month, day, lon, is_dst =2000, 7, 5, 60, True
    *gst, incr = lst2gst(5,54,20,lon)
    gst = tuple(gst)
    assert gst == approx((1,54,20))
    ut = gst2ut(year,month,day,*gst)
    # Here we lost precision so we neded to compare in decimal hours
    assert hms2h(*ut[0:-1]) == approx(hms2h(7,0,0),abs=2)
    assert ut2lon_lct(*ut[0:-1],lon)


def test_equat2horiz():
    None
    #assert equat2horiz(25,16.495833,-0.508333) == approx((-20.577738,80.525393))
    #alt, az = equat2horiz(-80,7,degms2deg(49,54,20))
    #assert deg2degms(alt) == approx((51,28,21,-1))
    #assert deg2degms(az) == approx((267,7,4,1))





