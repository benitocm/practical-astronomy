"""
This module contains the tests for timeconv function
"""
# Standard library imports


# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np

from myastro.orbit import EphemrisInput
from myastro.pert_enckes import calc_eph_by_enckes
from myastro.pert_cowells import calc_eph_by_cowells
from myastro.twobodys import calc_eph_twobodys


import myastro.data_catalog as dc
import myastro.util as ut
import myastro.coord as co


def calc_diff_seconds(my_df, exp_df):
    my_df['r_AU_2'] = my_df['r[AU]']
    my_df['ra_2'] = my_df['ra'].map(co.make_ra)
    my_df['de_2'] = my_df['dec'].str.replace("'","m").str.replace('"',"s").map(co.make_lon)
    cols=['date','ra_2','de_2','r_AU_2']
    df = my_df[cols].copy()
    df = exp_df.merge(my_df, on='date')
    df['dist_ss'] = df.apply(lambda x: ut.angular_distance(x['ra_1'],x['de_1'],x['ra_2'],x['de_2']), axis=1).map(np.rad2deg)*3600.0
    df['dist_ss^2'] = df['dist_ss'] * df['dist_ss']
    return np.sqrt(sum(df['dist_ss^2'])) 


TEST_DATA_PATH= '/home/anybody/wsl_projs/practical-astronomy/data'


def test_body_with_enkes():

    fn = TEST_DATA_PATH+'/jpl_ceres_2020-May-15_2020-Jun-14.csv'
    exp_df = dc.read_jpl_data(fn)    
    eph = EphemrisInput(from_date="2020.05.15.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODYS)
    df = calc_eph_by_enckes(CERES, eph)
    assert calc_diff_seconds(df, exp_df) < 2.6
    

def test_comet_with_enkes():    
    fn = TEST_DATA_PATH+'/jpl_halley_1985-Nov-15_1985-Apr-05.csv'
    exp_df = dc.read_jpl_data(fn)    


    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")


    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)
    df = calc_eph_by_enckes(HALLEY_J2000, eph, 'comet')
    assert calc_diff_seconds(df, exp_df) < 15.11
    


def test_body_with_cowells():
    fn = TEST_DATA_PATH+'/jpl_ceres_2020-May-15_2020-Jun-14.csv'
    exp_df = dc.read_jpl_data(fn)    

    eph = EphemrisInput(from_date="2020.05.15.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODYS)
    df = calc_eph_by_cowells(CERES, eph)
    assert len(df) == len(exp_df)
    assert calc_diff_seconds(df, exp_df) < 2.7


def test_comet_with_cowells():    
    fn = TEST_DATA_PATH+'/jpl_halley_1985-Nov-15_1985-Apr-05.csv'
    exp_df = dc.read_jpl_data(fn)    

    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")


    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)
    df = calc_eph_by_cowells(HALLEY_J2000, eph, 'comet')
    assert len(df) == len(exp_df)
    assert calc_diff_seconds(df, exp_df) < 430.20


def test_comet_with_twobodys_B1950():    

    fn = TEST_DATA_PATH+'/jpl_halley_1985-Nov-15_1985-Apr-05.csv'
    exp_df = dc.read_jpl_data(fn)    

    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")

    HALLEY_1950 = dc.CometElms(name="1P/Halley",
                epoch_name=None ,
                q =  0.5870992 ,
                e = 0.9672725 ,
                i_dg = 162.23932 ,
                Node_dg = 58.14397 ,
                w_dg = 111.84658 ,
                tp_str = "19860209.44",
                equinox_name = "B1950")

    df = calc_eph_twobodys(HALLEY_1950, eph, type='comet')
    assert len(df) == len(exp_df)
    assert calc_diff_seconds(df, exp_df) < 354.11


def test_comet_with_twobodys_J2000():    

    fn = TEST_DATA_PATH+'/jpl_halley_1985-Nov-15_1985-Apr-05.csv'
    exp_df = dc.read_jpl_data(fn)    

    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")

    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)

    df = calc_eph_twobodys(HALLEY_J2000, eph, type='comet')
    assert len(df) == len(exp_df)
    assert calc_diff_seconds(df, exp_df) < 69418.96

    
def test_body_with_twobodys():    

    fn = TEST_DATA_PATH+'/jpl_ceres_2020-May-15_2020-Jun-14.csv'
    exp_df = dc.read_jpl_data(fn)    

    eph = EphemrisInput(from_date="2020.05.15.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODYS)

    df = calc_eph_twobodys(CERES, eph, type='body')
    assert calc_diff_seconds(df, exp_df) < 2.4

    
    
