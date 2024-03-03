import pandas as pd 
import numpy as np
import datetime as dt
import os
#%matplotlib notebook
import streamlit as st

#from sqlalchemy import create_engine
#from sqlalchemy.engine import URL
#
#
#url = URL.create(
#    drivername="postgresql",
#    username="cece",
#    password="holland2023",
#    host="localhost",
#    port=5432,
#    database="postgres"#"uzh"
#)
#
#engine = create_engine(url)
#connection = engine.connect() # the actual connection to the database
#
#
#from sqlalchemy.orm import sessionmaker
#
#Session = sessionmaker(bind=engine)
#session = Session()

def get_nuclides_dataframe():
    nuclide_name_list=['11C', ' 13N', ' 15O', ' 18F', ' 32P', ' 35S', ' 43Sc', ' 44Sc', ' 47Sc', ' 45Ti', ' 48V', ' 51Cr', ' 51Mn', ' 52Mn', ' 52Fe', ' 55Co', ' 57Ni', ' 60Cu', ' 61Cu', ' 62Cu', ' 64Cu', ' 66Cu', ' 67Cu', ' 62Zn', ' 67Ga', ' 68Ga', ' 69Ge', ' 70As', ' 71As', ' 72As', ' 74As', ' 76As', ' 77As', ' 76Br', ' 77Br', ' 81mKr', ' 82Rb', ' 82mRb', ' 83Sr', ' 89Sr', ' 86Y', ' 90Y', ' 89Zr', ' 97Zr', ' 90Nb', ' 99Mo', ' 94mTc', ' 99mTc', ' 97Ru', ' 105Rh', ' 111Ag', ' 111In', ' 110mIn', ' 123I', ' 124I', ' 125I', ' 131I', ' 127Xe', ' 133Xe', ' 134La', ' 134Ce', ' 153Sm', ' 149Tb', ' 152Tb', ' 155Tb', ' 161Tb', ' 166Ho', ' 165Er', ' 169Er', ' 177Lu', ' 186Re', ' 188Re', ' 192Ir', ' 195mPt', ' 198Au', ' 197Hg', ' 197mHg', ' 201Tl', ' 203Pb', ' 212Pb', ' 212Bi', ' 213Bi', ' 211At', ' 223Ra', ' 225Ac', ' 227Th']
    #nuclides_dataframe=pd.read_sql_table('nuclide', connection)
    return nuclide_name_list

def get_dc_hl(nuclide_name):
    from radioactivedecay import Nuclide
    nuclide_value = nuclide_name
    nuclides_dataframe = get_nuclides_dataframe()
    selected_nuclide=nuclides_dataframe[nuclides_dataframe==nuclide_value]
    nucl=Nuclide(selected_nuclide)
    half_life=nucl.half_life('h')
    decay_constant=np.log(2)/half_life
    print(half_life)                               
    return decay_constant, half_life

def get_organ_list():
    organs_df=pd.read_sql_table('organ', connection)
    organs_list=list(organs_df['name'])
    return sorted(organs_list)


def save_to_csv(data, file_name):
    file_path=os.path.join('Files', file_name)
    data.to_csv(file_path, index=True)
    
def read_from_csv(file_name):
    file_path=os.path.join('Files', file_name)
    data=pd.read_csv(file_path)
    return data
    
#########################
#long function beginning
#########################



def decay_correct(time_exp, date_exp, full_syr_datetime, residual_syr_datetime,injection_datetime, syr_initial_activity, syr_residual_activity, dose_cal_bckg, decay_const):
    
    start_counting_datetime=dt.datetime.combine(date_exp, time_exp)
    full_syr_datetime=dt.datetime.strptime(full_syr_datetime, '%Y-%m-%dT%H:%M:%S.%f')
    residual_syr_datetime=dt.datetime.strptime(residual_syr_datetime, '%Y-%m-%dT%H:%M:%S.%f')
    injection_datetime=dt.datetime.strptime(injection_datetime, '%Y-%m-%dT%H:%M:%S.%f')
    start_counting_datetime= dt.datetime.combine(date_exp, time_exp)
    injected_activity_at_injection= (syr_initial_activity - dose_cal_bckg) * \
    np.exp(-float(decay_const) * ((injection_datetime - full_syr_datetime).total_seconds())/3600 ) - \
    (float(syr_residual_activity)-float(dose_cal_bckg)) * \
    np.exp(-float(decay_const) * (((injection_datetime-residual_syr_datetime).total_seconds())/3600)) 

    injected_activity_at_counting_start= injected_activity_at_injection * np.exp(-decay_const *(((start_counting_datetime-injection_datetime).total_seconds())/3600))
    
    print(start_counting_datetime)
    print(injected_activity_at_injection)
    return injected_activity_at_injection, injected_activity_at_counting_start


#########################
#long function ending
#########################

#def write_anagraphics():
#    from database_creation import Experimenter, session
#    qry_object = session.query(Experimenter).filter(Experimenter.email == st.session_state.email).first()
#    if qry_object is None:
#        session.add(Experimenter(name=st.session_state.name, surname=st.session_state.surname, 
#                                 email=st.session_state.email, institution=st.session_state.institution))
#    else:
#        qry_object.name = st.session_state.name
#        qry_object.surname = st.session_state.surname
#        qry_object.institution = st.session_state.institution 
#    session.commit()
#    
#def write_syringe_table():
#    from database_creation import Syringe, Experimenter, Nuclide, session
#    df=st.session_state.input_table_std
#    
#    qry_object = session.query(Nuclide).filter(Nuclide.name == st.session_state.nuclide).first()
#    nuclide_id=qry_object.id
#    qry_object_e = session.query(Experimenter).filter(Experimenter.email == st.session_state.email).first()
#    experimenter_id=qry_object_e.id
#    counting_start_datetime=dt.datetime.combine(st.session_state.date_exp, st.session_state.time_exp)
#    
#    a1,a2 = decay_correct(st.session_state.time_exp, st.session_state.date_exp, 
#                                                 df['Full Syringe Calibration datetime'].iloc[0],
#                                                 df['Empty Syringe Calibration datetime'].iloc[0],
#                                                 df['Injection datetime'].iloc[0],
#                                                 df['Initial activity syringe (MBq)'].iloc[0],
#                                                 df['Syringe Residual activity (MBq)'].iloc[0],
#                                                 st.session_state.dose_calibrator_bckg,
#                                                 qry_object.decayConstant)
#    
#    s=Syringe(
#            dose_cal_bckg=st.session_state.dose_calibrator_bckg,
#            nuclide_id=nuclide_id,
#            experimenter_id=experimenter_id,
#            mass_full=df['Full syringe before injection (g)'].iloc[0],
#            mass_residual=df['Empty after injection (g)'].iloc[0],
#            activity_full=df['Initial activity syringe (MBq)'].iloc[0],
#            activity_residual=df['Syringe Residual activity (MBq)'].iloc[0],
#            assay_time_full=df['Full Syringe Calibration datetime'].iloc[0],
#            assay_time_residual=df['Empty Syringe Calibration datetime'].iloc[0],
#            injection_datetime=df['Injection datetime'].iloc[0],
#            counting_start_datetime=counting_start_datetime,
#            calc_activity_injected=a1,
#            calc_activity_start_time=a2)
#            
#    session.add(s)
#    session.commit()
#
#    
#def write_calibration_syringe():
#    from database_creation import Calibration_Syringe, Experimenter, Nuclide, session
#    df=st.session_state.calibration_syringe_table
#    print(df)
#    qry_object = session.query(Nuclide).filter(Nuclide.name == st.session_state.nuclide).first()
#    nuclide_id=qry_object.id
#    qry_object_e = session.query(Experimenter).filter(Experimenter.email == st.session_state.email).first()
#    experimenter_id=qry_object_e.id
#    counting_start_datetime=dt.datetime.combine(st.session_state.date_exp, st.session_state.time_exp)
#    
#    a1,a2 = decay_correct(st.session_state.time_exp, st.session_state.date_exp, 
#                                                 df['Full Syringe Calibration datetime'].iloc[0],
#                                                 df['Empty Syringe Calibration datetime'].iloc[0],
#                                                 df['Dilution datetime'].iloc[0],
#                                                 df['Initial activity syringe (MBq)'].iloc[0],
#                                                 df['Residual activity (MBq)'].iloc[0],
#                                                 st.session_state.dose_calibrator_bckg,
#                                                 qry_object.decayConstant)
#    
#    print(a1)
#    print(a2)
#    s=Calibration_Syringe(
#    
#        experimenter_id=experimenter_id,
#        nuclide_id=nuclide_id,
#        dose_cal_bckg=st.session_state.dose_calibrator_bckg,
#        counting_start_datetime=counting_start_datetime,
#        injected_mass=df['Injected mass(g)'].iloc[0], 
#        activity_full=df['Initial activity syringe (MBq)'].iloc[0], 
#        activity_residual=df['Residual activity (MBq)'].iloc[0],
#        assay_time_full=df['Full Syringe Calibration datetime'].iloc[0],
#        assay_time_residual=df['Empty Syringe Calibration datetime'].iloc[0],
#        dilution_datetime=df['Dilution datetime'].iloc[0],
#        calc_activity_injected=a1,
#        calc_activity_start_time=a2 )
#    
#    session.add(s)
#    session.commit()
#    
#
    

"""  
def write_gamma_counter_results():
    from database_creation import Gamma_counter_results, Experimenter, Mouse, Organ, Syring, session
    
    qry_object_experimenter = session.query(Experimenter).filter(Experimenter.email == st.session_state.email).first()
    experimenter_id=qry_object_experimenter.id
    qry_object_mouse = session.query(Mouse).filter(Mouse.experimenter_id == experimenter_id and Mouse.name == st.session_state.mouse_name).first()
    moouse_id=qry_object_mouse.id
    qry_object = session.query(Nuclide).filter(Nuclide.name == st.session_state.nuclide).first()
    nuclide_id=qry_object.id
    qry_object_organ = session.query(Organ).filter(Organ.name == st.session_state.organ).first()
    organ_id=qry_object_organ.id
    
    # retrieve syringe ID
    syringe_id = ...
    
    
    Gamma_counter_results(mouse_id=mouse_id,
                          organ_id=organ_id,
                          syringe_id=syringe_id,
                          tube_tare=Column(Float(), default=0.0)
    tube_full=Column(Float(), default=0.0)
    tube_norm_cpm=Column(Integer(), default=0)
    tube_net=Column(Float(), Computed('gamma_count_results.tube_full-gamma_count_results.tube_tare'))
    #tube_cpm_per_g=Column(Float(), Computed('gamma_count_results.tube_norm_cpm / (gamma_count_results.tube_full-gamma_count_results.tube_tare)'))
    average_background_counts = Column(Float(), default=281.0) # TODO
    calibration_factor = Cloumn(Float(), default=13426520.17) # TODO
    background_corrected_normalised_cpm =Column(Float(), default=0.0) #(gamma_count_results.tube_norm_cpm-gamma_count_results.average_background_counts)/gamma_count_results.tube_net
    calibrated_sample_activity =Column(Float(), default=0.0) #gamma_count_results.background_corrected_normalised_cpm/calibration_factor
    tissue_uptake =Column(Float(), default=0.0) #gamma_count_results.calibrated_sample_activity/syringe.calc_activity_start_time
    suv =Column(Float(), default=0.0) #gamma_count_results.tissue_uptake/mouse_weight*100
                    
"""
