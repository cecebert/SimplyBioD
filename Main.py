import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from functions import *
import datetime as dt
import sys
import time

import re
sys.path.append('../')
from streamlit_sortables import sort_items
from radioactivedecay import Nuclide

st.set_page_config(page_title='SimplyBioD', layout='wide', initial_sidebar_state ='expanded')

def get_nuclides_dataframe():
    nuclide_name_list=['11C', ' 13N', ' 15O', ' 18F', ' 32P', ' 35S', ' 43Sc', ' 44Sc', ' 47Sc', ' 45Ti', ' 48V', ' 51Cr', ' 51Mn', ' 52Mn', ' 52Fe', ' 55Co', ' 57Ni', ' 60Cu', ' 61Cu', ' 62Cu', ' 64Cu', ' 66Cu', ' 67Cu', ' 62Zn', ' 67Ga', ' 68Ga', ' 69Ge', ' 70As', ' 71As', ' 72As', ' 74As', ' 76As', ' 77As', ' 76Br', ' 77Br', ' 81mKr', ' 82Rb', ' 82mRb', ' 83Sr', ' 89Sr', ' 86Y', ' 90Y', ' 89Zr', ' 97Zr', ' 90Nb', ' 99Mo', ' 94mTc', ' 99mTc', ' 97Ru', ' 105Rh', ' 111Ag', ' 111In', ' 110mIn', ' 123I', ' 124I', ' 125I', ' 131I', ' 127Xe', ' 133Xe', ' 134La', ' 134Ce', ' 153Sm', ' 149Tb', ' 152Tb', ' 155Tb', ' 161Tb', ' 166Ho', ' 165Er', ' 169Er', ' 177Lu', ' 186Re', ' 188Re', ' 192Ir', ' 195mPt', ' 198Au', ' 197Hg', ' 197mHg', ' 201Tl', ' 203Pb', ' 212Pb', ' 212Bi', ' 213Bi', ' 211At', ' 223Ra', ' 225Ac', ' 227Th']
    #nuclides_dataframe=pd.read_sql_table('nuclide', connection)
    return nuclide_name_list

def get_dc_hl(nuclide_name):
    nuclide_value = nuclide_name
    nuclides_dataframe = get_nuclides_dataframe()
    selected_nuclide=nuclide_name
    nucl=Nuclide(selected_nuclide)
    half_life=nucl.half_life('h')
    decay_constant=np.log(2)/half_life  
    print(nucl)
    return decay_constant, half_life

st.title('Simply BioD: online biodistribution tool')

with st.sidebar:
    st.header('Quickstart tutorial', divider='blue')
    st.subheader('Experimenter information and experimental details')
    with st.expander(label='Section explanation'):
        
        st.markdown('<div style="text-align: justify;">In this section you can input the anagraphic information of the experimenter. On the right side, you can choose the nuclide and introduce the date and time at which every decay correction will be carried. </div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
    st.divider()
    st.subheader('Gamma counter calibration')
    with st.expander(label='Section explanation'):
        
    
        st.markdown('<div style="text-align: justify;">Here you can insert the data correspondent to the calibration of the gamma counter and the background of the dose calibrator. All of the tables are dynamic and by clicking on the last row of each one, you will create a new row which can also be deleted. </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;"> In the "Standard Samples Parameters" tables you will be required to insert the data used for the syringe or tube used to calibrate the activity per gram of the injectate. This will be then correlated to the diluted standard table (named: "Calibration standard") that will allow to produce a calibration constant expressed in MBq / (CPM x g) </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
            
        st.markdown('<div style="text-align: justify;"> In the results section you will find all the descriptive data that will be used for the following calculations. </div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader('Injection parameters')
    
    with st.expander(label='Section explanation'):
        
        st.markdown('<div style="text-align: justify;"> You can insert as many rows as the animals that you employed in the experiments. </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;"> Notice that each row has to be fully compiled by the user to be taken into account for the calculations, otherwise the data will not be included in the next sections </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader('Organs gamma counting')
    
    with st.expander(label='Section explanation'):
        
        st.markdown('<div style="text-align: justify;"> Select the organs from the dropdown menu, you can remove unwanted selections by pressing the remove symbol on each selection. </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;"> Upon selection of the first organ, two tables will be generated. The right one corresponds to the mass of each organ sorted by animal number. The left table corresponds to the CPM of each organ of each mouse. </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader('Plotting the inserted values')
    
    with st.expander(label='Section explanation'):
        
        st.markdown('<div style="text-align: justify;"> With the first widget, you can select the sorting order for plotting the biodistribution data. You can simply drag and drop the organs in the order that you prefer. </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;"> Here the aggregated data plots are customizable in terms of color, opacity, size ratio and font sizes for the labels and the axis titles. </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;"> The contrast ratio plots allow you to select the pivot point for the normalization. This corresponds to a rescaling of each contrast ratio respect to the selected organ (which will be set to 1 by default). </div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify;">    </div>', unsafe_allow_html=True)
    
    st.divider()
    
    
    
    
    
    
### setting up the info input

#Anagraphic input

columns_exp_info = st.columns(2)

columns_exp_info[0].subheader("Experimenter information", divider='blue')



name=columns_exp_info[0].text_input('Enter name', key='__name')

surname=columns_exp_info[0].text_input('Enter surname', key='_surname')

email=columns_exp_info[0].text_input('Enter your email', key='_email')

institution=columns_exp_info[0].text_input('Enter your institution name', key='_institution')


#Experiment details input


columns_exp_info[1].subheader("Experimental details", divider='blue')

df_nuclides = get_nuclides_dataframe()

pretty_names_nuclides=[]
for i in range(len(df_nuclides)):

    name=re.split('(\d+)',df_nuclides[i])
    
    name_pretty='$^{'+str(name[1])+'}$'+str(name[2])
    
    pretty_names_nuclides.append(name_pretty)
    

nuclide_selection=columns_exp_info[1].selectbox("Select the Nuclide", df_nuclides)

(dc, hl) = get_dc_hl(nuclide_selection)  

columns_exp_info[1].success('Half-life of the nuclide: ' +str(np.round(hl, 2)) + ' h     ||     Decay constant: ' + str(np.round(dc, 3)) + ' h$^{-1}$')

organs_for_widget=str.split('Blood Tumor Heart Lungs Liver Spleen Stomach Pancreas Kidney Sm.Intestine Lg.Intestine Fat Muscle Bone Skin Urine Gallbladder Tail Lymphnode Sal.Gland Occ.Bone Duodenum Jejunum Ileum Cecum Asc.Colon Desc.Colon Rectum Brain Adr.Gland')

columns_exp_info[1].info('  Notice: The following datetime inputs will be the reference point to all decay corrections', icon='ℹ️')

columns_exp_info_2 = columns_exp_info[1].columns(2)

date_exp=columns_exp_info_2[0].date_input("Date of Start of counting time", key='_date_exp', format='DD/MM/YYYY')

time_exp=columns_exp_info_2[1].time_input("Start of counting time",step=60, key='_time_exp')



anagraphic_info=('Name',name, 'Surname',surname,'Email',email,'Institution', institution)

exp_info=('Nuclide', nuclide_selection,'Date',date_exp,'Start of counting time', time_exp)


mega_list=[]

mega_list.extend(anagraphic_info)
mega_list.extend(exp_info)



#Gamma counter calibration setup

st.header('Gamma counter calibration', divider='blue')

st.subheader('Dose calibrator background')

st.caption('Background of the dose calibrator machine')

dose_cal_bckg=st.number_input(label='Value in MBq', step=1e-5, min_value=0.0, value=0.0001, format='%.5f')

st.subheader('Standard Samples Parameters')

st.caption('Input values for the standard activity sample')

input_table_std =pd.DataFrame(columns=['Injected mass(g)',
                                           'Initial activity syringe (MBq)',
                                           'Residual activity (MBq)',
                                           'Full Syringe Calibration datetime',
                                           'Empty Syringe Calibration datetime',
                                           'Dilution datetime'], index=range(1))


table1=st.data_editor(input_table_std, num_rows='fixed',hide_index=True,
               column_config={
        'Injected mass (g)': st.column_config.NumberColumn(format='%f', required=True),
        'Initial activity syringe (MBq)' : st.column_config.NumberColumn(format='%f', required=True),
        'Residual activity (MBq)' : st.column_config.NumberColumn(format='%f', required=True),
        'Full Syringe Calibration datetime' : st.column_config.DatetimeColumn(required=True),
        'Empty Syringe Calibration datetime': st.column_config.DatetimeColumn(required=True),
        'Dilution datetime' : st.column_config.DatetimeColumn(required=True),                      
    },key='calibration_syringe', use_container_width=True)

input_table_std=pd.DataFrame(table1)

mega_list.extend([input_table_std.to_dict(orient='tight',index=False)])

    
# Configuring the Calibration Standard Table

st.subheader('Calibration Standard')

st.caption('Input')

input_table_gc_calib =pd.DataFrame(columns=['Tare (g)',
                                        'Container + water (g)',
                                        'Container + water + Std. Syringe (g)'], index=range(1))

table2=st.data_editor(input_table_gc_calib, num_rows='fixed', 
                   column_config={
                       'Tare (g)': st.column_config.NumberColumn(format='%f'),
                       'Container + water (g)': st.column_config.NumberColumn(format='%f'),
                       'Container + water + Std. Syringe (g)': st.column_config.NumberColumn(format='%f')
                       },use_container_width=True, hide_index=True)

st.caption('Output')

total_diluted_std=table2[table2.columns[2]]-table2[table2.columns[0]]

st.table(pd.DataFrame(total_diluted_std, columns=['Total diluted standard (g)']))

total_diluted_std=np.array(total_diluted_std)[0]

mega_list.extend([table2.to_dict(orient='tight',index=False), 
                  'Total diluted std.', 
                  total_diluted_std])



# Configuring the Blanking gamma counter table


st.subheader('Gamma counter background')

input_table_blank_tubes =pd.DataFrame(columns=['Counts (CPM)'])

st.caption('Input')

table3= st.data_editor(input_table_blank_tubes, num_rows='dynamic', column_config= 
                   {
                       'Counts (CPM)': st.column_config.NumberColumn(format='%f')
                   },key='blank_tubes')


st.session_state['gamma_counter_blank']=table3

average_bckg=table3.mean()

mega_list.extend(['Dose calibrator bckg. (MBq)',
                  dose_cal_bckg,
                  table3,
                  'Average Background (CPM)',
                  average_bckg[0]])

st.caption('Average Background:')

st.table(pd.DataFrame(average_bckg, columns=['Avg. background (CPM)']))

st.subheader('Standard Syringe/Tube calibration')

st.caption('Insert the data from the standard samples')

input_table_calib_syr=pd.DataFrame(columns=['Sample Mass (net,g)','Counts (CPM)'])



st.caption('Input')

table4= st.data_editor(input_table_calib_syr, 
                       num_rows='dynamic', 
                       column_config= 
                       {
                           'Sample Mass (net,g)': st.column_config.NumberColumn(format='%f'),
                           'Counts (CPM)': st.column_config.NumberColumn(format='%f')
                       }, 
                       key='standard_syringes')


bckg_corrected_norm_cpm_per_g= (table4[table4.columns[1]] - average_bckg[0]) / table4[table4.columns[0]]


st.caption('Output')

st.table((pd.DataFrame(bckg_corrected_norm_cpm_per_g,columns=['Nomalised CPM/g'])))


avg_bckg_corrected_norm_cpm_per_g=bckg_corrected_norm_cpm_per_g.mean()


total_bckg_corrected_norm_cpm_per_g=avg_bckg_corrected_norm_cpm_per_g*total_diluted_std



total_bckg_corrected_norm_cpm_per_g_injectate= total_bckg_corrected_norm_cpm_per_g / float(table1[table1.columns[0]][0])


mega_list.extend([pd.DataFrame(table3).to_dict(orient='tight',index=False), 
                  'Average gamma counter bckgr (CPM)', 
                  average_bckg[0], 
                  table4.to_dict(orient='tight',index=False), 
                  'Background-corrected norm. CPM/g', 
                  bckg_corrected_norm_cpm_per_g,
                 'Total bckg. corrected norm. CPM/g', 
                 total_bckg_corrected_norm_cpm_per_g,
                 'Total bckg. corrected norm. CPM/g of injectate',
                 total_bckg_corrected_norm_cpm_per_g_injectate])


st.divider()

st.subheader('Results')

st.caption('Gamma counter blanking and calibration results')

st.table(pd.DataFrame([avg_bckg_corrected_norm_cpm_per_g, total_bckg_corrected_norm_cpm_per_g,total_bckg_corrected_norm_cpm_per_g_injectate], index=['Average Normalised Bckg corrected : ', 'Total Normalised Bckg corrected (Standard tube): ','Total Normalised Bckg corrected CPM/g (Injectate)'], columns=['CPM/g']))

st.caption('Decay correction results')



try:
    
    corrected_syringe_activity_at_injection, corrected_syringe_activity_at_counting_start =decay_correct(time_exp, date_exp, table1['Full Syringe Calibration datetime'][0], table1['Empty Syringe Calibration datetime'][0], table1['Dilution datetime'][0], table1['Initial activity syringe (MBq)'][0], table1['Residual activity (MBq)'][0], dose_cal_bckg, dc )
    
    cpm_per_mbq_per_g=total_bckg_corrected_norm_cpm_per_g_injectate/(corrected_syringe_activity_at_counting_start/float(table1[table1.columns[0]][0]))
        
    st.table(pd.DataFrame([corrected_syringe_activity_at_injection, corrected_syringe_activity_at_counting_start, cpm_per_mbq_per_g ],
                                            index=['Corrected syr, activity (injection)', 'Corrected syr. activity (counting start)', 'CPM / MBq per g'], columns=['Value']))
    
    mega_list.extend(['Corrected syringe activity at injection (MBq)',
                      corrected_syringe_activity_at_injection,
                      'CPM per MBq per g of injectate',
                      cpm_per_mbq_per_g,
                      'Corrected syr. activity at counting start) (MBq)',
                      corrected_syringe_activity_at_counting_start])

    mega_list=pd.DataFrame(mega_list).to_csv()
    
    columns_download_button=st.columns(2)
    
    filename_1=columns_download_button[0].text_input(label='File Name', value='Data_1')
    
    columns_download_button[0].download_button(label='Download all data above as .csv', 
                      data=mega_list, 
                      file_name=str(filename_1)+'.csv', 
                      type='primary'
                      )
    
    
    
except:
    st.warning('You are missing some information for the decay corrections, please check the sections above', icon='⚠️')

st.subheader('Injection parameters', divider='blue')
    
st.caption('Insert the data from the injections, you can add or remove rows dynamically in this table')



input_table = pd.DataFrame(columns=['Mouse mass (g)',
                                    
                                    'Full syringe before injection (g)',
                                    
                                    'Empty after injection (g)',
                                    
                                    'Initial activity syringe (MBq)',
                                    
                                    'Residual activity (MBq)',
                                    
                                    'Full Syringe Calibration time',
                                    
                                    'Empty Syringe Calibration time',
                                    
                                    'Injection time',])


table5 = st.data_editor(input_table, num_rows='dynamic', 
               column_config={
    
        'Mouse mass (g)': st.column_config.NumberColumn(format='%f',required=True),
                   
        'Full syringe before injection (g)' : st.column_config.NumberColumn(format='%f',required=True),
                   
        'Empty after injection (g)' : st.column_config.NumberColumn(format='%f',required=True),
                   
        'Initial activity syringe (MBq)' : st.column_config.NumberColumn(format='%f',required=True),
                   
        'Residual activity (MBq)' : st.column_config.NumberColumn(format='%f',required=True),
                   
        'Full Syringe Calibration time' : st.column_config.DatetimeColumn(required=True),
                   
        'Empty Syringe Calibration time': st.column_config.DatetimeColumn(required=True),
                   
        'Injection time' : st.column_config.DatetimeColumn(required=True),
                       }, use_container_width=True)

table5=pd.DataFrame(table5)


st.header('Organs gamma counting', divider='blue')

org_list=st.multiselect("Select the organs used in the experiments", options=organs_for_widget, key='organs_list')



if not st.session_state.organs_list: 
        st.warning(
            'No organs are selected, please check your entries above', icon='⚠️')
else:
        
    cols1 = st.columns(2)
    
    cols1[0].caption('Insert the net mass of the organs (g)')
    
    input_table_organs_cpms=pd.DataFrame(columns=['Mouse ' + str(x+1) for x in range(len(table5))], 
                                         index=st.session_state.organs_list)
    
    input_table_organs_mass=pd.DataFrame(columns=['Mouse ' + str(x+1) for x in range(len(table5))], 
                                         index=st.session_state.organs_list)
    
    dic_col_org=defaultdict(list)
    
    for organ in st.session_state.organs_list:
        
        dic_col_org[organ]=st.column_config.NumberColumn(format='%f')

    table7=cols1[0].data_editor(input_table_organs_cpms, num_rows='fixed', column_config=dic_col_org,key='organs cpms', use_container_width=True)
        
    cols1[1].caption('Insert the CPM of each organ')
    
    table6= cols1[1].data_editor(input_table_organs_cpms, num_rows='fixed', column_config=dic_col_org, use_container_width=True)
            
    table6_df=pd.DataFrame(table6)
    
    vis = pd.DataFrame(columns=['organ','mouse','value'])
    
    st.header('Plotting the inserted values', divider='blue')
    
    try:
        mega_list_2=[]
        
        mega_list_2.extend(['Mouse Data (user input)', table5.to_dict(orient='tight', index=False),'Organ mass (g) (user input)', table7, 'Organ CPMs (user input)', table6])
        
        table7_float=table7.astype(float)
        
        table6_float=table6.astype(float)
        
        table8=(table6_float-float(average_bckg)).divide(table7_float)
        
        table8
        
        table8_alt_formatted=table8.T.reset_index().melt(id_vars='index')
        
        table8_alt_formatted.columns=['mouse', 'organ', 'CPM / g']
        
        table9=table8.divide(cpm_per_mbq_per_g).T
        
        table_9_alt_formatted=table8.divide(cpm_per_mbq_per_g).T.reset_index().melt(id_vars='index')
        
        table_9_alt_formatted.columns=['mouse', 'organ', 'MBq / g']
        
        st.markdown('Select the order of the organs by drag and dropping the items in the list')
        
        sorted_organs=sort_items(org_list)
        
        table_9_alt_formatted['organ']=pd.Categorical(table_9_alt_formatted['organ'], categories=sorted_organs)
        
        table10=table_9_alt_formatted.sort_values(by='organ')
        
        table11=table10.pivot_table(index='organ', columns='mouse', values='MBq / g')
        
        table12=[]
    
    
        for i in range(len(table5)):
            mouse_decay_correction_at_injection, mouse_decay_correction_at_counting_start=decay_correct(time_exp, 
                                                                                                        date_exp, 
                                                                                                        table5['Full Syringe Calibration time'][i],  
                                                                                                        table5['Empty Syringe Calibration time'][i], 
                                                                                                        table5['Injection time'][i], 
                                                                                                        table5['Initial activity syringe (MBq)'][i], 
                                                                                                        table5['Residual activity (MBq)'][i], 
                                                                                                        dose_cal_bckg, dc  )
            
    
    
            table12.append([mouse_decay_correction_at_counting_start,mouse_decay_correction_at_injection])
            
        
        table12=pd.DataFrame(table12, columns=['Activity at counting start, MBq', 'Activity at injection, MBq'])
        
        
        table13=table11.divide(table12['Activity at counting start, MBq'].values) *100
        
        table13
        
        table13_alt=pd.melt(table13.T.reset_index(),id_vars='mouse')
        
        table14=[]
        
        for i in range(len(table5)):
            
            suv=table13[table13.columns[i]]*table5['Mouse mass (g)'].values[i] /100
            
            table14.append(suv)
            
        table14=pd.DataFrame(table14)
        
        
        table14_alt=table14.T.reset_index().melt(id_vars='organ')
        
        table14_alt.columns=['organ', 'mouse', 'value']
        
        
        
        table15=pd.DataFrame([table14.mean(), table14.std()], index=['Mean SUV', 'Std. Dev. SUV'])
        
        table15_alt=table15.T.reset_index().melt(id_vars='organ')
        
        table17=pd.DataFrame([table13.T.mean(), table13.T.std()], index=['Mean ID/g %', 'Std. Dev. ID/g %'])
        
        table17_alt=table17.T.reset_index().melt( value_vars='value', var_name='organ')
        
        table17=table17.T.reset_index()
        
        st.subheader('Aggregated data plots')
    
        st.markdown('Customize your mean SUV plot')
        
        plot1_customisation=st.columns(3)
        
        color13=plot1_customisation[0].color_picker(label='Plot color', value='#009119', key='color13')
        
        opacity13=plot1_customisation[0].number_input(label='Plot opacity', min_value=0.1, max_value=1., step=0.01, value=0.5 , key='opactiy13')
        
        p1_width=plot1_customisation[1].number_input('Width of the plot',value=400, min_value=100, step=10, max_value=2000)
        
        p1_heigth=plot1_customisation[1].number_input('Height of the plot',value=300, min_value=100, step=10, max_value=2000)
        
        p1_lab_fontsize=plot1_customisation[2].number_input('Label fontsize', value=11.,min_value=1., step=0.5, max_value=50., key='p1lfontsize')
        
        p1_axtitle_fontsize=plot1_customisation[2].number_input('Axis titles fontsize', value=15.,min_value=1., step=0.5, max_value=50., key='p1axtitfontsize')
        
        chart1=alt.Chart(table14_alt).mark_bar(color=color13, opacity=opacity13).encode(alt.X('organ', sort=None).title('Organ'), alt.Y('mean(value)').title('Mean SUV (g)'))
        
        chart2=alt.Chart(table14_alt).mark_errorbar(extent='stdev', ticks=True, color='black', size=8).encode(alt.X('organ', sort=None), alt.Y('mean(value)').title(''))
        
        mega_list_2.extend(['Mean SUV (g)',table14, 'Descriptive stats. on SUV', table14.describe()])
         
        st.divider()
        
        plot1_results=st.columns(2)
        
        
        st.altair_chart(alt.layer(chart1, chart2, data=table14_alt).configure_axis(labelColor='black', labelFont='Arial', titleColor='Black', titleFont='Arial', labelFontWeight='normal', labelFontSize=p1_lab_fontsize, titleFontWeight='normal', titleFontSize=p1_axtitle_fontsize).properties(width=p1_width,height=p1_heigth)
        , theme=None)
        
        st.caption('Plot values and descriptive statistics')
        
        st.table(table14.describe())
        
        st.divider()
        
        st.markdown('Customize your Mean % ID / g plot')
        
        plot2_customisation=st.columns(3)
        
        color9=plot2_customisation[0].color_picker(label='Plot color', value='#99007d')
        
        opacity9=plot2_customisation[0].number_input(label='Plot opacity', min_value=0.1, max_value=1., step=0.01, value=0.5)
        
        chart3=alt.Chart(table13_alt).mark_bar(color=color9, opacity=opacity9).encode(alt.X('organ', sort=None), alt.Y('mean(value)').title('Mean %ID / g'))
        
        chart4=alt.Chart(table13_alt).mark_errorbar(extent='stdev', ticks=True, color='black', size=8).encode(alt.X('organ', sort=None).title('Organ'), alt.Y('mean(value)').title(''))
        
        mega_list_2.extend(['Mean %ID/g',table13.T, 'Descriptive stats. on Mean %ID/g', table13.describe()])

        p2_width=plot2_customisation[1].number_input('Width of the plot (pixels)', value=400,min_value=100, step=10, max_value=2000, key='p2width')
        
        p2_heigth=plot2_customisation[1].number_input('Height of the plot (pixels)', value=300,min_value=100, step=10, max_value=2000, key='p2height')
        
        p2_lab_fontsize=plot2_customisation[2].number_input('Label fontsize', value=11.,min_value=1., step=0.5, max_value=50., key='p2lfontsize')
        
        p2_axtitle_fontsize=plot2_customisation[2].number_input('Axis titles fontsize', value=15.,min_value=1., step=0.5, max_value=50., key='p2axtitfontsize')
        
        
        st.divider()
        
        plot2_results=st.columns(2)
        
        st.altair_chart(alt.layer(chart3, chart4, data=table13_alt).configure_axis(
            labelColor='black', 
            labelFont='Arial', 
            titleColor='Black', 
            titleFont='Arial', 
            labelFontWeight='normal', 
            labelFontSize=p2_lab_fontsize, 
            titleFontWeight='normal', 
            titleFontSize=p2_axtitle_fontsize).properties(
            width=p2_width,
            height=p2_heigth)
        , theme=None)
        
        st.caption('Plot values and descriptive statistics')
        
        st.table(table13.T.describe())
        
        st.divider()
        
        ## Insert contrast ratio elaboration here
        
        st.subheader('Contrast ratio')
        
        st.markdown('Based on the % ID / g')
        
        contrast_pivot=st.selectbox("Select the organ to use as pivot for the calculation of the contrast ratio", options=org_list, key='organ pivot')
        
        table_contrast=pd.DataFrame(table13.astype('float').reset_index())
    
        divisor=table_contrast.iloc[table_contrast[table_contrast['organ']==contrast_pivot].index]
    
        table_contrast=table_contrast.drop(columns='organ').astype('float') / np.array(divisor.drop(columns='organ').astype('float'))
        
        table_contrast.index=table13.index
        
        table_contrast1=table_contrast.reset_index().melt(id_vars='organ')
        
        table_contrast_stats=table_contrast.T.describe().T
        
        table_contrast_alt=table_contrast.reset_index().melt(id_vars='organ')
                
        chart_contrast_id=alt.Chart(table_contrast_alt).mark_bar().encode(alt.X('organ').title('Organ',sort=None), alt.Y('mean(value)').title('Mean Contrast Ratio (% ID/g)'))
        
        errorbars_contrast_id=alt.Chart(table_contrast_alt).mark_errorbar(extent='stdev', ticks=True, color='black', size=8).encode(alt.X('organ',sort=None).title('Organ'), alt.Y('mean(value)').title('Mean Contrast Ratio (% ID/g)'))
        
        st.altair_chart(alt.layer(chart_contrast_id,errorbars_contrast_id, data=table_contrast_alt).configure_axis(
            labelColor='black', 
            labelFont='Arial', 
            titleColor='Black', 
            titleFont='Arial', 
            labelFontWeight='normal', 
            labelFontSize=p2_lab_fontsize, 
            titleFontWeight='normal', 
            titleFontSize=p2_axtitle_fontsize).properties(
            width=p2_width,
            height=p2_heigth), theme=None)
        
        mega_list_2.extend(['Contrast ratio', table_contrast, 'Descriptive stats. on contrast ratio', table_contrast.T.describe()])
        
        st.markdown('Contrast ratio based on the % ID /g')
        
        st.table(table_contrast.T.describe())
        
        table_contrast=pd.DataFrame(table14.T.astype('float').reset_index())
        
        divisor=table_contrast.iloc[table_contrast[table_contrast['organ']==contrast_pivot].index]
        
        table_contrast=table_contrast.drop(columns='organ').astype('float') / np.array(divisor.drop(columns='organ').astype('float'))
        
        table_contrast.index=table13.index
        
        table_contrast1=table_contrast.reset_index().melt(id_vars='organ')
        
        table_contrast_stats=table_contrast.T.describe().T
        
        table_contrast_alt=table_contrast.reset_index().melt(id_vars='organ')
                
        chart_contrast_id=alt.Chart(table_contrast_alt).mark_bar().encode(alt.X('organ').title('Organ',sort=None), alt.Y('mean(value)').title('Mean Contrast Ratio (SUV)'))
        
        errorbars_contrast_id=alt.Chart(table_contrast_alt).mark_errorbar(extent='stdev', ticks=True, color='black', size=8).encode(alt.X('organ',sort=None).title('Organ'), alt.Y('mean(value)').title('Mean Contrast Ratio (SUV)'))
        
        st.markdown('Based on the SUV')
        
        st.altair_chart(alt.layer(chart_contrast_id,errorbars_contrast_id, data=table_contrast_alt).configure_axis(
            labelColor='black', 
            labelFont='Arial', 
            titleColor='Black', 
            titleFont='Arial', 
            labelFontWeight='normal', 
            labelFontSize=p2_lab_fontsize, 
            titleFontWeight='normal', 
            titleFontSize=p2_axtitle_fontsize).properties(
            width=p2_width,
            height=p2_heigth), theme=None)
        
        st.markdown('Contrast ratio based on the SUV')
        
        st.table(table_contrast.T.describe())
        
        st.divider()
        ####
        
        st.header('Individual data plots')
        
        st.subheader('CPM/g for each organ or grouped by mouse')
        
        chart5=alt.Chart(table8_alt_formatted).mark_bar().encode(alt.Column('organ'), alt.X('mouse').title(''), alt.Y('CPM / g', axis=alt.Axis(tickCount=5, format=".1e")).title('Norm. CPM/g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        chart6=alt.Chart(table8_alt_formatted).mark_bar().encode(alt.Column('mouse'), alt.X('organ').title(''), alt.Y('CPM / g', axis=alt.Axis(tickCount=5, format=".1e")).title('Norm. CPM/g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        mega_list_2.extend(['Norm.CPM/g', table8.T])
        
        st.altair_chart(chart5,theme=None)
        
        st.altair_chart(chart6,theme=None)
        
        st.markdown('CPM / g table')
        
        st.table(table8.T)
        
        st.divider()
        
        st.subheader('Activity/g for each organ or grouped by mouse')
        
        chart7=alt.Chart(table_9_alt_formatted).mark_bar().encode(alt.Column('organ'), alt.X('mouse').title(''), alt.Y('MBq / g', axis=alt.Axis(tickCount=5)).title('MBq / g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        chart8=alt.Chart(table10).mark_bar().encode(alt.Column('mouse'), alt.X('organ', sort=None).title(''), alt.Y('MBq / g', axis=alt.Axis(tickCount=5)).title('MBq / g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        mega_list_2.extend(['MBq/g Table', table11.T])
        
        st.altair_chart(chart7,theme=None)
        
        st.altair_chart(chart8,theme=None)
        
        st.markdown('Activity / g table (MBq/g)')
        
        st.table(table11.T)
        
        st.divider()
        
        st.subheader('% ID/g for each organ or grouped by mouse')
        
        chart9=alt.Chart(table13_alt).mark_bar().encode(alt.Column('organ'), alt.X('mouse', sort=None), alt.Y('value').title('ID / g (%)'), alt.Color('mouse').scale(scheme='tableau20'))
    
        chart10=alt.Chart(table13_alt).mark_bar().encode(alt.Column('mouse'), alt.X('organ', sort=None), alt.Y('value').title('ID / g (%)'), alt.Color('mouse').scale(scheme='tableau20'))
        
        st.altair_chart(chart9,theme=None)
        
        st.altair_chart(chart10,theme=None)
        
        mega_list_2.extend(['% ID/g Table', table13.T])
        
        st.markdown('% Injected dose per gram table')
        
        st.table(table13.T)
        
        st.divider()
        
        st.subheader('SUV for each organ or grouped by mouse')
        
        chart11=alt.Chart(table14_alt).mark_bar().encode(alt.Column('organ'), alt.X('mouse', sort=None).title(''), alt.Y('value').title('SUV (g)'), alt.Color('mouse').scale(scheme='tableau20'))
        
        chart12=alt.Chart(table14_alt).mark_bar().encode(alt.Column('mouse'), alt.X('organ', sort=None).title(''), alt.Y('value').title('SUV (g)'), alt.Color('mouse').scale(scheme='tableau20'))
        
        st.altair_chart(chart11,theme=None)
        
        st.altair_chart(chart12,theme=None)
        
        st.markdown('SUV table')
        
        mega_list_2.extend(['SUV Table', table14])
        
        st.table(table14)
        
        mega_list_2=pd.DataFrame(mega_list_2).to_csv()
    
        columns_download_button_2=st.columns(2)
        
        filename_2=columns_download_button_2[0].text_input(label='File Name', value='Data_2', key='asdasdf')
        
        columns_download_button_2[0].download_button(label='Download all data above as .csv', 
                      data=mega_list_2, 
                      file_name=str(filename_2)+'.csv', 
                      type='primary', key='asdas'
                      )
        
        
    except:
        st.warning('Missing data, plotting will not begin until all data will be provided')
        