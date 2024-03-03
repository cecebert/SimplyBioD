import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from functions import *
import sys
import time
sys.path.append('../')
from streamlit_sortables import sort_items

st.set_page_config(page_title='SimplyBioD',layout="wide")



st.title('Page for biodistribution experiments')

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



columns_exp_info[0].text_input('Enter name', key='__name')

columns_exp_info[0].text_input('Enter surname', key='_surname')

columns_exp_info[0].text_input('Enter your email', key='_email')

columns_exp_info[0].text_input('Enter your institution name', key='_institution')


#Experiment details input


columns_exp_info[1].subheader("Experimental details", divider='blue')

df_nuclides = get_nuclides_dataframe()

if 'nuclide' not in st.session_state:
    st.session_state['nuclide']='11C'
if '_nuclide' not in st.session_state:
    st.session_state['_nuclide']='11C'

nuclide_selection=columns_exp_info[1].selectbox("Select the Nuclide", df_nuclides, key = 'nuclide')

(dc, hl) = get_dc_hl(nuclide_selection)  

organs_for_widget=str.split('Blood Tumor Heart Lungs Liver Spleen Stomach Pancreas Kidney Sm.Intestine Lg.Intestine Fat Muscle Bone Skin')

columns_exp_info[1].info('  Notice: The following datetime inputs will be the reference point to all decay corrections', icon='ℹ️')

columns_exp_info_2 = columns_exp_info[1].columns(2)

date_exp=columns_exp_info_2[0].date_input("Date of the experiment", key='_date_exp')

time_exp=columns_exp_info_2[1].time_input("Starting time of the experiment",step=60, key='_time_exp')

#Gamma counter calibration setup

st.header('Gamma counter calibration', divider='blue')

columns_gamma_cnt_cal=st.columns(3)

columns_gamma_cnt_cal[0].subheader('Dose calibrator background')

columns_gamma_cnt_cal[0].caption('Background of the dose calibrator machine')

dose_cal_bckg=columns_gamma_cnt_cal[0].number_input(label='Value in MBq', step=1e-5, min_value=0.0, value=0.0001, format='%.5f')

columns_gamma_cnt_cal[1].subheader('Standard Samples Parameters')

columns_gamma_cnt_cal[1].caption('Input values for the standard activity sample')

input_table_std =pd.DataFrame(columns=['Injected mass(g)',
                                           'Initial activity syringe (MBq)',
                                           'Residual activity (MBq)',
                                           'Full Syringe Calibration datetime',
                                           'Empty Syringe Calibration datetime',
                                           'Dilution datetime'], index=range(1))


table1=columns_gamma_cnt_cal[1].data_editor(input_table_std, num_rows='fixed', 
               column_config={
        'Injected mass (g)': st.column_config.NumberColumn(format='%f', required=True),
        'Initial activity syringe (MBq)' : st.column_config.NumberColumn(format='%f', required=True),
        'Residual activity (MBq)' : st.column_config.NumberColumn(format='%f', required=True),
        'Full Syringe Calibration datetime' : st.column_config.DatetimeColumn(required=True),
        'Empty Syringe Calibration datetime': st.column_config.DatetimeColumn(required=True),
        'Dilution datetime' : st.column_config.DatetimeColumn(required=True),                      
    },key='calibration_syringe')

input_table_std=pd.DataFrame(table1)
    
# Configuring the Calibration Standard Table

columns_gamma_cnt_cal[2].subheader('Calibration Standard')

columns_gamma_cnt_cal[2].caption('Input')

input_table_gc_calib =pd.DataFrame(columns=['Tare (g)',
                                        'Container + water (g)',
                                        'Container + water + Std. Syringe (g)'], index=range(1))

table2=columns_gamma_cnt_cal[2].data_editor(input_table_gc_calib, num_rows='fixed', 
                   column_config={
                       'Tare (g)': st.column_config.NumberColumn(format='%f'),
                       'Container + water (g)': st.column_config.NumberColumn(format='%f'),
                       'Container + water + Std. Syringe (g)': st.column_config.NumberColumn(format='%f')
                       },)
columns_gamma_cnt_cal[2].caption('Output')

total_diluted_std=table2[table2.columns[2]]-table2[table2.columns[0]]

columns_gamma_cnt_cal[2].write(pd.DataFrame(total_diluted_std, columns=['Total diluted standard (g)']))

total_diluted_std=np.array(total_diluted_std)[0]

# Configuring the Blanking gamma counter table


columns_gamma_cnt_cal[0].subheader('Gamma counter background')

input_table_blank_tubes =pd.DataFrame(columns=['Counts (CPM)'])

subcolumns_gamma_cnt_cal=columns_gamma_cnt_cal[0].columns(2)

subcolumns_gamma_cnt_cal[0].caption('Input')

table3= subcolumns_gamma_cnt_cal[0].data_editor(input_table_blank_tubes, num_rows='dynamic', column_config= 
                   {
                       'Counts (CPM)': st.column_config.NumberColumn(format='%f')
                   },key='blank_tubes')

st.session_state['gamma_counter_blank']=table3

average_bckg=table3.mean()

subcolumns_gamma_cnt_cal[1].caption('Average Background:')

subcolumns_gamma_cnt_cal[1].write(average_bckg)

columns_gamma_cnt_cal[1].subheader('Standard Syringe/Tube calibration')

columns_gamma_cnt_cal[1].caption('Insert the data from the standard samples')

input_table_calib_syr=pd.DataFrame(columns=['Sample Mass (net,g)','Counts (CPM)'])

subcolumns2=columns_gamma_cnt_cal[1].columns(2)

subcolumns2[0].caption('Input')

table4= subcolumns2[0].data_editor(input_table_calib_syr, num_rows='dynamic', column_config= 
                       {
                           'Sample Mass (net,g)': st.column_config.NumberColumn(format='%f'),
                           'Counts (CPM)': st.column_config.NumberColumn(format='%f')
                       }, key='standard_syringes')

bckg_corrected_norm_cpm_per_g= (table4[table4.columns[1]] - average_bckg[0]) / table4[table4.columns[0]]

subcolumns2[1].caption('Output')

subcolumns2[1].write((pd.DataFrame(bckg_corrected_norm_cpm_per_g,columns=['Nomalised CPM/g'])))



avg_bckg_corrected_norm_cpm_per_g=bckg_corrected_norm_cpm_per_g.mean()



total_bckg_corrected_norm_cpm_per_g=avg_bckg_corrected_norm_cpm_per_g*total_diluted_std



total_bckg_corrected_norm_cpm_per_g_injectate= total_bckg_corrected_norm_cpm_per_g / float(table1[table1.columns[0]][0])


st.divider()
st.subheader('Results')



results_subcolumns=st.columns(3)

results_subcolumns[0].caption('Gamma counter blanking and calibration results')

results_subcolumns[0].write(pd.DataFrame([avg_bckg_corrected_norm_cpm_per_g, total_bckg_corrected_norm_cpm_per_g,total_bckg_corrected_norm_cpm_per_g_injectate], index=['Average Normalised Bckg corrected : ', 'Total Normalised Bckg corrected (Standard tube): ','Total Normalised Bckg corrected CPM/g (Injectate)'], columns=['CPM/g']))

results_subcolumns[1].caption('Decay correction results')

try:
    
    corrected_syringe_activity_at_injection, corrected_syringe_activity_at_counting_start =decay_correct(time_exp, date_exp, table1['Full Syringe Calibration datetime'][0], table1['Empty Syringe Calibration datetime'][0], table1['Dilution datetime'][0], table1['Initial activity syringe (MBq)'][0], table1['Residual activity (MBq)'][0], dose_cal_bckg, dc )
    
    cpm_per_mbq_per_g=total_bckg_corrected_norm_cpm_per_g_injectate/(corrected_syringe_activity_at_counting_start/float(table1[table1.columns[0]][0]))
        
    results_subcolumns[1].write(pd.DataFrame([corrected_syringe_activity_at_injection, corrected_syringe_activity_at_counting_start, cpm_per_mbq_per_g ],
                                            index=['Corrected syr, activity (injection)', 'Corrected syr. activity (counting start)', 'CPM / MBq per g'], columns=['Value']))
    
except:
    results_subcolumns[1].warning('You are missing some information for the decay corrections, please check the sections above', icon='⚠️')
    

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
                       })

table5=pd.DataFrame(table5)


st.header('Organs gamma counting', divider='blue')

org_list=st.multiselect("Select the organs used in the experiments", options=organs_for_widget, key='organs_list')



if not st.session_state.organs_list: 
        st.warning(
            'No organs are selected, please check your entries above', icon='⚠️')
else:
        
    cols1 = st.columns(2)
    
    cols1[0].caption('Insert the net mass of the organs (g)')
    
    input_table_organs_cpms=pd.DataFrame(columns=['Mouse ' + str(x) for x in range(len(table5))], 
                                         index=st.session_state.organs_list)
    
    input_table_organs_mass=pd.DataFrame(columns=['Mouse ' + str(x) for x in range(len(table5))], 
                                         index=st.session_state.organs_list)
    
    dic_col_org=defaultdict(list)
    
    for organ in st.session_state.organs_list:
        
        dic_col_org[organ]=st.column_config.NumberColumn(format='%f')

    table7=cols1[0].data_editor(input_table_organs_cpms, num_rows='fixed', column_config=dic_col_org,key='organs cpms')
        
    cols1[1].caption('Insert the CPM of each organ')
    
    table6= cols1[1].data_editor(input_table_organs_cpms, num_rows='fixed', column_config=dic_col_org)
            
    table6_df=pd.DataFrame(table6)
    
    vis = pd.DataFrame(columns=['organ','mouse','value'])
    
    st.header('Plotting the inserted values', divider='blue')
    
    try:
        
        table7_float=table7.astype(float)
        
        table6_float=table6.astype(float)
        
        table8=table6_float.divide(table7_float)
        
        table8_alt_formatted=table8.T.reset_index().melt(id_vars='index')
        
        table8_alt_formatted.columns=['mouse', 'organ', 'CPM / g']
        
        table_9_alt_formatted=table8.subtract(average_bckg[0]).divide(cpm_per_mbq_per_g).T.reset_index().melt(id_vars='index')
        
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
         
        st.divider()
        
        plot1_results=st.columns(2)
        
        
        st.altair_chart(alt.layer(chart1, chart2, data=table14_alt).configure_axis(labelColor='black', labelFont='Arial', titleColor='Black', titleFont='Arial', labelFontWeight='normal', labelFontSize=p1_lab_fontsize, titleFontWeight='normal', titleFontSize=p1_axtitle_fontsize).properties(width=p1_width,height=p1_heigth)
        , theme=None)
        
        st.caption('Plot values and descriptive statistics')
        
        st.write(table14.describe())
        
        st.divider()
        
        st.markdown('Customize your Mean % ID / g plot')
        
        plot2_customisation=st.columns(3)
        
        color9=plot2_customisation[0].color_picker(label='Plot color', value='#99007d')
        
        opacity9=plot2_customisation[0].number_input(label='Plot opacity', min_value=0.1, max_value=1., step=0.01, value=0.5)
        
        chart3=alt.Chart(table13_alt).mark_bar(color=color9, opacity=opacity9).encode(alt.X('organ', sort=None), alt.Y('mean(value)').title('Mean %ID / g'))
        
        chart4=alt.Chart(table13_alt).mark_errorbar(extent='stdev', ticks=True, color='black', size=8).encode(alt.X('organ', sort=None).title('Organ'), alt.Y('mean(value)').title(''))
        
        
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
        
        st.write(table13.T.describe())
        
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
        
        
        st.markdown('Contrast ratio based on the % ID /g')
        
        st.write(table_contrast.T.describe())
        
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
        
        st.write(table_contrast.T.describe())
        
        st.divider()
        ####
        
        st.subheader('Individual data plots')
        
        chart5=alt.Chart(table8_alt_formatted).mark_bar().encode(alt.Column('organ'), alt.X('mouse').title(''), alt.Y('CPM / g', axis=alt.Axis(tickCount=5, format=".1e")).title('Norm. CPM/g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        chart6=alt.Chart(table8_alt_formatted).mark_bar().encode(alt.Column('mouse'), alt.X('organ').title(''), alt.Y('CPM / g', axis=alt.Axis(tickCount=5, format=".1e")).title('Norm. CPM/g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        st.altair_chart(chart5,theme=None)
        
        st.altair_chart(chart6,theme=None)
        
        st.markdown('CPM / g table')
        
        st.write(table8.T)
        
        chart7=alt.Chart(table_9_alt_formatted).mark_bar().encode(alt.Column('organ'), alt.X('mouse').title(''), alt.Y('MBq / g', axis=alt.Axis(tickCount=5)).title('MBq / g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        chart8=alt.Chart(table10).mark_bar().encode(alt.Column('mouse'), alt.X('organ', sort=None).title(''), alt.Y('MBq / g', axis=alt.Axis(tickCount=5)).title('MBq / g'), alt.Color('mouse').scale(scheme='tableau20'))
        
        st.altair_chart(chart7,theme=None)
        
        st.altair_chart(chart8,theme=None)
        
        st.markdown('Activity / g table')
        
        st.write(table11.T)
        
        chart9=alt.Chart(table13_alt).mark_bar().encode(alt.Column('organ'), alt.X('mouse', sort=None), alt.Y('value').title('ID / g (%)'), alt.Color('mouse').scale(scheme='tableau20'))
    
        chart10=alt.Chart(table13_alt).mark_bar().encode(alt.Column('mouse'), alt.X('organ', sort=None), alt.Y('value').title('ID / g (%)'), alt.Color('mouse').scale(scheme='tableau20'))
        
        st.altair_chart(chart9,theme=None)
        
        st.altair_chart(chart10,theme=None)
        
        st.markdown('% Injected dose per gram')
        
        st.write(table13.T)
        
        chart11=alt.Chart(table14_alt).mark_bar().encode(alt.Column('organ'), alt.X('mouse', sort=None).title(''), alt.Y('value').title('SUV (g)'), alt.Color('mouse').scale(scheme='tableau20'))
        
        chart12=alt.Chart(table14_alt).mark_bar().encode(alt.Column('mouse'), alt.X('organ', sort=None).title(''), alt.Y('value').title('SUV (g)'), alt.Color('mouse').scale(scheme='tableau20'))
        
        st.altair_chart(chart11,theme=None)
        
        st.altair_chart(chart12,theme=None)
        
        st.markdown('SUV table')
        
        st.write(table14)
        
        chart13=alt.Chart(table_contrast1).mark_bar().encode(alt.Column('organ'), alt.X('mouse').title(''), alt.Y('value', axis=alt.Axis(tickCount=5)).title('Contrast ratio'), alt.Color('mouse').scale(scheme='tableau20'))
        
        chart14=alt.Chart(table_contrast1).mark_bar().encode(alt.Column('mouse'), alt.X('organ').title(''), alt.Y('value', axis=alt.Axis(tickCount=5)).title('Contrast ratio'), alt.Color('mouse').scale(scheme='tableau20'))
        
        st.altair_chart(chart13, theme=None)
        
        st.altair_chart(chart14, theme=None)
        
        st.markdown('Contrast ratio based on % ID / g')
        
        st.write(table_contrast.T)
        
    except:
        st.warning('Missing data, plotting will not begin until all data will be provided')
            
