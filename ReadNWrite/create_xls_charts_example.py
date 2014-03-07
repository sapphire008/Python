""" 
Use XlsxWriter to create charts and plot in xlsx files
"""
import sys
sys.path.append('/hsgs/projects/jhyoon1/pkg64/pythonpackages/')
sys.path.append('/hsgs/projects/jhyoon1/pkg64/pythonpackages/XlsxWriter-0.5.1')
# Import xlswriter library
import xlrd
import xlsxwriter
# Open the Excel workbook to import data
workbook = xlrd.open_workbook('/hsgs/projects/jhyoon1/midbrain_pilots/haldol/MID_ROI_betas.xlsx')
# Open another Excel workbook to write the data and create charts
chart_workbook = xlsxwriter.Workbook('/hsgs/projects/jhyoon1/midbrain_pilots/haldol/MID_ROI_betas_2.xlsx');
# Get a list of worksheets present in the workbook
worksheet_list = workbook.sheet_names();
# Convert all contents to strings
worksheet_list = [str(i) for i in worksheet_list];
for w in range(0,len(worksheet_list)):
    # Get the worksheet
    worksheet = None;# clear the variable beforehand
    worksheet = workbook.sheet_by_name(worksheet_list[w]);
    # Grab the cell contents to be plotted
    # Column headers
    col_headers = worksheet.row_values(29);
    col_headers = col_headers[2:];
    col_headers = [str(i) for i in col_headers];
    
    # Original subject by subject data
    Original_Data = None;
    Original_Data = [[]]*worksheet.nrows;
    for r in range (0, worksheet.nrows):
        tmp = None;
        tmp = worksheet.row_values(r);
        # convert all unicode to string and keep all float
        Original_Data[r] = [str(i) if isinstance(i,basestring) else i for i in tmp]
            

    # With data imported and saved, create charts
    chart_worksheet = None;
    chart_worksheet = chart_workbook.add_worksheet(worksheet_list[w])
    # Write the imported data to the new worksheet
    for r in range(0,worksheet.nrows):
        chart_worksheet.write_row('A'+str(r+1),Original_Data[r]);
    
    # Create a new Chart object
    chart = None;
    chart = chart_workbook.add_chart({'type':'column'});
    # Configure the first chart
    chart.add_series({'values':worksheet_list[w]+'!$C$31:$H$31',
                      'categories':worksheet_list[w]+'!$C$30:$H$30',
                      'name':'C',
                      'y_error_bars':{
                      'type':'fixed',
                      'value':worksheet_list[w]+'!$C$41:$H$41',
                      'end_style':1,
                      'direction':'both'}});#Cue_C
    chart.add_series({'values':worksheet_list[w]+'!$C$32:$H$32',
                      'categories':worksheet_list[w]+'!$C$30:$H$30',
                      'name':'SZ',
                      'y_error_bars':{
                      'type':'fixed',
                      'value':worksheet_list[w]+'$!C$42:$H$42',
                      'end_style':1,
                      'direction':'both'}});#Cue_C});#Cue_SZ
    chart.set_title({'name':worksheet_list[w]+' Cue Betas by Groups by Conditions'});
    chart.set_legend({'position':'right'});
    chart.set_size({'width':720,'height':576});
    chart.set_x_axis({'name':'Conditions'});
    chart.set_y_axis({'name':'Beta Values',
                      'major_gridlines':{'visible':False}});
    # Insert the chart
    chart_worksheet.insert_chart('B48',chart);

    # Configure the second chart
    chart = None;
    chart = chart_workbook.add_chart({'type':'column'});
    chart.add_series({'values':worksheet_list[w]+'!$I$31:$T$31',
                      'categories':worksheet_list[w]+'!$I$30:$T$30',
                      'name':'C',
                      'y_error_bars':{
                      'type':'fixed',
                      'value':worksheet_list[w]+'$!I$41:$T$41',
                      'end_style':1,
                      'direction':'both'}});#Feedback_C
    chart.add_series({'values':worksheet_list[w]+'!$I$32:$T$32',
                      'categories':worksheet_list[w]+'!$I$30:$T$30',
                      'name':'SZ',
                      'y_error_bars':{
                      'type':'fixed',
                      'value':worksheet_list[w]+'$!I$42:$T$42',
                      'end_style':1,
                      'direction':'both'}});#Feedback_SZ
    # Insert the second chart
    chart.set_title({'name':worksheet_list[w]+' Feedback Betas by Groups by Conditions'});
    chart.set_legend({'position':'right'});
    chart.set_size({'width':1500,'height':576});
    chart.set_x_axis({'name':'Conditions'});
    chart.set_y_axis({'name':'Beta Values',
                      'major_gridlines':{'visible':False}});
    # Insert the chart
    chart_worksheet.insert_chart('B80',chart);

# At the end, close the workbook
chart_workbook.close();
