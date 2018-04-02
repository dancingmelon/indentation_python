# -*- coding: utf-8 -*-

# <editor-fold desc="to be commented out for production code">
import sys

sys.path.append(r'C:\ProgramData\Anton Paar\Common files\scripts\src')

import pprint

pp = pprint.PrettyPrinter(indent=4)  # use pp.pprint(stuff) for pretty printing embedded list and dict
# </editor-fold>

import init_script

from jsonrpctcp import connect

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.chart import (
    ScatterChart,
    Reference,
    Series,
)
from script_tools import indentation_port


"""
  Indentation client sample code

  demonstrates :

  - ls
  - docs
  - docs.open
  - groups
  - parameters
  - parameters.get_value
  - curves
  - curves.get_data
  - curves.breakpoints
"""

if __name__ == '__main__':
    indent = connect('127.0.0.1', indentation_port)

    result = indent.ls()
    print 'Connected to %(server_name)s, V%(server_version)s' % result

    result = indent.docs.open(path='C:\\ProgramData\\Anton Paar\\Indentation\\Sample\\Demo.MIT')
    doc_id=result['doc_id']

    docs = indent.docs()
    print docs
    doc_name= docs['docs'][doc_id]['name']
    print 'Opened', doc_name

    groups = indent.groups(doc_id=doc_id)
    parameters = indent.parameters(doc_id=doc_id)

    # iterate on document structure to display Hit value
    print groups
    for g_id in groups['indexes']:
        g = groups['groups'][g_id]
        print 'G:', g['name']
        for d_id in g['indexes']:
            d = g['data'][d_id]
            value = indent.parameters.getvalue(doc_id=doc_id,
                                               data_id=d_id,
                                               param_id='13002' )
            print '  M', d['name'], ':',parameters['13002']['name'], '=',value['value'], parameters['13002']['unit']

    # get curves information, retreive a curve dataset and plot it with matplotlib
    curves = indent.curves(doc_id=doc_id)
    print curves

    curve_data = indent.curves.getdata(doc_id=doc_id,
                                       data_id = 3,
                                       page_index=0,
                                       page_size=800,
                                       curve_type='ctStatic')

    breakpoints =  indent.curves.breakpoints(doc_id=doc_id,
                                       data_id = 3)
    print 'breakpoints', breakpoints

    dim_count = curve_data['dim_count']

    def header_from_curve(curve_dict, index):
        key = str(index)
        if key in curve_dict:
            return  curve_dict[key]['DisplayName']
        else:
            return ''

    header = [header_from_curve(curves['ctStatic'], i) for i in range(dim_count)]

    #create Excel file
    wb = Workbook()
    ws = wb.active

    ws.append(header)
    for row in curve_data['data']:
        ws.append(row)

    chart = ScatterChart()
    chart.title = "Indentation Curve"
    chart.style = 13
    chart.x_axis.title = 'Pd'
    chart.y_axis.title = 'Fn'

    xvalues = Reference(ws, min_col=3, min_row=2, max_row=curve_data['count'])
    values = Reference(ws, min_col=2, min_row=2, max_row=curve_data['count'])
    series = Series(values, xvalues, title_from_data=False)
    chart.series.append(series)

    ws.add_chart(chart, "O2")

    wb.save("indentation_curve.xlsx")

    #plot curve
    c = np.transpose(np.array(curve_data['data']))

    plt.figure(1)
    plt.subplot(311)
    plt.plot(c[0], c[1], 'b')

    plt.subplot(312)
    plt.plot(c[0], c[2], 'r')

    plt.subplot(313)
    plt.plot(c[2],c[1], 'g')

    plt.show()

