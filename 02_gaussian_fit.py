#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
  Gaussian Fit
"""

# <editor-fold desc="to be commented out for production code">
import sys

sys.path.append(r'C:\ProgramData\Anton Paar\Common files\scripts\src')

import pprint

pp = pprint.PrettyPrinter(indent=4)  # use pp.pprint(stuff) for pretty printing embedded list and dict
# </editor-fold>

import init_script

import os


import webbrowser

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.figure_factory as ff

from jsonrpctcp import connect
from script_tools import indentation_port, get_parameters_id_from_name, get_current_doc_group, info

import numpy as np
from matplotlib.mlab import griddata
from sklearn import mixture
from scipy import linalg

          
if __name__ == '__main__':

    indent = connect('127.0.0.1', indentation_port)

    result = indent.ls()
    info('Connected to %(server_name)s, V%(server_version)s' % result)

    doc_id, doc_name, group_id, group_name = get_current_doc_group(indent)
    info('Opened: %s %s' % (doc_name, doc_id))
    info('Group: %s %s' % (group_name, group_id))

    # extract Hit, Eit values from the group
    parameters = indent.parameters(doc_id=doc_id)
    Eit_id = get_parameters_id_from_name(parameters, 'EIT (O&P)')
    Hit_id = get_parameters_id_from_name(parameters, 'HIT (O&P)')
    X_id = get_parameters_id_from_name(parameters, 'X')
    Y_id = get_parameters_id_from_name(parameters, 'Y')
    info('Parameters id ok')

    groups = indent.groups(doc_id=doc_id)
    Eit_list = np.array(indent.parameters.getvalues(doc_id=doc_id, 
                                                    group_id=group_id, 
                                                    param_id=Eit_id )['values'])
    Hit_list = np.array(indent.parameters.getvalues(doc_id=doc_id, 
                                                    group_id=group_id, 
                                                    param_id=Hit_id )['values'])
    X_list = np.array(indent.parameters.getvalues(doc_id=doc_id, 
                                                  group_id=group_id, 
                                                  param_id=X_id )['values'])
    Y_list = np.array(indent.parameters.getvalues(doc_id=doc_id, 
                                                  group_id=group_id, 
                                                  param_id=Y_id )['values'])

    # Filter out NaN values
    is_nan = np.isnan(Eit_list)
    Eit_list = Eit_list[~is_nan]
    Hit_list = Hit_list[~is_nan]
    X_list = X_list[~is_nan]
    Y_list = Y_list[~is_nan]
    info('Parameters values ok')
    
    # Convert 3 column data to matplotlib grid for Contour plots
    xi = np.linspace(min(X_list), max(X_list), 100)
    yi = np.linspace(min(Y_list), max(Y_list), 100)
    Z_Eit = griddata(X_list, Y_list, Eit_list, xi, yi,  interp='linear')
    Z_Hit = griddata(X_list, Y_list, Hit_list, xi, yi,  interp='linear')
    info('Griddata ok')

    # compute histograms
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    Eit_dist = np.histogram(Eit_list[~np.isnan(Eit_list)], bins='auto', density=True) 
    Hit_dist = np.histogram(Hit_list[~np.isnan(Hit_list)], bins='auto', density=True) 
    info('Histogram ok')
    
    # Create the interactive Graphs 
    # https://plot.ly/dash/
    app = dash.Dash()
    app.css.append_css({'external_url': 'https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css'})
    app.layout = html.Div(
        className='container',
        children=[
            html.Div([
                html.H3('Indentation Gaussian Fit', className='text-muted'),
                html.Div([html.P('Document: %s'%doc_name), html.P('Group: %s' % group_name)]),
                html.Hr()],
                className='header clearfix'
            ),


            html.Div(id='results-id',
                     className='row',
            ),

            html.Div(
                className='row',
                children=
                    html.Div(
                        className='col-6',
                        children=
                            html.Div(
                                className='card',
                                style={'padding-bottom':'2em', 'margin-bottom':'2em'},
                                children=
                                    html.Div(
                                        className='card-body',
                                        children=[
                                            html.Div(id='phasecount-disp-id',
                                                     className='card-title'
                                            ),
                                            dcc.Slider(
                                                id='phasecount-id',
                                                min=1,
                                                max=9,
                                                marks={i: str(i) for i in range(1, 10)},
                                                value=3,
                                            ),
                                        ]
                                    )
                            )
                    )
            ),


            html.H3('Eit distribution'),
            html.Hr(),
            html.Div(
                className='row',
                children= [
                    html.Div(
                        className='col-6',
                        children=
                            dcc.Graph(
                                id='hist-eit-graph',
                                figure={
                                    'data': [go.Bar(
                                        x= Eit_dist[1],
                                        y= Eit_dist[0],
                                    )],
                                    'layout': {
                                        'title': 'Eit distribution',
                                        'xaxis': dict(
                                            title='Eit GPa'
                                        ),
                                    }
                                }
                            )
                    ),

                    html.Div(
                        className='col-5',
                        children=
                        # https://plot.ly/python/contour-plots/
                        dcc.Graph(
                            id='map3-graph',
                            figure={
                                'data': [go.Contour(
                                    z= Z_Eit,
                                    x= xi,
                                    y= yi,
                                    colorscale='Viridis',
                                    autocontour=False,
                                    connectgaps= True,
                                )],
                                'layout': {
                                    'title': 'Eit = f(X,Y)',
                                }
                            }
                        )
                    )
                ]
            ),


            html.H3('Hit distribution'),
            html.Hr(),
            html.Div(
                className='row',
                children= [
                    html.Div(
                        className='col-6',
                        children=
                            dcc.Graph(
                                id='hist-hit-graph',
                                figure={
                                    'data': [go.Bar(
                                        x= Hit_dist[1],
                                        y= Hit_dist[0],
                                    )],
                                    'layout': {
                                        'title': 'Hit distribution',
                                        'xaxis': dict(
                                            title='Hit MPa'
                                        ),
                                    }
                                }
                            )
                    ),
                    html.Div(
                        className='col-5',
                        children=
                            dcc.Graph(
                                id='map4-graph',
                                figure={
                                    'data': [go.Contour(
                                        z= Z_Hit,
                                        x= xi,
                                        y= yi,
                                        colorscale='Jet',
                                        connectgaps= True,
                                    )],
                                    'layout': {
                                        'title': 'Hit = f(X,Y)',
                                    }
                                }
                            )
                    )
                ]
            ),


            html.H3('Eit visualization'),
            html.Hr(),
            html.Div(
                className='row',
                children= [
                    html.Div(
                        className='col-6',
                        children=
                            dcc.Graph(
                                id='map3d-graph',
                                figure={
                                    'data': [go.Surface(
                                        z= Z_Eit,
                                        x= xi,
                                        y= yi,
                                        colorscale='Viridis',
                                    )],
                                    'layout': {
                                        'title': 'Eit = f(X,Y)',
                                    }
                                }
                            )
                    ),
                    html.Div(
                        className='col-5',
                        children=
                            dcc.Graph(
                                id='map2-graph',
                                figure={
                                    'data': [go.Scatter(
                                        x= X_list, 
                                        y= Y_list, 
                                        name= 'Eit', 
                                        mode= 'markers',
                                        marker=dict(
                                            size='12',
                                            color = Eit_list,
                                            colorscale='Viridis',
                                            showscale=True
                                        ),
                                        opacity= 0.7
                                    )],
                                    'layout': {
                                        'title': 'Eit = f(X,Y)',
                                    }
                                }
                            )
                    )
                ]
            )
              
    ])

    def make_dash_table( array ):
        ''' Return a dash definitio of an HTML table for a Pandas dataframe '''
        table = []
        for index, row in enumerate(array):
            html_row = []
            for i, data in enumerate(row):
                html_row.append( html.Td([ data ]) )
            table.append( html.Tr( html_row ) )
        return table


    @app.callback(
        Output(component_id='phasecount-disp-id', component_property='children'),
        [Input(component_id='phasecount-id', component_property='value')]
    )
    def update_phasecount_div(phasecount_value):
        return 'Phase count: {}'.format(phasecount_value)


    @app.callback(
        Output(component_id='results-id', component_property='children'),
        [Input(component_id='phasecount-id', component_property='value'),
         Input(component_id='phasecount-disp-id', component_property='children')]
    )
    def update_results_div(phasecount_value, dummy):
        # Compute Gaussian Fit
        # http://scikit-learn.org/stable/modules/mixture.html
        # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
        # http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        cv_types = ['spherical', 'tied', 'diag', 'full']
        n_components = phasecount_value
        X = np.transpose(np.array([Eit_list, Hit_list]))
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full')
        gmm.fit(X)
        Y= gmm.predict(X)
        #print "Covar",  gmm.covariances_
        for cov in gmm.covariances_ :
            v, w = linalg.eigh(cov)
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180. * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)            
            #print 'v', v, 'angle', angle

        #print "Params",  gmm.get_params()

        table = []
        header = ['' , 'Eit [GPa]', 'Hit [MPa]', '%']
        table = [header]
        for c in range(n_components):
            row = [ '#%d' % (c+1)]
            row.append('%.2f' % gmm.means_[c][0])
            row.append('%.2f' % gmm.means_[c][1])
            row.append('%.2f' % (gmm.weights_[c]*100))
            table.append(row)

        result_table = make_dash_table(table) 


        c= ['hsl('+str(h)+',90%'+',50%)' for h in np.linspace(0, 360, n_components+1)]

        data = []
        for i in range(n_components):
            data.append(
                go.Scatter(
                            x= X[Y==i, 0], 
                            y= X[Y==i, 1], 
                            name= '#%d' % (i+1), 
                            mode= 'markers',
                            marker=dict(
                                size='8',
                                color = c[i],
                            ),
                            opacity= 0.7
                        )
            )

        return [
            html.Div(
                children=[
                    dcc.Graph(
                        id='fit-graph',                
                        figure={
                            'data': data,
                            'layout': {
                                'title': 'Eit / Hit',
                            }
                        }
                    )],
                className='col-6'
            ),
            html.Div(
                children=[
                    html.H4('Results'),
                    html.Table( result_table, className = "table table-bordered" ),                    
                ],
                className='col-5'
            ),
        ]
    info('Dash created')
    webbrowser.open_new('http://127.0.0.1:8050/')
        
    app.run_server(debug=False, processes=0)
    info('Dash ok')    
