# PROYECTO: SISTEMA PARA EL SEGUIMIENTO DE ECOSISTEMAS VENEZOLANOS
# AUTOR: Javier Martinez

import pandas as pd
import numpy as np

#----------------
# anomalia_periodo
#----------------

def anomalia_periodo(year = 2020):

    """
    Funcion para calcular las anomalias sst
    """

    value = list(filter(
                        lambda x: x[1] <= year and x[2] >= year,
                        [
                        [1,1950,1955,1936,1965],
                        [2,1956,1960,1941,1970],
                        [3,1961,1965,1946,1975],
                        [4,1966,1970,1951,1980],
                        [5,1971,1975,1956,1985],
                        [6,1976,1980,1961,1990],
                        [7,1981,1985,1966,1995],
                        [8,1986,1990,1971,2000],
                        [9,1991,1995,1976,2005],
                        [10,1996,2000,1981,2010],
                        [11,2001,2005,1986,2015],
                        [12,2006,2010,1991,2020],
                        [13,2011,2015,1996,2025],
                        [14,2016,2020,2001,2030],
                        [15,2021,2025,2006,2035],
                        [16,2026,2030,2011,2040],
                        [17,2031,2035,2016,2045],
                        [18,2036,2040,2021,2050],
                        [19,2041,2045,2026,2055],
                        [20,2046,2050,2031,2060],
                        [21,2051,2055,2036,2065],
                        ]
                        ))
    if value==[]:
        return 0
    else:
        return value[0][0]

#----------------
# periodo_anomalias_climaticas
#----------------

def periodo_anomalias_climaticas(data_pandas):

    """
    Funcion para determinar anomalias segun periodo
    """

    data_pandas['anomalia_periodo'] = data_pandas['year'].apply(anomalia_periodo)

    # determinando climatologica
    pd_climatologica = data_pandas.groupby(['month','anomalia_periodo'],as_index=False).agg({'climatologica':'mean'})

    # periodo de anomalias
    pd_periodo_anomalias = pd.DataFrame(
                                        [
                                    [1,1950,1955,1936,1965],
                                    [2,1956,1960,1941,1970],
                                    [3,1961,1965,1946,1975],
                                    [4,1966,1970,1951,1980],
                                    [5,1971,1975,1956,1985],
                                    [6,1976,1980,1961,1990],
                                    [7,1981,1985,1966,1995],
                                    [8,1986,1990,1971,2000],
                                    [9,1991,1995,1976,2005],
                                    [10,1996,2000,1981,2010],
                                    [11,2001,2005,1986,2015],
                                    [12,2006,2010,1991,2020],
                                    [13,2011,2015,1996,2025],
                                    [14,2016,2020,2001,2030],
                                    [15,2021,2025,2006,2035],
                                    [16,2026,2030,2011,2040],
                                    [17,2031,2035,2016,2045],
                                    [18,2036,2040,2021,2050],
                                    [19,2041,2045,2026,2055],
                                    [20,2046,2050,2031,2060],
                                    [21,2051,2055,2036,2065],
                                    ],
                            columns=['anomalia_periodo','date_init','date_end','range_init','range_end']

                        )


    pd_climatologia_final = pd.merge(pd_periodo_anomalias, pd_climatologica,on=['anomalia_periodo'],how='inner')
    return pd_climatologia_final[['date_init','date_end','anomalia_periodo','month','climatologica','range_init','range_end']]

#----------------
# forecast_oni_grap
#----------------

def forecast_oni_grap(pd_oni):

    """
    Funcion para el grafico del pronostico del ONI
    """

    import plotly.graph_objects as go
    from plotly.graph_objects import Layout

    data_fig = pd_oni[ pd_oni.index < pd_oni.index.max() ].copy()
    data_fig['color'] = data_fig['oni'].apply(lambda x: 0 if x<0 else 1)


    max_date = data_fig.index.max() + pd.DateOffset(months=5) 

    fig = go.Figure(layout=Layout(plot_bgcolor='rgba(0,0,0,0)'))
    fig.add_trace(go.Scatter(x=data_fig.index.tolist(), y=len(data_fig.index.tolist())*[0],
                            mode='lines',name='NIÑO3.4 NARX entrenamiento',
                            line=dict(color='#B0ACAC', width=2),
                            fill = 'tozeroy',
                            fillcolor = '#F5FF8D',
                            showlegend=False))#,fill='tozeroy'))

    fig.add_trace(go.Scatter(x=data_fig.index, 
                            y=data_fig.oni,
                            mode='lines+markers',
                            marker_symbol='x-thin',
                            marker_line_width=2,
                            marker_size=3,
                            marker_line_color='#003CAF',
                            marker_color='#003CAF',
                            name='NIÑO3.4 NARX entrenamiento',
                            line=dict(color='#0057FF', width=3),
                            fill = 'tonexty',
                            fillcolor = '#8FB5FE',
                            showlegend=False
                            ))

    # pronostico
    fig.add_trace(go.Scatter(x=data_fig[data_fig.type=='prediction'].index, 
                            y=data_fig[data_fig.type=='prediction'].oni,
                            text=data_fig[data_fig.type=='prediction'].oni.apply(lambda x: str(round(x,2)) ),
                            textposition="bottom right",
                            marker_symbol='star',
                            marker_line_width=3,
                            marker_size=3,
                            marker_line_color='#0057FF',
                            marker_color='#0057FF',
                            mode='lines+markers+text',
                            name='NIÑO3.4 NARX entrenamiento',
                            line=dict(color='#EF02F3', width=3),
                            showlegend=False
                            ))

    fig.add_annotation(x=max_date, y=0.5+0.2,text="débil",showarrow=False,yshift=10,font=dict(color='#FF6C6C') )
    fig.add_hline(y=0.5, line_width=0.75, line_dash="dash", line_color="#FF6C6C")

    fig.add_annotation(x=max_date, y=1+0.2,text="moderado",showarrow=False,yshift=10,font=dict(color='#FF3F3F') )
    fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="#FF3F3F")

    fig.add_annotation(x=max_date, y=1.5+0.2,text="fuerte",showarrow=False,yshift=10,font=dict(color='#FF0000') )
    fig.add_hline(y=1.5, line_width=1.25, line_dash="dash", line_color="#FF0000")

    fig.add_annotation(x=max_date, y=2+0.2,text="muy fuerte",showarrow=False,yshift=10,font=dict(color='#D70000') )
    fig.add_hline(y=2, line_width=1.50, line_dash="dash", line_color="#D70000")


    fig.add_hline(y=2.5, line_width=1.75, line_dash="dash", line_color="#AD0000")


    fig.add_annotation(x=max_date, y=-0.5-0.35,text="débil",showarrow=False,yshift=10,font=dict(color='#69A6FF') )
    fig.add_hline(y=-0.5, line_width=0.75, line_dash="dash", line_color="#69A6FF")

    fig.add_annotation(x=max_date, y=-1-0.35,text="moderado",showarrow=False,yshift=10,font=dict(color='#6979FF') )
    fig.add_hline(y=-1, line_width=1, line_dash="dash", line_color="#6979FF")

    fig.add_annotation(x=max_date, y=-1.5-0.35,text="fuerte",showarrow=False,yshift=10,font=dict(color='#3F53FF') )
    fig.add_hline(y=-1.5, line_width=1.25, line_dash="dash", line_color="#3F53FF")

    fig.add_annotation(x=max_date, y=-2-0.35,text="muy fuerte",showarrow=False,yshift=10,font=dict(color='#001BFF') )
    fig.add_hline(y=-2, line_width=1.5, line_dash="dash", line_color="#001BFF")

    fig.add_hline(y=-2.5, line_width=1.75, line_dash="dash", line_color="#00059A")

    # el nino y la nina
    fig.add_annotation(x=data_fig.index.max() - pd.DateOffset(months=4*12), y=2+0.2,text="El Niño",showarrow=False,yshift=15,font=dict(color='#D70000') )
    fig.add_annotation(x=data_fig.index.max() - pd.DateOffset(months=4*12), y=-2-0.35,text="La Niña",showarrow=False,yshift=15,font=dict(color='#001BFF') )

    # linea de pronostico
    fig.add_vline(x=data_fig[data_fig.type=='prediction'].index.min(), line_width=3, line_dash="dash", line_color="#580606")

    fig.update_xaxes(tickformat="%Y/%m",showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)

    fig.update_traces(textfont_size=14)
    fig.update_layout(title="""
                            Índice Niño Oceánico (ONI) pronóstico periodo {date_init} al {date_fin}
                            <br><sup>Promedio de 3-meses para las anomalías SST en la región Niño 3.4 (variación periodos base de 30-años)
                            </sup>
                            """.format(date_init=str(data_fig[data_fig.type=='prediction'].index.min().strftime('%Y/%m')),
                                    date_fin=str(data_fig[data_fig.type=='prediction'].index.max().strftime('%Y/%m')) ),
                    xaxis_title='Mes',
                    yaxis_title='Promedio 3-Meses anomalías SST (°C)',
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    height=800,
                    width=1500,
                    font = dict(size = 22),
                    xaxis_range=[data_fig.index.max() - pd.DateOffset(months=5*12), max_date + pd.DateOffset(months=5) ]
                    )

    return fig


#----------------
# temperatura_nino34_plot
#----------------

def temperatura_nino34_plot(data_model):

    """
    Funcion para la grafica de SST
    """
    
    import plotly.graph_objects as go
    from plotly.graph_objects import Layout

    fig = go.Figure(layout=Layout(plot_bgcolor='rgba(0,0,0,0)'))
    fig.add_trace(go.Scatter(x=data_model.index, y=data_model['nino34_mean'],
                             mode='lines',name='NIÑO 3.4',
                             line=dict(color='#C10101', width=2),
                             xperiodalignment="middle"
                            ))

    fig.update_xaxes(tickformat="%Y/%m",
                     showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                     ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10
                    )
    
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                     ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)

    fig.update_traces(textfont_size=18)
    fig.update_layout(title='Temperatura promedio en la región NIÑO 3.4',
                      xaxis_title='Mes',
                      yaxis_title='Temperatura (°C)',
                      legend_title_text='Serie',
                      legend_title = dict( font = dict(size = 30)),
                      legend=dict(y=0.5,
                                  traceorder='reversed',
                                  font_size=11),

                       uniformtext_minsize=8,
                       uniformtext_mode='hide',
                       height=800,
                       width=1500,
                       font = dict(size = 22),
                       xaxis_range=[data_model.index.min() - pd.DateOffset(months=1*12), data_model.index.max() + pd.DateOffset(months=1*12)]
                     )

    return fig
    
#----------------
# seasonal_decompose_plot
#----------------

def seasonal_decompose_plot(data, model='multiplicable', period=12):

    """
    Modelos multiplicativo y aditivo SST
    """

    from statsmodels.tsa.seasonal import seasonal_decompose
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # descomposicion
    result = seasonal_decompose(data,
                                model=model,
                                period=period)

    # tipo de modelo
    if model=='multiplicable':
        modelo = 'Multiplicativo'
    elif model=='additive':
        modelo = 'Aditivo'
    else :
        modelo = ''

    # Creando figura
    fig = make_subplots(rows=4,cols=1,
                        column_titles=['SST promedio','Tendencia','Componente Estacional','Residuos']
                       )

    # Observado
    fig.append_trace(go.Scatter(x=result.observed.index, y=result.observed,
                                mode='lines',name='Tempreatura promedio observada NIÑO 3.4',
                                line=dict(color='#C10101', width=2)),
                     row=1, col=1)
    # Tendencia
    fig.append_trace(go.Scatter(x=result.trend.index, y=result.trend,
                                mode='lines',name='NIÑO 3.4',
                                line=dict(color='#FF0303', width=2)),
                     row=2, col=1)
    # Estacional
    fig.append_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal,
                                mode='lines',name='NIÑO 3.4',
                                line=dict(color='#FF4242', width=2)),
                     row=3, col=1)
    # Residuales
    fig.append_trace(go.Scatter(x=result.resid.index, y=result.resid,
                                mode='lines',name='NIÑO 3.4',
                                line=dict(color='#FE5C5C', width=2)),
                     row=4, col=1)

    # Ejes
    fig['layout']['xaxis']['title']=''
    fig['layout']['xaxis2']['title']=''
    fig['layout']['xaxis3']['title']=''
    fig['layout']['xaxis4']['title']='Meses'


    fig['layout']['yaxis']['title']='Temperatura (°C)'
    fig['layout']['yaxis2']['title']='Tendencia'
    fig['layout']['yaxis3']['title']='Estacional'
    fig['layout']['yaxis4']['title']='Residuos'

    # Formato
    fig.update_xaxes(tickformat="%Y/%m",showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                     ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                     ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)
    fig.update_layout(height=800,
                      width=1500,
                      template='plotly_white',
                      showlegend = False,
                      title_text="""
                                 Descomposición de la serie SST promedio región Niño 3.4
                                 <br><sup>Modelo {modelo}</sup>
                                 """.format(modelo=modelo),
                      xaxis_range=[data.index.min() - pd.DateOffset(months=1*12), data.index.max() + pd.DateOffset(months=1*12)]
                     )
    return fig

#----------------
# oni_periodos_base
#----------------

def oni_periodos_base(pd_normalidad,pd_temperatura):
        
        """
        Funcion para grafico de ONI periodos base
        """

        import plotly.graph_objects as go
        from plotly.graph_objects import Layout

        fig = go.Figure(layout=Layout(plot_bgcolor='rgba(0,0,0,0)'))
        periodos = pd_normalidad.base.unique().tolist()
        color = [
                '#FB5CFF','#F700FF','#910095',
                '#7E8EFA','#5B67BD', '#0022FF',
                '#6ADDFF','#00C5FF', '#0092BD',
                #'#6FFE76','#00FF0C','#00A808',
                '#F8FF20','#DFF100','#AEBB02',
                '#FFC162','#FFB900','#DA9E00']

        for base in periodos:
        
                data_base = pd_normalidad\
                                .query("base=='{base}'".format(base=base))\
                                .sort_values(['date_init','month'],ascending=[True,True])\
                                .copy()
                data_temperatura = pd_temperatura\
                                        .query("base=='{base}'".format(base=base))\
                                        .sort_values(['date_init','month'],ascending=[True,True])\
                                        .copy()
                # colores
                col = color[periodos.index(base)]
                
                fig.add_trace(go.Scatter(x=data_base.mes, y=data_base['climatologica'],
                                        mode='lines+markers',
                                        marker_symbol='x-thin',
                                        marker_line_width=2,
                                        marker_size=5,
                                        marker_line_color=col,
                                        marker_color=col,
                                        name=base,
                                        line=dict(color=col, width=2), #0.5*(periodos.index(base)+1)/2),
                                        showlegend=True))
                fig.add_trace(go.Box(x=data_temperatura.mes, y=data_temperatura['nino34_mean'],
                                        opacity=0.1,
                                        name=base,line=dict(color=col),
                                        showlegend=False
                                ))



        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                        ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                        ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)



        #fig.update_traces(textfont_size=18)
        fig.update_layout(title="""
                                SST promedio en la región Niño 3.4
                                <br><sup>Periodos base de 30 años</sup>
                                """,
                        xaxis_title='Mes',
                        yaxis_title='SST promedio (°C)',
                        legend_title_text='Periodo base',
                        legend_title = dict( font = dict(size = 20)),
                        legend=dict(y=0.5,
                                traceorder='reversed',
                                font_size=11),

                        uniformtext_minsize=8,
                        uniformtext_mode='hide',
                        height=800,
                        width=1500,
                        font = dict(size = 20)
                        )
        return fig



#----------------
# grafico_oni
#----------------

def grafico_oni(data_fig):

    """
    Funcion para la graficacion del ONI
    """

    import plotly.graph_objects as go
    from plotly.graph_objects import Layout

    data_fig['color'] = data_fig['oni'].apply(lambda x: 0 if x<0 else 1)


    fig = go.Figure(layout=Layout(plot_bgcolor='rgba(0,0,0,0)'))
    fig.add_trace(go.Scatter(x=data_fig.index.tolist(), y=len(data_fig.index.tolist())*[0],
                            mode='lines',name='NIÑO3.4 NARX entrenamiento',
                            line=dict(color='#B0ACAC', width=2),
                            fill = 'tozeroy',
                            fillcolor = '#F5FF8D',
                            showlegend=False))#,fill='tozeroy'))

    fig.add_trace(go.Scatter(x=data_fig.index, 
                            y=data_fig.oni,
                            mode='lines+markers',
                            marker_symbol='x-thin',
                            marker_line_width=2,
                            marker_size=3,
                            marker_line_color='#003CAF',
                            marker_color='#003CAF',
                            name='NIÑO3.4 NARX entrenamiento',
                            line=dict(color='#0057FF', width=3),
                            fill = 'tonexty',
                            fillcolor = '#8FB5FE',
                            showlegend=False
                            ))

    max_date = data_fig.index.max() + pd.DateOffset(months=5*12)

    fig.add_annotation(x=max_date, y=0.5+0.2,text="débil",showarrow=False,yshift=10,font=dict(color='#FF6C6C') )
    fig.add_hline(y=0.5, line_width=0.75, line_dash="dash", line_color="#FF6C6C")

    fig.add_annotation(x=max_date, y=1+0.2,text="moderado",showarrow=False,yshift=10,font=dict(color='#FF3F3F') )
    fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="#FF3F3F")

    fig.add_annotation(x=max_date, y=1.5+0.2,text="fuerte",showarrow=False,yshift=10,font=dict(color='#FF0000') )
    fig.add_hline(y=1.5, line_width=1.25, line_dash="dash", line_color="#FF0000")

    fig.add_annotation(x=max_date, y=2+0.2,text="muy fuerte",showarrow=False,yshift=10,font=dict(color='#D70000') )
    fig.add_hline(y=2, line_width=1.50, line_dash="dash", line_color="#D70000")


    fig.add_hline(y=2.5, line_width=1.75, line_dash="dash", line_color="#AD0000")


    fig.add_annotation(x=max_date, y=-0.5-0.35,text="débil",showarrow=False,yshift=10,font=dict(color='#69A6FF') )
    fig.add_hline(y=-0.5, line_width=0.75, line_dash="dash", line_color="#69A6FF")

    fig.add_annotation(x=max_date, y=-1-0.35,text="moderado",showarrow=False,yshift=10,font=dict(color='#6979FF') )
    fig.add_hline(y=-1, line_width=1, line_dash="dash", line_color="#6979FF")

    fig.add_annotation(x=max_date, y=-1.5-0.35,text="fuerte",showarrow=False,yshift=10,font=dict(color='#3F53FF') )
    fig.add_hline(y=-1.5, line_width=1.25, line_dash="dash", line_color="#3F53FF")

    fig.add_annotation(x=max_date, y=-2-0.35,text="muy fuerte",showarrow=False,yshift=10,font=dict(color='#001BFF') )
    fig.add_hline(y=-2, line_width=1.5, line_dash="dash", line_color="#001BFF")

    fig.add_hline(y=-2.5, line_width=1.75, line_dash="dash", line_color="#00059A")


    fig.add_annotation(x=data_fig.index.min() + pd.DateOffset(months=10*12), y=2+0.2,text="El Niño",showarrow=False,yshift=15,font=dict(color='#D70000') )
    fig.add_annotation(x=data_fig.index.min() + pd.DateOffset(months=10*12), y=-2-0.35,text="La Niña",showarrow=False,yshift=15,font=dict(color='#001BFF') )


    fig.update_xaxes(tickformat="%Y/%m",showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)

    fig.update_layout(title="""
                            Índice Niño Oceánico (ONI) periodo {date_init} al {date_fin}
                            <br><sup>Promedio de 3-meses para las anomalías SST en la región Niño 3.4 (variación periodos base de 30-años)
                            </sup>
                            """.format(date_init=str(data_fig.index.min().strftime('%Y/%m')),
                                    date_fin=str(data_fig.index.max().strftime('%Y/%m')) ),
                    xaxis_title='Mes',
                    yaxis_title='Promedio 3-Meses anomalías SST (°C)',
    #                   legend_title_text='Serie',
    #                   legend_title = dict( font = dict(size = 30)),
    #                   legend=dict(y=0.5,
    #                               #traceorder='reversed',
    #                               font_size=22),

                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    height=800,
                    width=1500,
                    font = dict(size = 22),
                    xaxis_range=[data_fig.index.min() - pd.DateOffset(months=1*12), data_fig.index.max() + pd.DateOffset(months=10*12)]
                    )

    return fig