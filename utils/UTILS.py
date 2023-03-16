# PROYECTO: SISTEMA PARA EL SEGUIMIENTO DE ECOSISTEMAS VENEZOLANOS
# AUTOR: Javier Martinez

import numpy as np
import pandas as pd

# Objeto para transformacion
class LogMinimax:
    """
    Transformacion LogMinimax
    """

    @classmethod
    def create(cls, values):

        clase = cls()

        clase.values = values
        clase.log_values = np.log(clase.values + 1)
        clase.max = clase.log_values.max()
        clase.min = clase.log_values.min()

        return clase

    def transformacion(self):

        return (self.log_values - self.min)/(self.max - self.min)

    def inversa(self,y):

        return  np.exp( ( y*(self.max - self.min) ) + self.min ) - 1

# Funcion para metricas
def metrics(observado,prediccion):

    """
    Calculo de las metricas del modelo
    """
    
    from sklearn.metrics import (mean_absolute_percentage_error,mean_absolute_error,mean_squared_error,r2_score)

    return {
            'mape':100*mean_absolute_percentage_error(observado, prediccion),
            'mae':mean_absolute_error(observado, prediccion),
            'mse':mean_squared_error(observado, prediccion,squared=False),
            'rmse':mean_squared_error(observado, prediccion,squared=True),
            'r2': r2_score(observado, prediccion, multioutput='variance_weighted')
            }

# Función splitdata narx
def split_data(pd_model_id,exog_order,auto_order,exog_delay,prediction_order,exogena,y_output):
    """
    Funcion para dale estructura a los datos
    """

    x_data = []
    y_data = []

    min_index = max([exog_order+exog_delay,auto_order])
    index_split = pd_model_id[min_index:].index

    for t in range(len(index_split)):

        pd_to_split = pd_model_id[pd_model_id.index<=index_split[t]][-min_index-1:]

        exogen_values = pd_to_split[(pd_to_split.shape[0]-exog_delay-exog_order):(pd_to_split.shape[0]-exog_delay)][[exogena]].values.reshape(-1)
        auto_values = pd_to_split[-auto_order-1:][[y_output]].values.reshape(-1)

        x_data.append(np.concatenate([exogen_values, auto_values[:-1]],axis=None))
        y_data.append(auto_values[-1])
        
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

# Función prediccion narx
def predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order,exogena,y_output):
    """
    Funcion para predecir a un paso
    """
    
    data_proces = pd.concat([data_predict,data_exogena[list(data_predict)]])
    data_proces['type'] = 'data_in'

    date_min = data_proces[data_proces[y_output].isnull()].index.min()
    date_max = data_proces[data_proces[y_output].isnull()].index.max()

    date = date_min
    while date <= date_max:
        x_data_test, y_data_test = split_data(data_proces[data_proces.index<=date],
                                                exog_order,
                                                auto_order,
                                                exog_delay,
                                                prediction_order,
                                                exogena,y_output)

        predit = model.predict(x_data_test[-1].reshape(1, x_data_test.shape[1]), verbose=0).reshape(-1)
        data_proces.loc[(data_proces.index==date),y_output]=predit

        date = data_proces[data_proces[y_output].isnull()].index.min()

    return data_proces[data_proces.index>=date_min]



# Función para grafico sst
def graf_sst(data_figure_ajuste,data_figure_validacion,data_figure_pronostico,y,y_predict):
    
    import plotly.graph_objects as go
    from plotly.graph_objects import Layout
    import pandas as pd

    fig = go.Figure(layout=Layout(plot_bgcolor='rgba(0,0,0,0)'))

    fig.add_annotation(x=data_figure_ajuste.index.max() - pd.DateOffset(months=3*12),#pd.Timestamp('2019-01-01'),
                y=29,
                text="Entrenamiento",
                showarrow=False,
                yshift=10)

    fig.add_trace(go.Scatter(x=data_figure_ajuste.index, y=data_figure_ajuste[y_predict],
                            mode='lines+markers',name='Pronóstico entrenamiento',
                            marker_symbol='x-thin',
                            marker_line_width=3,
                            marker_size=3,
                            marker_line_color='#000000',
                            marker_color='#000000',
                            line=dict(color='#FF7203', width=2)))

    fig.add_trace(go.Scatter(x=data_figure_ajuste.index, y=data_figure_ajuste[y],
                            mode='lines+markers',name='SST entrenamiento',
                            marker_symbol='x-thin',
                            marker_line_width=3,
                            marker_size=3,
                            marker_line_color='#000000',
                            marker_color='#000000',
                            line=dict(color='#C10101', width=2)))


    months = int(data_figure_pronostico.shape[0]/3)
    fig.add_annotation(x= data_figure_validacion.index.min() + pd.DateOffset(months=months)  ,#pd.Timestamp('2021-07-01'),
                y=29,
                text="Validación",
                showarrow=False,
                yshift=10)
    fig.add_trace(go.Scatter(x=data_figure_validacion.index, y=data_figure_validacion[y_predict],
                        mode='lines+markers',name='Pronóstico validación',                       
                            marker_symbol='square',
                            marker_line_width=2,
                            marker_size=3,
                            marker_line_color='#030BFF',
                            marker_color='#030BFF', 
                            line=dict(color='#FF7203', width=2)
                            ))

    fig.add_trace(go.Scatter(x=data_figure_validacion.index, y=data_figure_validacion[y],
                        mode='lines+markers',name='SST validación',
                        marker_symbol='square',
                        marker_line_width=2,
                        marker_size=3,
                        marker_line_color='#030BFF',
                        marker_color='#030BFF', 
                        line=dict(color='#C10101', width=2)))



    fig.add_annotation(x=data_figure_pronostico.index.min() + pd.DateOffset(months=months),#pd.Timestamp('2022-09-01'),
                y=29,
                text="Pronóstico",
                showarrow=False,
                yshift=10)
    fig.add_trace(go.Scatter(x=data_figure_pronostico.index, y=data_figure_pronostico[y_predict],
                            text=data_figure_pronostico[y_predict].apply(lambda x: str(round(x,2)) ),
                            textposition="top right",
                            marker_symbol='star',
                            marker_line_width=3,
                            marker_size=3,
                            marker_line_color='#EA9800',
                            marker_color='#EA9800',
                            mode='lines+markers+text',name='Pronóstico SST',
                            line=dict(color='#FF7203', width=2,dash='dash')))

    fig.add_vline(x=data_figure_ajuste.index.max(), line_width=3, line_dash="dash", line_color="#580606")
    fig.add_vline(x=data_figure_validacion.index.max(), line_width=3, line_dash="dash", line_color="#580606")


    fig.update_xaxes(tickformat="%Y/%m",showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)


    fig.update_traces(textfont_size=11)
    fig.update_layout(title="""
                        SST promedio en la región NIÑO 3.4 
                        <br><sup>Pronóstico para el periodo {date_init} al {date_fin}</sup>
                        """.format(date_init=str(data_figure_pronostico.index.min().strftime('%Y/%m')),
                                date_fin=str(data_figure_pronostico.index.max().strftime('%Y/%m')) ),
                    xaxis_title='Mes',
                    yaxis_title='Temperatura (°C)',
                    legend_title_text='Serie',
                    legend_title = dict( font = dict(size = 25)),
                    legend=dict(y=0.5,
                                #traceorder='reversed',
                                font_size=22),
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    height=800,
                    width=1500,
                    font = dict(size = 22),
                    xaxis_range=[ data_figure_ajuste.index.max() - pd.DateOffset(months=5*12) , data_figure_pronostico.index.max()+pd.DateOffset(months=3)  ],

                    )

    return fig

# funcion para el grafico de la precipitacion total
def precipitacion_graf( pd_precipitacion,
                        rows=5, 
                        cols=3,
                        park = 'Cerro Saroche',
                        range=[0,10]
                        ):
    """
    Funcion para generar el grafico de la precipitacion total para cada id del parque.]
    """

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    column_titles = list(map(lambda x: 'Punto id:' + str(int(x)), pd_precipitacion.id_point.unique().tolist() ))

    fig = make_subplots(rows=rows,
                        cols=cols,
                        subplot_titles=column_titles,
                        horizontal_spacing=0.1,
                        vertical_spacing=0.13)


    row_order = pd_precipitacion.id_point.unique().reshape(rows,cols).tolist()
    for order in list(map(lambda x: [row_order.index(x)]+x,
                            row_order
                            )):
        
        for col in order[1:]:
            
            coll = order[1:].index(col)+1
            id_ = order[1:][coll-1]
            
            if id_==0:
                showlegend=True
            else:
                showlegend=False
                
                
            data_fig = pd_precipitacion[pd_precipitacion.id_point==id_].copy()

            fig.add_trace(
                go.Scatter(x=data_fig.index,
                        y=data_fig.precipitacion_mm,
                        mode='lines',
                        name='Precipitación total',# 096ACC,
                        line=dict(color='#0059FF', width=2),
                        legendgroup='group1',
                        showlegend=showlegend,
                        
                        ),
                row=order[0]+1, 
                col=coll,
                    
            )
            
            
            fig.update_xaxes(title_text='Mes',title_font=dict(size=12))
            fig.update_yaxes(title_text='Precipitación (mm)',
                            title_font=dict(size=12),
                            range=range
                            )
            
    fig['layout']['title']['y']=0.98
    fig['layout']['margin']['t']=100

    min_date = pd_precipitacion.index.min()
    max_date = pd_precipitacion.index.max()
            
    #        
    fig.update_xaxes(tickformat="%Y/%m",showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='#E4E4E4',mirror=True,
                    ticks="outside", tickwidth=2, tickcolor='#5C2B05', ticklen=10)        
    fig.update_layout( height=800,
                    width=1500,
                    showlegend = True,
                    font = dict(size = 14),
                    template='plotly_white',
                    title_text=f"""
                                Precipitación total periodo {min_date.strftime("%Y/%m")} al {max_date.strftime("%Y/%m")}
                                <br><sup>Parque {park}</sup>
                                """,
                    xaxis_range=[min_date,max_date],
                    legend_title_text='Serie',
                    legend_title = dict( font = dict(size = 14)),
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    legend = dict(
                                    #orientation="h",
                                    yanchor="bottom",
                                    y=1.05,
                                    xanchor="right",
                                    x=1,
                                    font_size=14
                                )
                    )
    return fig