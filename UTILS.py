# PROYECTO: SISTEMA PARA EL SEGUIMIENTO DE ECOSISTEMAS VENEZOLANOS
# AUTOR: Javier Martinez

import numpy as np

# Objeto para transformacion
class LogMinimax:
    """
    Transformacion LogMinimax
    """

    @classmethod
    def create(cls, values):

        clase = cls()

        clase.values = values
        clase.log_values = np.log(clase.values)
        clase.max = clase.log_values.max()
        clase.min = clase.log_values.min()

        return clase

    def transformacion(self):

        return (self.log_values - self.min)/(self.max - self.min)

    def inversa(self,y):

        return  np.exp( ( y*(self.max - self.min) ) + self.min )

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
                        """.format(date_init=str(data_figure_pronostico.index.min().strftime('%Y-%m-%d')),
                                date_fin=str(data_figure_pronostico.index.max().strftime('%Y-%m-%d')) ),
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