from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from tensorflow import keras


#--------------------------
# MyLRSchedule
#--------------------------

class MyLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate 
    """

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate / (step + 1)


#--------------------------
# KERAS_NARX_NINO34
#--------------------------

class model_base:

    #--
    def estructura_red(self, nout, layers, activation, kernel_initializer):
        """
        Funcion para dar estructura a la red neuronal
        """

        # Integrando valores
        self.nout = nout
        self.layers = layers
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        # Definiendo la red
        self.layer_input = keras.layers.Input(shape=(self.layers[0],))
        self.hidden = keras.layers.Dense(units = self.layers[1],
                                        activation=self.activation[0],
                                        use_bias=True,
                                        kernel_initializer=self.kernel_initializer[0],
                                        bias_initializer="zeros",
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None
                                    )(self.layer_input)



        if len(layers)>2:
            for layer in zip(layers[2:],activation[1:],kernel_initializer[1:]):
                self.hidden = keras.layers.Dense(units = layer[0],
                                                activation = layer[1],
                                                use_bias=True,
                                                kernel_initializer=layer[2],
                                                bias_initializer="zeros",
                                                kernel_regularizer=None,
                                                bias_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                bias_constraint=None
                                        )(self.hidden)


        self.output = keras.layers.Dense(self.nout)(self.hidden)

        # Definiendo modelo
        self.model = keras.models.Model(inputs=[self.layer_input],
                                        outputs=[self.output])

    #--
    def taining_model(self, learning_rate,
                            loss,
                            metrics,
                            epochs,
                            batch_size,
                            validation_split,
                            path_checkpoint,
                            patience,
                            min_delta,
                            monitor,
                            mode):
        
        """
        Funcion para el entrenamiento de la red
        """

        # Integrando valores
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.path_checkpoint = path_checkpoint
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor 
        self.mode = mode

        # Definiendo optimizador y learning rate
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                beta_1=0.9,
                                                beta_2=0.999,
                                                epsilon=1e-08,
                                                amsgrad=False,
                                                name="Adam"
                                            )

        # Compilando el modelo a entrenar
        self.model.compile( optimizer=self.optimizer,
                            loss=self.loss,
                            metrics=self.metrics,
                            loss_weights=None,
                            weighted_metrics=None,
                            run_eagerly=None,
                            steps_per_execution=None,
                            jit_compile=None)

        # Entrenando la red
        self.es_callback = keras.callbacks\
                            .EarlyStopping( monitor=self.monitor,
                                            min_delta=self.min_delta,
                                            patience=self.patience,
                                            verbose=0,
                                            mode=self.mode,
                                            baseline=None,
                                            restore_best_weights=False
                                            )

        self.modelckpt_callback = keras.callbacks\
                                    .ModelCheckpoint(filepath = self.path_checkpoint,
                                                    monitor=self.monitor,
                                                    verbose=0,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode="auto",
                                                    save_freq="epoch",
                                                    options=None,
                                                    initial_value_threshold=None
                                                    )

                                        
    #---
    @classmethod
    def create(cls, nout = 1,
                    layers = [2, 2, 2],
                    activation = "selu",
                    kernel_initializer = "lecun_normal",
                    learning_rate = 1e-3,
                    loss = "mean_squared_error",
                    metrics = ['mae'], 
                    epochs = 5000,
                    batch_size = 32,
                    validation_split = 0.30,
                    path_checkpoint = "./model_checkpoint/narx_sst_nino34.h5",
                    patience = 100,
                    min_delta = 0,
                    monitor = "val_loss",
                    mode = 'min'):

        """
        Metodo de clase para crear el objeto
        """

        # Clase
        clase = cls()
    
        # Estructura
        clase.estructura_red(nout = nout,
                            layers = layers,
                            activation = activation,
                            kernel_initializer = kernel_initializer)

        # Entrenamiento
        clase.taining_model(learning_rate = learning_rate,
                            loss = loss,
                            metrics = metrics,
                            epochs = epochs,
                            batch_size = batch_size,
                            validation_split = validation_split,
                            path_checkpoint = path_checkpoint,
                            patience = patience,
                            min_delta = min_delta,
                            monitor = monitor,
                            mode = mode)

        return clase



#--------------------------
# KERAS_NARX_NINO34
#--------------------------
class KERAS_NARX_NINO34:

    """
    Clase para el entrenamiento de modelos Keras
    """

    def __init__(self,pd_model, y_output, exogena, prediction_order, auto_order, exog_order, exog_delay):
        self.pd_model = pd_model[ [y_output] + exogena].sort_index(ascending=True)
        self.y_output = y_output
        self.exogena = exogena
        self.prediction_order = prediction_order
        self.auto_order = auto_order
        self.exog_order = exog_order
        self.exog_delay = exog_delay


    def split_data(self):
        """
        Funcion para determinar la data de entrenamiento
        """

        self.data_test = self.pd_model[:-self.prediction_order]
        self.input_data, self.output_data = self.date_window(pd_model = self.data_test,
                                        auto_order = self.auto_order,
                                        exog_order = self.exog_order,
                                        exog_delay = self.exog_delay,
                                        exogena = self.exogena,
                                        output = self.y_output)

        # Neuronas capa entrada
        _ , self.ninp = self.input_data.shape


    def training_model(self, layers,
                            activation,
                            kernel_initializer,
                            learning_rate,
                            loss,
                            metrics,
                            epochs,
                            batch_size ,
                            validation_split,
                            path_checkpoint,
                            patience,
                            min_delta,
                            monitor,
                            mode):

        """
        Funcion para el Entrenamiento del modelo Keras
        """

        self.class_model = model_base.create(layers = layers,
                                            activation = activation,
                                            kernel_initializer = kernel_initializer,
                                            learning_rate = learning_rate,
                                            loss = loss,
                                            metrics = metrics,
                                            epochs = epochs,
                                            batch_size = batch_size,
                                            validation_split = validation_split,
                                            path_checkpoint = path_checkpoint,
                                            patience = patience,
                                            min_delta = min_delta,
                                            monitor = monitor,
                                            mode = mode,
                                            nout = 1)

        self.model = self.class_model.model
        self.epochs = epochs
        self.validation_split = validation_split

        self.history = self.model.fit(  x=self.input_data,
                                        y=self.output_data,
                                        batch_size=None,
                                        epochs=self.epochs,
                                        verbose=0,
                                        callbacks=[self.class_model.es_callback, self.class_model.modelckpt_callback],
                                        validation_split=self.validation_split,
                                        validation_data=None,
                                        shuffle=True,
                                        class_weight=None,
                                        sample_weight=None,
                                        initial_epoch=0,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        validation_batch_size=None,
                                        validation_freq=1,
                                        max_queue_size=10,
                                        workers=1,
                                        use_multiprocessing=False)


    def loss_plot(self,title,path):
        """
        Funcion para la visualizacion de la perdida
        """

        self.visualizar_perdida(history=self.history, title=title, path=path)


    def validation_test(self,batch_size):

        """
        Funcion para estudiar resultados
        """

        self.batch_size = batch_size

        # Prediccion para entrenamiento
        prediction_test = self.model.predict(self.input_data,
                                        batch_size=self.batch_size,
                                        verbose=0,
                                        steps=None,
                                        callbacks=None,
                                        max_queue_size=10,
                                        workers=1,
                                        use_multiprocessing=False)

        # Guardando predicciones
        self.pd_validation = self.pd_model.copy().reset_index(drop=False)
        self.pd_validation['prediction'] = np.nan
        self.pd_validation['type'] = 'training'
        self.pd_validation.loc[ self.data_test.shape[0]: , 'type' ] = 'validation'

        # Prediccion training
        self.pd_validation.loc[ range(max(self.auto_order, self.exog_order),(self.data_test.shape[0])), 'prediction' ] = prediction_test

        # Generando auto predicion
        pd_selfPrediction = self.data_test.copy()
        pd_selfPrediction['type'] = 'historical'

        for x in range(self.prediction_order):
            selfPrediction = self.forecast_one_step( data=pd_selfPrediction.copy(),
                                                model=self.model,
                                                auto_order=self.auto_order,
                                                exog_order=self.exog_order,
                                                exog_delay=self.exog_delay,
                                                exogena=self.exogena,
                                                y_output=self.y_output,
                                                batch_size=self.batch_size)

            selfPrediction[self.exogena] = selfPrediction[self.y_output]

            pd_selfPrediction = pd.concat([pd_selfPrediction, selfPrediction]).copy()  

        self.pd_validation.loc[ self.data_test.shape[0]: , 'prediction' ] = pd_selfPrediction[-self.prediction_order:][self.y_output].to_numpy()

        # Self prediction 
        pd_selfPrediction = self.pd_model.copy()
        for x in range(self.prediction_order):
            selfPrediction = self.forecast_one_step( data=pd_selfPrediction.copy(),
                                                model=self.model,
                                                auto_order=self.auto_order,
                                                exog_order=self.exog_order,
                                                exog_delay=self.exog_delay,
                                                exogena=self.exogena,
                                                y_output=self.y_output,
                                                batch_size=self.batch_size)

            selfPrediction[self.exogena] = selfPrediction[self.y_output]
            pd_selfPrediction = pd.concat([pd_selfPrediction, selfPrediction]).copy()  


        pd_forecast = pd_selfPrediction[-self.prediction_order:][[self.y_output]+['type']].reset_index()
        pd_forecast['prediction'] = pd_forecast[self.y_output]
        pd_forecast[self.y_output] = np.nan
        pd_forecast[self.exogena] = np.nan

        self.pd_validation = pd.concat([self.pd_validation, pd_forecast[self.pd_validation.columns]])



    #--
    @staticmethod
    def forecast_one_step(data,model,auto_order,exog_order,exog_delay,exogena,y_output,batch_size):
        """
        Funcion para la prediccion a one step
        """

        # Pandas Data Frame
        pd_update = pd.DataFrame()
        # Agregando Index 
        pd_update['periodo'] = pd.to_datetime([ (data.index.max() + pd.DateOffset(months=1) ).strftime('%Y-%m')])
        pd_update = pd_update.set_index('periodo')
        #print(pd_update.index )
        pd_update[y_output] = np.nan
        for col in exogena:
            pd_update[col] = np.nan

        pd_update['type'] = 'self_prediction'

        # Formato a los datos
        input_data_validate, _ = KERAS_NARX_NINO34.date_window(pd_model=data.copy(),
                                                                auto_order=auto_order,
                                                                exog_order=exog_order,
                                                                exog_delay=exog_delay,
                                                                exogena=exogena,
                                                                output=y_output)

        # Data para pronostico
        past_row = input_data_validate[-1].reshape(1, input_data_validate.shape[1])

        # prediccion
        prediction_validation = model.predict(  past_row,
                                                batch_size=batch_size,
                                                verbose=0,
                                                steps=None,
                                                callbacks=None,
                                                max_queue_size=10,
                                                workers=1,
                                                use_multiprocessing=False)

        pd_update[y_output] = prediction_validation.flat[0]

        return pd_update

    #--
    @staticmethod
    def date_window(pd_model,auto_order,exog_order,exog_delay,exogena,output):
        """
        Funcion para estructurar los datos de entrada del modelo NARX
        """

        X = pd_model[exogena].to_numpy().astype(float)
        y = pd_model[[output]].to_numpy().astype(float)

        # Data para el mmodelo
        input_data = []
        output_data = []
        for t in range(max(auto_order, exog_order), len(y)):
            input_data.append(np.concatenate((y[(t - auto_order + 1):(t + 1)], X[(t - exog_delay - exog_order + 1):(t - exog_delay + 1)]), axis=0) )
            output_data.append(np.array(y[t]))

        # Input del modelo
        input_data = np.array(input_data)
        # Output del modelo
        output_data = np.array(output_data)

        return (input_data.reshape(input_data.shape[0],input_data.shape[1]), output_data)

    #--
    @staticmethod
    def visualizar_perdida(history, title, path):
        """
        Funcion para visualizar la funcion de perdida de la red NARX
        """
        params = {'legend.fontsize': 'x-large',
                #'figure.figsize': (15, 10),
                'axes.labelsize': 'x-large',
                'axes.titlesize':'x-large',
                'xtick.labelsize':'x-large',
                'ytick.labelsize':'x-large'}

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure(figsize=(15,10))
        pylab.rcParams.update(params)
        plt.plot(epochs, loss, "b", label="Entrenamiento")
        plt.plot(epochs, val_loss, "r", label="Validación")
        
        plt.title('Pérdida en el entrenamiento y validación',loc='left',fontsize=18)
        plt.suptitle(title,ha='left',fontsize=30,x=0.12)
        plt.xlabel("Épocas",fontsize=18)
        plt.ylabel("Pérdida",fontsize=18)
        plt.legend()

        plt.savefig(path)

        plt.show()



    #--
    @staticmethod
    def metrics(hidden_layer_sizes,observado,prediccion):

        """
        Calculo de las metricas del modelo
        """
        
        from sklearn.metrics import (mean_absolute_percentage_error,mean_absolute_error,mean_squared_error,r2_score)

        return {'hidden_layer_sizes': hidden_layer_sizes,
              'map':[mean_absolute_percentage_error(observado, prediccion)],
              'mae':[mean_absolute_error(observado, prediccion)],
              'rmse':[mean_squared_error(observado, prediccion,squared=False)],
              'r2': [r2_score(observado, prediccion, multioutput='variance_weighted')],
              }


    #--
    @staticmethod
    def graf(data_figure_ajuste,data_figure_validacion,data_figure_pronostico,y,y_predict):
        
        import plotly.graph_objects as go
        from plotly.graph_objects import Layout

        fig = go.Figure(layout=Layout(plot_bgcolor='rgba(0,0,0,0)'))

        fig.add_annotation(x=data_figure_ajuste.index.max() - pd.DateOffset(months=3*12) ,#pd.Timestamp('2019-01-01'),
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
                                mode='lines+markers',name='SSTT entrenamiento',
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
                            mode='lines+markers',name='SSTT validación',
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
                                mode='lines+markers+text',name='Pronóstico SSTT',
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
                        xaxis_range=[ data_figure_ajuste.index.max() - pd.DateOffset(months=5*12) , data_figure_pronostico.index.max() ],

                        )

        return fig