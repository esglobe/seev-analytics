from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from tensorflow import keras


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
        self.hidden = keras.layers.Dense(self.layers[1],
                                    activation=self.activation,
                                    kernel_initializer=self.kernel_initializer)(self.layer_input)

        if len(layers)>2:
            for layer in self.layers[2:]:
                self.hidden = keras.layers.Dense(layer,
                                        activation=self.activation,
                                        kernel_initializer=self.kernel_initializer)(self.hidden)

        self.output = keras.layers.Dense(self.nout)(self.hidden)

        # Definiendo modelo
        self.model = keras.models.Model(inputs=[self.layer_input],
                                        outputs=[self.output])

    #--
    def taining_model(self, learning_rate,
                            loss,
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.path_checkpoint = path_checkpoint
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor 
        self.mode = mode

        # Definiendo optimizador y learning rate
        self.optimizer = keras.optimizers.Nadam(learning_rate = self.learning_rate)

        # Compilando el modelo a entrenar
        self.model.compile(loss=self.loss, optimizer = self.optimizer)   

        # Entrenando la red
        self.es_callback = keras.callbacks\
                            .EarlyStopping(monitor=self.monitor,
                                        min_delta=self.min_delta,
                                        patience=self.patience,
                                        mode=self.mode )

        self.modelckpt_callback = keras.callbacks\
                                    .ModelCheckpoint(
                                            monitor=self.monitor,
                                            filepath=self.path_checkpoint ,
                                            verbose=0,
                                            save_weights_only=True,
                                            save_best_only=True)

                                        
    #---
    @classmethod
    def create(cls, nout = 1,
                    layers = [2, 2, 2],
                    activation = "selu",
                    kernel_initializer = "lecun_normal",
                    learning_rate = 1e-3,
                    loss = "mean_squared_error",
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

        self.history = self.model.fit(x=self.input_data,
                                    y=self.output_data,
                                    epochs=self.epochs,
                                    verbose=0,
                                    validation_split=self.validation_split,
                                    callbacks=[self.class_model.es_callback, self.class_model.modelckpt_callback]
                                    )


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



    #--
    def forecast_one_step(self,data,model,auto_order,exog_order,exog_delay,exogena,y_output,batch_size):
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
        input_data_validate, _ = self.date_window(pd_model=data.copy(),
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