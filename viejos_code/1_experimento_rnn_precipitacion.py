# PROYECTO: SISTEMA PARA EL SEGUIMIENTO DE ECOSISTEMAS VENEZOLANOS
# AUTOR: Javier Martinez

# Proceso de experimentos modelos RNN 
# Precipitacion total


import os
import sys

from utils.MONGO import CONEXION
from datetime import datetime
import pandas as pd
import numpy as np

from tensorflow import keras

from utils.UTILS import *

#
def precipitacion_data_rnn(data_pd,auto_order,y_output):
    """
    Funcion para darle estructura a los datos para modelo rnn
    """

    x_data = []
    y_data = []

    for t in range(auto_order+1, data_pd.shape[0]+1):
        x_data.append( np.array(data_pd[(t-auto_order-1):(t-1)]) )
        y_data.append( np.array( data_pd[(t-auto_order-1):t][[y_output]] )[-1] )

    x_data = np.array(x_data)
    y_data = np.array(y_data).reshape(x_data.shape[0],1,1)

    return x_data, y_data

#
def one_step_predict(data_for_test,pd_sst_pron,auto_order,y_output,feature):
    """
    Funcion para el pronostico a un paso de la precipitacion
    """

    for t in pd_sst_pron.index:

        x_entrada, y_salida = precipitacion_data_rnn(data_pd=data_for_test[data_for_test.index < t],
                                                    auto_order=auto_order,
                                                    y_output=y_output)

        predict = model.predict(x_entrada, verbose=0).reshape(-1)

        data_forecast = pd.DataFrame({y_output:predict[-1],
                                feature:pd_sst_pron[pd_sst_pron.index == t][feature][0]},
                                index = [t])
                                
        data_for_test = pd.concat([data_for_test, data_forecast]).copy()

    return data_for_test

#---
if __name__ == "__main__":

    # Parque
    park = sys.argv[1]
    id_point = int(sys.argv[2])
    y_output = sys.argv[3]
    feature =  sys.argv[4]

    prediction_order = int(sys.argv[5]) # rango de prediccion
    auto_order = int(sys.argv[6]) # componente autoregresiva


    # Creando la conexión con MongoDB
    db = CONEXION.conexion()
    db.list_collection_names()


    # Realizando consulta
    meteorological = db.meteorological.find({"park":park})

    # Generando pandas dataframe
    data_pandas = pd.DataFrame([file for file in meteorological])
    data_pandas['periodo'] = data_pandas.time.apply(lambda x: datetime.fromordinal(x))
    data_pandas['mes_year'] =  data_pandas['periodo'].dt.strftime('%B-%Y')
    data_pandas.index = pd.to_datetime(data_pandas.periodo)
    data_pandas.head()

    
    # Creando directorio
    DIR = f'./{park}/experiments/rnn'
    experimento = f'precipitacion/rnn_prec_{id_point}_{prediction_order}_{auto_order}'
    os.mkdir(f'{DIR}/{experimento}')

    pd_precipitacion = data_pandas[['id_point', 'latitud', 'longitud',
                                    'precipitacion_mm']]


    # Realizando consulta
    data_sst = db.estimateSSTNino34.find()

    # Generando pandas dataframe
    pd_sst = pd.DataFrame([file for file in data_sst])[['oni','time']]
    pd_sst['periodo'] = pd_sst.time.apply(lambda x: datetime.fromordinal(x))
    pd_sst.index = pd.to_datetime(pd_sst.periodo)

    oni_max = pd_sst.oni.max()
    oni_min = pd_sst.oni.min()

    pd_sst['oni'] = pd_sst['oni'].apply(lambda x: (x-oni_min)/(oni_max-oni_min))

    # Entrenamiento
    pd_model = pd.merge(pd_precipitacion.reset_index(drop=False),pd_sst[['oni']].reset_index(drop=False),
                        on=['periodo'],
                        how='left'
                        )

    # Pronostico
    pd_sst_pron = pd_sst[['periodo','oni']][pd_sst.periodo > pd_model.periodo.max()].copy()


    # Data
    data_pd = pd_model.query(f'id_point=={id_point}').copy()
    data_pd.index = pd.to_datetime(data_pd.periodo)

    # Transformacion
    transformacion = LogMinimax.create( data_pd.precipitacion_mm.to_numpy() )
    data_pd['precip_t'] = transformacion.transformacion()

    data_pd = data_pd[[y_output,feature]].sort_index().copy()
    data_pd.head()

    # Redefiniendo serie temporal
    x_data = []
    y_data = []

    for t in range(auto_order+1, data_pd.shape[0]+1):
        x_data.append( np.array(data_pd[(t-auto_order-1):(t-1)]) )
        y_data.append( np.array( data_pd[(t-auto_order-1):t][[y_output]] )[-1] )

    x_data = np.array(x_data)
    y_data = np.array(y_data).reshape(x_data.shape[0],1,1)

    # Entrenamiento y test
    x_train = x_data[:-prediction_order]
    x_vasl = x_data[-prediction_order:]

    y_train = y_data[:-prediction_order]
    y_vasl = y_data[-prediction_order:]

    # Modelo RNN
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Metrícas
    mae = keras.metrics.MeanAbsoluteError()
    rmse = keras.metrics.RootMeanSquaredError()

    model = keras.models.Sequential()

    rate = 0.2
    model.add(keras.layers.LSTM(auto_order, return_sequences=True ))
    model.add(keras.layers.Dropout(rate))

    model.add(keras.layers.LSTM(auto_order, return_sequences=False ))
    model.add(keras.layers.Dropout(rate))

    model.add(keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[mae,rmse]) 

    callback = keras.callbacks.EarlyStopping(
                                                monitor="loss",
                                                min_delta=0,
                                                patience=10,
                                                verbose=0,
                                                mode="min",
                                                baseline=None,
                                                restore_best_weights=False,
                                            )


    print(f'Iniciando entrenamiento {experimento}')

    epochs=500
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=1,
                        verbose=0,
                        workers=2,
                        callbacks=[callback])

    # Save History
    import pickle
    with open(f'{DIR}/{experimento}/rnn_precipitacion_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Evaluacion
    # make predictions
    trainPredict = model.predict(x_train, verbose=0).reshape(-1)
    testPredict = model.predict(x_vasl, verbose=0).reshape(-1)

    # Data de test
    trainind_pd = pd.DataFrame(trainPredict,
                                index = data_pd.index[:-prediction_order][-len(trainPredict):],
                                columns=['prediction']
                                )

    trainind_pd[y_output] = y_train.reshape(-1)
    trainind_pd['type'] = 'training'
    trainind_pd['precipitacion_mm'] = trainind_pd[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    trainind_pd['prediction_precipitacion_mm'] = trainind_pd['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    trainig_metrics = metrics(observado=trainind_pd.precipitacion_mm,prediccion=trainind_pd.prediction_precipitacion_mm)

    # Data de Validacion
    validation_pd = pd.DataFrame(testPredict,
                                index = data_pd.index[-prediction_order:],
                                columns=['prediction']
                                )

    validation_pd[y_output] = y_vasl.reshape(-1)
    validation_pd['type'] = 'validation'

    validation_pd['precipitacion_mm'] = validation_pd[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    validation_pd['prediction_precipitacion_mm'] = validation_pd['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    validation_metrics = metrics(observado=validation_pd.precipitacion_mm,prediccion=validation_pd.prediction_precipitacion_mm)

    # Test
    pd_test = one_step_predict(data_for_test=data_pd[:(data_pd.shape[0] - auto_order)],
                    pd_sst_pron=data_pd[-auto_order:],
                    auto_order=auto_order,
                    y_output=y_output,
                    feature=feature)

    pd_test = pd_test[-auto_order:].rename(columns={'precip_t':'prediction'})
    pd_test['type'] = 'test'
    pd_test[y_output] = data_pd[-auto_order:][y_output]


    pd_test['precipitacion_mm'] = pd_test[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    pd_test['prediction_precipitacion_mm'] = pd_test['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    test_metrics = metrics(observado=pd_test.precipitacion_mm,prediccion=pd_test.prediction_precipitacion_mm)

    # Pronostico
    pd_prediction = one_step_predict(data_for_test=data_pd,
                                    pd_sst_pron=pd_sst_pron,
                                    auto_order=auto_order,
                                    y_output=y_output,
                                    feature=feature)
    pd_prediction = pd_prediction.rename(columns={'precip_t':'prediction'})

    pd_prediction['type'] = 'prediction'

    pd_prediction[y_output] = np.nan

    pd_prediction['precipitacion_mm'] =  np.nan
    pd_prediction['prediction_precipitacion_mm'] = pd_prediction['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    # Resultados del modelo
    dict_metrics = {'epocas':[len(history.epoch)],
                    'auto_order':[auto_order],
                    'id_point':[id_point],
                    'training_mse':[history.history["loss"][-1]],
                    'training_rmse':[history.history["root_mean_squared_error"][-1]],
                    'training_mae':[history.history["mean_absolute_error"][-1]],
                    'trainig_mape':[trainig_metrics['mape']],
                    'trainig_r':[trainig_metrics['r2']],
                    'validation_mse':[validation_metrics["mse"]],
                    'validation_rmse':[validation_metrics["rmse"]],
                    'validation_mae':[validation_metrics["mae"]],
                    'validation_mape':[validation_metrics['mape']],
                    'validation_r':[validation_metrics['r2']],
                    'test_mse':[test_metrics["mse"]],
                    'test_rmse':[test_metrics["rmse"]],
                    'test_mae':[test_metrics["mae"]],
                    'test_mape':[test_metrics['mape']],
                    'test_r':[test_metrics['r2']]
                    }

    experimento_pd = pd.DataFrame.from_dict(dict_metrics)
    experimento_pd.to_csv(f'{DIR}/{experimento}/summary.csv',index=False)

    columns = [ 'precip_t',
                'prediction',
                #'oni',
                'type',
                'precipitacion_mm',
                'prediction_precipitacion_mm']

    # Uniendo informacion
    pd_summary = pd.concat([trainind_pd[columns], 
                            pd_test[columns], 
                            pd_prediction[columns]
                            ])
    pd_summary['periodo'] = pd.to_datetime(pd_summary.index.values)
    pd_summary = pd_summary[ ['periodo']+ columns ]
    pd_summary.to_pickle(f'{DIR}/{experimento}/predicciones.pkl')



