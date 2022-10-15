
# ## Análisis Precipitación Total Parque Nacional Cerro Saroche
# 
# **PROYECTO:** SISTEMA PARA EL SEGUIMIENTO DE ECOSISTEMAS VENEZOLANOS \
# **AUTOR:** Javier Martinez

import os
import pickle
import sys
 
from utils.MONGO import CONEXION
from utils.UTILS import *
from datetime import datetime
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings('ignore')

#---------------------
def split_data(pd_model_id,exog_order,auto_order,exog_delay,prediction_order):
    """
    Funcion para dale estructura a los datos
    """

    min_index = max([exog_order,auto_order])

    x_data = []
    y_data = []

    for t in pd_model_id[min_index:].index:

        #t = pd_model_id[min_index:].index.min()

        to_split = pd_model_id[[y_output,exogena]]
        to_split = to_split[(t-pd.DateOffset(months=auto_order)):t].copy()

        # Exogena
        x_exo = to_split[exogena][(t-pd.DateOffset(months=exog_delay+exog_order)):(t-pd.DateOffset(months=exog_delay+1))]\
                            .to_numpy()\
                            .astype(float)\
                            .reshape(-1)

        # Auto
        x_auto = to_split[y_output][:-1]\
                    .to_numpy()\
                    .astype(float)\
                    .reshape(-1)

        x_data.append(np.concatenate([x_exo, x_auto],axis=None))
        y_data.append(to_split[y_output][-1])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

#---------------------
def predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order):
  """
  Funcion para predecir a un paso
  """
  data_predict = data_predict.copy()
  data_predict['type'] = 'data_in'

  for t in data_exogena.index:
    
      x_data_test, y_data_test = split_data(data_predict,
                                          exog_order,
                                          auto_order,
                                          exog_delay,
                                          prediction_order)


      predit = model.predict(x_data_test[-1].reshape(1, x_data_test.shape[1]), verbose=0).reshape(-1)
      exo = data_exogena[data_exogena.index==t][exogena][0]

      data_test = pd.DataFrame({'periodo':t, y_output:predit, exogena:exo,'type':'data_out'},index=[0])
      data_test.index = pd.to_datetime(data_test.periodo)

      data_predict = pd.concat([data_predict,
                                data_test[list(data_predict)]
                              ])

  return data_predict

#---------------------

#---
if __name__ == "__main__":
    print('> Directorio actual: ', os.getcwd())

    # Parque
    park = sys.argv[1]
    id_point = int(sys.argv[2])

    y_output = sys.argv[3]
    exogena = sys.argv[4]

    prediction_order = int(sys.argv[5])# rango de prediccion
    auto_order = int(sys.argv[6]) # componente autoregresiva
    exog_order = int(sys.argv[7])# componente exogena qm
    exog_delay = int(sys.argv[8])# componente exogena dm

    f_activation = sys.argv[9]

    # Parametros de modelos
    patience = 100
    epochs=500

    # Directorio del experimento
    DIR = f'./{park}/'
    experimento = f'experiments/narx/precipitacion/id_point_{id_point}'

    try:
        os.mkdir(f'{DIR}{experimento}')
    except:
        pass

    # Consulta de la data
    # Realizando consulta
    # Creando la conexión con MongoDB
    db = CONEXION.conexion()
    db.list_collection_names()

    meteorological = db.meteorological.find({"park":park, 'id_point':id_point})

    # Generando pandas dataframe
    data_pandas = pd.DataFrame([file for file in meteorological])
    data_pandas['periodo'] = data_pandas.time.apply(lambda x: datetime.fromordinal(x))
    data_pandas['mes_year'] =  data_pandas['periodo'].dt.strftime('%B-%Y')
    data_pandas.index = pd.to_datetime(data_pandas.periodo)

    # # Estudio Precipitación
    pd_precipitacion = data_pandas[['id_point', 'latitud', 'longitud',
                                    'precipitacion_mm']]

    # Realizando consulta
    data_sst = db.estimateSSTNino34.find()

    # Generando pandas dataframe
    pd_sst = pd.DataFrame([file for file in data_sst])[['nino34_mean','oni','time']]
    pd_sst['periodo'] = pd_sst.time.apply(lambda x: datetime.fromordinal(x))
    pd_sst.index = pd.to_datetime(pd_sst.periodo)

    # Transformacion
    oni_transformacion = MinMaxScaler() #LogMinimax.create( pd_sst.oni.to_numpy() )
    oni_transformacion.fit(pd_sst[['oni']])

    pd_sst['oni'] = oni_transformacion.transform( pd_sst[['oni']] )

    # Transformacion
    sst_transformacion = LogMinimax.create( pd_sst.nino34_mean.to_numpy() )

    pd_sst['sst_t'] = sst_transformacion.transformacion()

    # # Integrando base de datos
    # Entrenamiento
    pd_model = pd.merge(pd_precipitacion.reset_index(drop=False),pd_sst[['oni','sst_t']].reset_index(drop=False),
                        on=['periodo'],
                        how='left'
                        )

    # Pronostico
    pd_sst_pron = pd_sst[['periodo','oni','sst_t']][pd_sst.periodo > pd_model.periodo.max()].copy()

    # # Ajustando modelo NARX
    # Transformacion
    transformacion = LogMinimax.create( pd_precipitacion.precipitacion_mm.to_numpy() )

    pd_model['precip_t'] = transformacion.transformacion()

    # Modelo según ID point
    pd_model_id = pd_model[pd_model.id_point==id_point]
    pd_model_id.index = pd.to_datetime(pd_model_id.periodo)
    pd_model_id = pd_model_id[[y_output,exogena]]

    x_data, y_data = split_data(pd_model_id,exog_order,auto_order,exog_delay,prediction_order)

    # Entrenamiento y validación
    x_train = x_data[:-prediction_order]
    x_vasl = x_data[-prediction_order:]

    y_train = y_data[:-prediction_order]
    y_vasl = y_data[-prediction_order:]

    # Modelo NARX
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Metrícas
    mae = keras.metrics.MeanAbsoluteError()
    rmse = keras.metrics.RootMeanSquaredError()

    confi = {'Input':{'batch_size':None,
                    'name':'input',
                    'dtype':None,
                    'sparse':None,
                    'tensor':None,
                    'ragged':None,
                    'type_spec':None},
            'Dense':{'use_bias':True,
                    'kernel_regularizer':None,
                    'bias_regularizer':None,
                    'activity_regularizer':None,
                    'kernel_constraint':None,
                    'bias_constraint':None
                    }
            }

    total = int(2*x_train.shape[-1]/3)
    n_neurons = [int(total)]
    #n_neurons = [total]

    activation = len(n_neurons)*[f_activation]
    kernel_initializer = 'lecun_normal'
    bias_initializer = 'zeros'

    # Modelo
    model = keras.models.Sequential()

    # Entradas
    model.add(keras.layers.Input(shape=(x_train.shape[-1],),
                                        batch_size = confi.get('Input').get('batch_size'),
                                        name = confi.get('Input').get('name'),
                                        dtype = confi.get('Input').get('dtype'),
                                        sparse = confi.get('Input').get('sparse'),
                                        tensor = confi.get('Input').get('tensor'),
                                        ragged = confi.get('Input').get('ragged'),
                                        type_spec = confi.get('Input').get('type_spec')
                                        ))

    model.add(keras.layers.Dense(   units=n_neurons[0],
                                    activation=activation[0],
                                    use_bias = confi.get('Dense').get('use_bias'),
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer = confi.get('Dense').get('kernel_regularizer'),
                                    bias_regularizer = confi.get('Dense').get('bias_regularizer'),
                                    activity_regularizer = confi.get('Dense').get('activity_regularizer'),
                                    kernel_constraint = confi.get('Dense').get('kernel_constraint'),
                                    bias_constraint = confi.get('Dense').get('bias_constraint')
                                    ))
                                    
    model.add(keras.layers.Dropout(0.1))


    # Hidden Leyers
    if len(n_neurons)>1:
        for index in list( range(1, len(n_neurons)) ):

            model.add(keras.layers.Dense(   units=n_neurons[index],
                                            activation=activation[index],
                                            use_bias = confi.get('Dense').get('use_bias'),
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer = confi.get('Dense').get('kernel_regularizer'),
                                            bias_regularizer = confi.get('Dense').get('bias_regularizer'),
                                            activity_regularizer = confi.get('Dense').get('activity_regularizer'),
                                            kernel_constraint = confi.get('Dense').get('kernel_constraint'),
                                            bias_constraint = confi.get('Dense').get('bias_constraint')
                                            ))
                                            
            #model.add(keras.layers.Dropout(0.001))

    # Out
    model.add(keras.layers.Dense(   units=1,
                                    activation='linear',
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer
                                    ))
                                    

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[mae,rmse]) 

    callback = keras.callbacks.EarlyStopping(
                                                monitor="loss",
                                                min_delta=0,
                                                patience=patience,
                                                verbose=0,
                                                mode="min",
                                                baseline=None,
                                                restore_best_weights=False,
                                            )

    # Entrenamiento
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=1,
                        verbose=0,
                        workers=2,
                        callbacks=[callback])

    print(f'Total epocas:{len(history.epoch)}')

    # Evaluación entrenamiento
    # make predictions
    trainPredict = model.predict(x_train, verbose=0).reshape(-1)
    testPredict = model.predict(x_vasl, verbose=0).reshape(-1)

    # Data de test
    trainind_pd = pd.DataFrame(trainPredict,
                                index = pd_model_id[-(trainPredict.shape[0]+prediction_order):-(prediction_order)].index,
                                columns=['prediction']
                                )

    trainind_pd[y_output] = y_train.reshape(-1)
    trainind_pd['type'] = 'training'

    trainind_pd['precipitacion_mm'] = trainind_pd[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    trainind_pd['prediction_precipitacion_mm'] = trainind_pd['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    trainind_pd = pd.merge(trainind_pd,pd_model_id[[exogena]].reset_index(drop=False),
                            on=['periodo'],
                            how='left')

    trainind_pd.index = pd.to_datetime(trainind_pd.periodo)

    # Validacion entrenamiento
    trainig_metrics = metrics(observado=trainind_pd.precipitacion_mm,
                            prediccion=trainind_pd.prediction_precipitacion_mm)

    # Evaluación validación
    # Data de Validacion
    validation_pd = pd.DataFrame(testPredict,
                                index = pd_model_id[-prediction_order:].index,
                                columns=['prediction']
                                )

    validation_pd[y_output] = y_vasl.reshape(-1)
    validation_pd['type'] = 'validation'

    validation_pd['precipitacion_mm'] = validation_pd[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    validation_pd['prediction_precipitacion_mm'] = validation_pd['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    validation_pd = pd.merge(validation_pd,pd_model_id[[exogena]].reset_index(drop=False),
                            on=['periodo'],
                            how='left')

    # Validacion entrenamiento
    validation_metrics = metrics(observado=validation_pd.precipitacion_mm,
                            prediccion=validation_pd.prediction_precipitacion_mm)

    # Test
    data_exogena = pd_model_id[-prediction_order:][[exogena]]
    data_predict = pd_model_id[pd_model_id.index < data_exogena.index.min()][[y_output,exogena]]

    pd_test = predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order)

    pd_test = pd_test[pd_test.type=='data_out'].rename(columns={y_output:'prediction'})
    pd_test['type'] = 'test'

    pd_test['precip_t'] = pd_model_id[pd_model_id.index > trainind_pd.periodo.max()][y_output]

    pd_test['precipitacion_mm'] = pd_test[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    pd_test['prediction_precipitacion_mm'] = pd_test['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    # Validacion entrenamiento
    test_metrics = metrics(observado=pd_test.precipitacion_mm,
                            prediccion=pd_test.prediction_precipitacion_mm)

    # Resultados del modelo
    dict_metrics = {'epocas':[len(history.epoch)],
                    'prediction_order':[prediction_order],
                    'auto_order':[auto_order],
                    'exog_order':[exog_order],
                    'exog_delay':[exog_delay],
                    'activation':[activation[0]],
                    'id_point':[id_point],
                    'n_neurons':str(n_neurons),
                    'capas':[len(n_neurons)],
                    'training_mse':[trainig_metrics["mse"]],
                    'training_rmse':[trainig_metrics["rmse"]],
                    'training_mae':[trainig_metrics["mae"]],
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

    # Pronósticoimport os
    data_predict = pd_model_id[[y_output,exogena]]
    data_exogena = pd_sst_pron[pd_sst_pron.index>data_predict.index.max()][[exogena]]

    pd_prediction = predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order)

    pd_prediction = pd_prediction[pd_prediction.type=='data_out'].rename(columns={y_output:'prediction'})
    pd_prediction['type'] = 'prediction'

    pd_prediction['precipitacion_mm'] = np.nan
    pd_prediction['prediction_precipitacion_mm'] = pd_prediction['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    pd_prediction['precip_t'] = np.nan

    # Uniendo informacion
    pd_summary = pd.concat([trainind_pd[list(pd_prediction)].reset_index(drop=False), 
                            pd_test[list(pd_prediction)].reset_index(drop=False), 
                            validation_pd[list(pd_prediction)].reset_index(drop=False), 
                            pd_prediction[list(pd_prediction)].reset_index(drop=False)
                            ])

    # Logica de guardado
    if os.listdir(f'{DIR}{experimento}') == []:

        # Modelo
        model.save(f'{DIR}{experimento}/model.h5')

        # Pesos
        model.save_weights(f'{DIR}{experimento}/weights.h5')

        # History
        with open(f'{DIR}{experimento}/history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
        # guardando resultados
        pd_summary.to_pickle(f'{DIR}{experimento}/predicciones.pkl')

    else:
        files = [x for x in os.listdir(f'{DIR}{experimento}') if x.find('summary')!=-1 ]
        total_summary = pd.concat([pd.read_csv(f'{DIR}{experimento}/{file}') for file in files])
        print( f"Actual: {validation_metrics['r2']}; Best Model: {total_summary.validation_r.max()}" )

        if validation_metrics['r2'] > total_summary.validation_r.max(): 

            # Modelo
            model.save(f'{DIR}{experimento}/model.h5')

            # Pesos
            model.save_weights(f'{DIR}{experimento}/weights.h5')

            # History
            with open(f'{DIR}{experimento}/history.pkl', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            
            # guardando resultados
            pd_summary.to_pickle(f'{DIR}{experimento}/predicciones.pkl')


    # Guardando Summary
    experi = f'{DIR}{experimento}/{id_point}_{len(n_neurons)}_{activation[0]}_{prediction_order}_{auto_order}_{exog_order}_{exog_delay}'
    experimento_pd.to_csv(f'{experi}_summary.csv',index=False)