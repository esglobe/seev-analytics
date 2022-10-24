# ## Variación espacio-temporal precipitación total
# 
# **PROYECTO:** SISTEMA PARA EL SEGUIMIENTO DE ECOSISTEMAS VENEZOLANOS \
# **AUTOR:** Javier Martinez

# Directorio

import os
import pickle
import sys

print('> Directorio actual: ', os.getcwd()) 

import pandas as pd

from utils.MONGO import CONEXION
from utils.UTILS import *
from datetime import datetime

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


import warnings
warnings.filterwarnings('ignore')


#---
if __name__ == "__main__":
    print('> Directorio actual: ', os.getcwd())

    # # Conexión MongoDB 
    # Creando la conexión con MongoDB
    db = CONEXION.conexion()
    db.list_collection_names()


    # Parque
    park = sys.argv[1]
    id_point = int(sys.argv[2])
    print(id_point)

    y_output = sys.argv[3]
    exogena = sys.argv[4]

    prediction_order = int(sys.argv[5])# rango de prediccion
    auto_order = int(sys.argv[6]) # componente autoregresiva
    exog_order = int(sys.argv[7])# componente exogena qm
    exog_delay = int(sys.argv[8])# componente exogena dm

    f_activation = sys.argv[9]

    # Parametros de modelos
    patience = 5
    epochs=1000

    # # Descargando información

    # Realizando consulta
    meteorological = db.meteorological.find({"park":park, 'id_point':id_point})

    # Generando pandas dataframe
    data_pandas = pd.DataFrame([file for file in meteorological])
    data_pandas['periodo'] = data_pandas.time.apply(lambda x: datetime.fromordinal(x))
    data_pandas['mes_year'] =  data_pandas['periodo'].dt.strftime('%B-%Y')


    print(data_pandas[data_pandas.ndvi_media.notnull()].periodo.min())
    print((data_pandas[data_pandas.ndvi_media.notnull()].periodo.max()))

    pd_precipitacion = pd.read_pickle(f'./{park}/data/ann_precipitacion.pkl')[['park',
                                                                                'periodo',
                                                                                'year',
                                                                                'month',
                                                                                'id_point',
                                                                                'latitud',
                                                                                'longitud',
                                                                                'type',
                                                                                'prediction_ann',
                                                                                'ndvi_media']]


    # Transformacion
    ndvi_transformacion = MinMaxScaler() #LogMinimax.create( pd_sst.oni.to_numpy() )
    ndvi_transformacion.fit(pd_precipitacion[['prediction_ann','ndvi_media']])

    pd_precipitacion[['precipitation_ann_t','ndvi_t']] = ndvi_transformacion.transform( pd_precipitacion[['prediction_ann','ndvi_media']] )


    # Directorio experimentos
    DIR = f'./{park}/'
    experimento = f'experiments/narx/ndvi/{id_point}'

    try:
        os.mkdir(f'{DIR}{experimento}')
    except:
        pass

    # # Ajustando modelo NARX
    pd_model_id = pd_precipitacion[pd_precipitacion.id_point==id_point]
    pd_model_id.index = pd.to_datetime(pd_model_id.periodo)
    pd_model_id = pd_model_id[[y_output,exogena]].dropna().sort_index()


    x_data, y_data = split_data(pd_model_id,exog_order,auto_order,exog_delay,prediction_order,exogena,y_output)

    # Entrenamiento y validación
    x_train = x_data[:-prediction_order]
    x_vasl = x_data[-prediction_order:]

    y_train = y_data[:-prediction_order]
    y_vasl = y_data[-prediction_order:]

    # Modelo NARX

    import os
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
    n_neurons = [total]

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
                                            
            # model.add(keras.layers.Dropout(0.001))
            # print()

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
    trainind_pd['precipitation_ann_t'] = np.nan

    trainind_pd['id_point'] = id_point

    trainind_pd[['prediction_ann','ndvi_prediction']] = ndvi_transformacion.inverse_transform(trainind_pd[['precipitation_ann_t','prediction']])
    trainind_pd[['prediction_ann','ndvi_media']] = ndvi_transformacion.inverse_transform(trainind_pd[['precipitation_ann_t',y_output]])

    trainind_pd = trainind_pd.reset_index(drop=False)[['id_point', 'periodo','type','ndvi_prediction','ndvi_media']]

    # Validacion entrenamiento
    trainig_metrics = metrics(observado=trainind_pd.ndvi_media,
                            prediccion=trainind_pd.ndvi_prediction)


    # Data de test
    validation_pd = pd.DataFrame(testPredict,
                                index = pd_model_id[-prediction_order:].index,
                                columns=['prediction']
                                )

    validation_pd[y_output] = y_vasl.reshape(-1)
    validation_pd['type'] = 'validation'
    validation_pd['precipitation_ann_t'] = np.nan

    validation_pd['id_point'] = id_point

    validation_pd[['prediction_ann','ndvi_prediction']] = ndvi_transformacion.inverse_transform(validation_pd[['precipitation_ann_t','prediction']])
    validation_pd[['prediction_ann','ndvi_media']] = ndvi_transformacion.inverse_transform(validation_pd[['precipitation_ann_t',y_output]])

    validation_pd = validation_pd.reset_index(drop=False)[['id_point', 'periodo','type','ndvi_prediction','ndvi_media']]

    # Validacion entrenamiento
    validation_metrics = metrics(observado=validation_pd.ndvi_media,
                            prediccion=validation_pd.ndvi_prediction)

    # Sección Test


    data_exogena = pd_model_id[-prediction_order:][[exogena]]
    data_exogena[y_output] = np.nan
    data_predict = pd_model_id[pd_model_id.index < data_exogena.index.min()][[y_output,exogena]]

    pd_test = predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order,exogena,y_output)

    pd_test = predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order,exogena,y_output)

    pd_test = pd_test.rename(columns={y_output:'prediction'})
    pd_test['type'] = 'test'

    pd_test[y_output] = pd_model_id[pd_model_id.index > trainind_pd.periodo.max()][y_output]

    pd_test['id_point'] = id_point

    pd_test[['prediction_ann','ndvi_prediction']] = ndvi_transformacion.inverse_transform(pd_test[['precipitation_ann_t','prediction']])
    pd_test[['prediction_ann','ndvi_media']] = ndvi_transformacion.inverse_transform(pd_test[['precipitation_ann_t',y_output]])

    pd_test = pd_test.reset_index(drop=False)[['id_point', 'periodo','type','ndvi_prediction','ndvi_media']]

    # Validacion entrenamiento
    test_metrics = metrics(observado=pd_test.ndvi_media,
                            prediccion=pd_test.ndvi_prediction)


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

    # Pronóstico
    data_predict = pd_model_id[[y_output,exogena]]

    data_exogena = pd_precipitacion[(pd_precipitacion.periodo > data_predict.index.max()) & (pd_precipitacion.id_point==id_point)][[exogena,'periodo']]
    data_exogena.index = pd.to_datetime(data_exogena.periodo)
    data_exogena[y_output] = np.nan
    data_exogena = data_exogena.sort_index()[[exogena,y_output]]

    pd_prediction = predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order, exogena, y_output)
    pd_prediction = pd_prediction.rename(columns={y_output:'prediction'})
    pd_prediction['type'] = 'prediction'
    pd_prediction['id_point'] = id_point


    pd_prediction[['prediction_ann','ndvi_prediction']] = ndvi_transformacion.inverse_transform(pd_prediction[['precipitation_ann_t','prediction']])
    pd_prediction['ndvi_media'] = np.nan

    pd_prediction = pd_prediction.reset_index(drop=False)[['id_point', 'periodo','type','ndvi_prediction','ndvi_media']]

    # Uniendo informacion
    pd_summary = pd.concat([trainind_pd[list(pd_prediction)], 
                            pd_test[list(pd_prediction)], 
                            validation_pd[list(pd_prediction)], 
                            pd_prediction[list(pd_prediction)]
                            ])

    
    model_confi = {"id_point":id_point,
            "n_neurons":n_neurons,
            "activation":activation,
            "prediction_order":prediction_order,
            "auto_order":auto_order,
            "exog_order":exog_order,
            "exog_delay":exog_delay,
            "metrics":dict_metrics
            }

    # Logica de guardado
    if os.listdir(f'{DIR}{experimento}') == []:

        # Modelo
        model.save(f'{DIR}{experimento}/model.h5')

        # Pesos
        model.save_weights(f'{DIR}{experimento}/weights.h5')

        # History
        with open(f'{DIR}{experimento}/history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # confi
        with open(f'{DIR}{experimento}/model_confi.pkl', 'wb') as file_pi:
            pickle.dump(model_confi, file_pi)
        
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

            # confi
            with open(f'{DIR}{experimento}/model_confi.pkl', 'wb') as file_pi:
                pickle.dump(model_confi, file_pi)
            
            # guardando resultados
            pd_summary.to_pickle(f'{DIR}{experimento}/predicciones.pkl')


    experi = f'{DIR}{experimento}/{id_point}_{len(n_neurons)}_{activation[0]}_{prediction_order}_{auto_order}_{exog_order}_{exog_delay}'
    experimento_pd.to_csv(f'{experi}_summary.csv',index=False)