import os

print('> Directorio actual: ', os.getcwd())  

from utils.MONGO import CONEXION
from utils.UTILS import *
from datetime import datetime
import pandas as pd
import sys

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

#-----------------
def time_dormat(year = 2003, month = 1):
    """
    Funcion para calcular el periodo
    """

    if month < 10.:
        month = '0'+str(int(month))
    else:
        month = str(int(month))

    return str(int(year))+'-'+month+'-'+'01'
#-----------------


#---
if __name__ == "__main__":
    print('> Directorio actual: ', os.getcwd())

    # Datos de precipitación 
    park = sys.argv[1]#'cerro_saroche'
    test_size = float(sys.argv[2])#0.2
    random_state = int(sys.argv[3])#0
    f_activation = sys.argv[4]#'sigmoid'

    epochs = 1000
    patience = 20


    pd_precipitacion = pd.read_pickle(f'./{park}/data/narx_precipitacion.pkl')
    pd_precipitacion['precipitacion_narx'] = pd_precipitacion['prediction_precipitacion_mm']
    pd_precipitacion.head()

    pd_historical = pd_precipitacion[pd_precipitacion.type.isin(['training', #'test',
                                                                'test',
                                                                'prediction'])]
    # Preparando los datos
    variables = ['year',	'month',	'latitud',	'longitud', 'elevacion_media']

    trans_variable = MinMaxScaler()

    variables = ['year',	'month',	'latitud',	'longitud', 'elevacion_media']

    trans_variable.fit(pd_historical[variables])

    data_model = pd_historical[variables].copy()

    data_model[variables] = trans_variable.transform(pd_historical[variables]).astype(float)

    # Transformacion
    transformacion = LogMinimax.create( pd_historical['precipitacion_narx'].to_numpy() )

    data_model['precipitacion_narx'] = transformacion.transformacion()

    # ## Red ANN
    # Data entrenamiento y validación
    X_train, X_test, y_train, y_test = train_test_split(data_model[variables].to_numpy(), 
                                                        data_model['precipitacion_narx'].to_numpy(), 
                                                        test_size = test_size, 
                                                        random_state = random_state,
                                                        stratify=data_model[['latitud',	'longitud']])

    # Generando la red
    total = int(2*X_train.shape[1]/3)
    n_neuronas = [total-1, 1]

    activation = len(n_neuronas)*[f_activation]

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Metrícas
    mae = keras.metrics.MeanAbsoluteError()
    rmse = keras.metrics.RootMeanSquaredError()

    # ANN
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(units=n_neuronas[0], activation=activation[0], input_shape=X_train[0].shape))
    model.add(keras.layers.Dropout(0.1))

    if len(n_neuronas) > 1:
        for index in range(1,len(n_neuronas)):
        
            model.add(keras.layers.Dense(units=n_neuronas[index], activation=activation[index]))

    # Compilando modelo
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[mae,rmse]) 

    # Parada temprana
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
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=epochs,
                        batch_size=1,
                        verbose=0,
                        workers=2,
                        callbacks=[callback])

    print(f'Total epocas:{len(history.epoch)}')
    # Validación

    # make predictions
    trainPredict = model.predict(X_train, verbose=0).reshape(-1)
    testPredict = model.predict(X_test, verbose=0).reshape(-1)


    # Training
    trainind_pd = pd.DataFrame(y_train,
                                index = list(range(y_train.shape[0])) ,
                                columns=['precipitacion_narx']
                                )

    trainind_pd[variables] = X_train
    trainind_pd['prediction_ann'] = trainPredict

    trainind_pd[variables] = trans_variable.inverse_transform(trainind_pd[variables])
    trainind_pd['precipitacion_narx'] = trainind_pd['precipitacion_narx'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    trainind_pd['prediction_ann'] = trainind_pd['prediction_ann'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    trainind_pd[['year','month']] = trainind_pd[['year','month']].round(0)

    trainind_pd['periodo'] = trainind_pd[['year','month']].apply(lambda x: time_dormat(year = int(x.year), month = int(x.month) ),1)
    trainind_pd['periodo'] = pd.to_datetime(trainind_pd['periodo'] )

    trainind_pd['type'] = 'training'

    # Validacion entrenamiento
    training_metrics = metrics(observado=trainind_pd['precipitacion_narx'],
                            prediccion=trainind_pd['prediction_ann'])

    # Test
    test_pd = pd.DataFrame(y_test,
                            index = list(range(y_test.shape[0])) ,
                            columns=['precipitacion_narx']
                            )

    test_pd[variables] = X_test
    test_pd['prediction_ann'] = testPredict

    test_pd[variables] = trans_variable.inverse_transform(test_pd[variables])
    test_pd['precipitacion_narx'] = test_pd['precipitacion_narx'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
    test_pd['prediction_ann'] = test_pd['prediction_ann'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

    test_pd[['year','month']] = test_pd[['year','month']].round(0)
    test_pd['periodo'] = test_pd[['year','month']].apply(lambda x: time_dormat(year = x.year, month = x.month),1)
    test_pd['periodo'] = pd.to_datetime(test_pd['periodo'] )

    test_pd['type'] = 'test'

    # Validacion entrenamiento
    test_metrics = metrics(observado=test_pd['precipitacion_narx'],
                        prediccion=test_pd['prediction_ann'])

    # Resultados
    # Resultados del modelo
    dict_metrics = {'epocas':[len(history.epoch)],
                    'activation':str(activation),
                    'n_neurons':str(n_neuronas),
                    'capas':[len(n_neuronas)],
                    'training_mse':[training_metrics["mse"]],
                    'training_rmse':[training_metrics["rmse"]],
                    'training_mae':[training_metrics["mae"]],
                    'trainig_mape':[training_metrics['mape']],
                    'trainig_r':[training_metrics['r2']],
                    'test_mse':[test_metrics["mse"]],
                    'test_rmse':[test_metrics["rmse"]],
                    'test_mae':[test_metrics["mae"]],
                    'test_mape':[test_metrics['mape']],
                    'test_r':[test_metrics['r2']]
                    }
                    
    experimento_pd = pd.DataFrame.from_dict(dict_metrics)

    model_confi = {
                "n_neurons":n_neuronas,
                "activation":activation,
                "metrics":dict_metrics
                }

    pd_trainig_test = pd.concat([trainind_pd,test_pd])

    for_join = pd_historical\
                    .groupby(['latitud',	'longitud', 'id_point'],as_index=False)\
                    .count()[['latitud',	'longitud', 'id_point']]


    pd_final = pd.merge(pd_trainig_test,for_join,on=['latitud',	'longitud'],how = 'left').sort_values(['periodo','id_point'])

    pd_summary = pd.merge(pd_historical,
                            pd_final[['id_point','periodo','prediction_ann']],
                            on = ['id_point','periodo'],
                            how = 'left')

    DIR = f'./{park}/experiments/ann'

    import pickle

    if os.listdir(DIR) == []:

        # Modelo
        model.save(f'{DIR}/model.h5')

        # Pesos
        model.save_weights(f'{DIR}/weights.h5')

        # History
        with open(f'{DIR}/history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # confi
        with open(f'{DIR}/model_confi.pkl', 'wb') as file_pi:
            pickle.dump(model_confi, file_pi)
        
        # guardando resultados
        pd_summary.to_pickle(f'{DIR}/predicciones.pkl')

    else:
        files = [x for x in os.listdir(f'{DIR}') if x.find('summary')!=-1 ]
        total_summary = pd.concat([pd.read_csv(f'{DIR}/{file}') for file in files])
        print( f"Actual: {training_metrics['r2']}; Best Model: {total_summary.trainig_r.max()}" )

        if training_metrics['r2'] > total_summary.trainig_r.max(): 

            # Modelo
            model.save(f'{DIR}/model.h5')

            # Pesos
            model.save_weights(f'{DIR}/weights.h5')

            # History
            with open(f'{DIR}/history.pkl', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            
            # confi
            with open(f'{DIR}/model_confi.pkl', 'wb') as file_pi:
                pickle.dump(model_confi, file_pi)
            
            # guardando resultados
            pd_summary.to_pickle(f'{DIR}/predicciones.pkl')


    experi = f'{DIR}/{len(history.epoch)}_{sum(n_neuronas)}_{str(activation)}'
    experimento_pd.to_csv(f'{experi}_summary.csv',index=False)