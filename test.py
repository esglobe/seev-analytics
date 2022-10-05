# %%
# PROYECTO: SISTEMA PARA EL SEGUIMIENTO DE ECOSISTEMAS VENEZOLANOS
# AUTOR: Javier Martinez

# Proceso de experimentos modelos NARX
import os
import sys

print('> Directorio actual: ', os.getcwd())  

from utils.MONGO import CONEXION
from utils.UTILS import *
from datetime import datetime
import pandas as pd

from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

# Creando la conexion con MongoDB
db = CONEXION.conexion()
db.list_collection_names()

print('cargando mongo db')

# Parque
park = sys.argv[1]

id_point = int(sys.argv[2])
y_output = sys.argv[3]
exogena = sys.argv[4]

prediction_order = int(sys.argv[5])# rango de prediccion
auto_order = int(sys.argv[6]) # componente autoregresiva
exog_order = int(sys.argv[7])# componente exogena qm
exog_delay = int(sys.argv[8])# componente exogena dm

activation = [sys.argv[9],sys.argv[9]]
kernel_initializer = 'lecun_normal'
bias_initializer = 'zeros'
patience = 15
epochs=500

print('park:', sys.argv[1])
print('id_point:', sys.argv[2])
print('y_output:', sys.argv[3])
print('exogena:', sys.argv[4])
print('prediction_order:', sys.argv[5])
print('auto_order:', sys.argv[6])
print('exog_order:', sys.argv[7])
print('exog_delay:', sys.argv[8])
print('activation:', sys.argv[9])

# Realizando consulta
meteorological = db.meteorological.find({"park":park, 'id_point':id_point})

# Generando pandas dataframe
data_pandas = pd.DataFrame([file for file in meteorological])
data_pandas['periodo'] = data_pandas.time.apply(lambda x: datetime.fromordinal(x))
data_pandas['mes_year'] =  data_pandas['periodo'].dt.strftime('%B-%Y')
data_pandas.index = pd.to_datetime(data_pandas.periodo)

DIR = f'./{park}/'

# # Estudio Precipitacion
pd_precipitacion = data_pandas[['id_point', 'latitud', 'longitud',
                                'precipitacion_mm']]

# Cargando data SST
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

# Transformacion
transformacion = LogMinimax.create( pd_precipitacion.precipitacion_mm.to_numpy() )
pd_model['precip_t'] = transformacion.transformacion()

# Modelo segun ID point
pd_model_id = pd_model[pd_model.id_point==id_point]
pd_model_id.index = pd.to_datetime(pd_model_id.periodo)
pd_model_id = pd_model_id[[y_output,exogena]]

# Definiendo estructura de datos
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

x_data, y_data = split_data(pd_model_id,exog_order,auto_order,exog_delay,prediction_order)

# Entrenamiento y validación
x_train = x_data[:-prediction_order]
x_vasl = x_data[-prediction_order:]

y_train = y_data[:-prediction_order]
y_vasl = y_data[-prediction_order:]

# Modelo NARX
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Metricas
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

# Neuronas
n_neurons = [int(2*x_train.shape[-1]/3),int(x_train.shape[-1])/3,]

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

# Hidden Leyers
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

# Out
model.add(keras.layers.Dense(   units=1,
                                activation='linear',
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
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

# Directorio experimento
experimento = f'experiments/narx/narx_prec_{id_point}_{len(n_neurons)}_{activation[0]}_{prediction_order}_{auto_order}_{exog_order}_{exog_delay}'

try:
    os.mkdir(f'{DIR}/{experimento}')
except:
    pass

# Entrenamiento
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=epochs,
                    batch_size=1,
                    verbose=0,
                    workers=2,
                    callbacks=[callback])

print(f'Total epocas:{len(history.epoch)}')

# Guardando experimento
model.save(f'{DIR}/{experimento}/narx_precipitacion_model.h5')
# Save Pesos
model.save_weights(f'{DIR}/{experimento}/narx_precipitacionweights.h5')

# Save History
import pickle
with open(f'{DIR}/{experimento}/narx_precipitacion_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Evaluacion entrenamiento
min_index = max([exog_order,auto_order])

# make predictions
trainPredict = model.predict(x_train, verbose=0).reshape(-1)
testPredict = model.predict(x_vasl, verbose=0).reshape(-1)

# Data de test
trainind_pd = pd.DataFrame(trainPredict,
                            index = pd_model_id[min_index:(min_index+trainPredict.shape[0])].index,
                            columns=['prediction']
                            )
                            
trainind_pd[y_output] = y_train.reshape(-1)
trainind_pd['type'] = 'training'

trainind_pd['precipitacion_mm'] = trainind_pd[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
trainind_pd['prediction_precipitacion_mm'] = trainind_pd['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )


trainind_pd = pd.merge(trainind_pd,pd_model_id[['oni']].reset_index(drop=False),
                        on=['periodo'],
                        how='left')

trainind_pd.index = pd.to_datetime(trainind_pd.periodo)

# Validacion
trainig_metrics = metrics(observado=trainind_pd.precipitacion_mm,
                          prediccion=trainind_pd.prediction_precipitacion_mm)

# Data de Validacion
validation_pd = pd.DataFrame(testPredict,
                            index = pd_model_id[(min_index+trainPredict.shape[0]):(min_index+trainPredict.shape[0]+prediction_order)].index,
                            columns=['prediction']
                            )

validation_pd[y_output] = y_vasl.reshape(-1)
validation_pd['type'] = 'validation'

validation_pd['precipitacion_mm'] = validation_pd[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
validation_pd['prediction_precipitacion_mm'] = validation_pd['prediction'].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

validation_pd = pd.merge(validation_pd,pd_model_id[['oni']].reset_index(drop=False),
                        on=['periodo'],
                        how='left')

# Validacion
validation_metrics = metrics(observado=validation_pd.precipitacion_mm,
                          prediccion=validation_pd.prediction_precipitacion_mm)

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

data_predict = pd_model_id[pd_model_id.index<=trainind_pd.periodo.max()][[y_output,exogena]]
data_exogena = pd_model_id[pd_model_id.index>trainind_pd.periodo.max()][[exogena]]

pd_test = predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order)

pd_test = pd_test[pd_test.type=='data_out']
pd_test['type'] = 'test'

pd_test['prediction'] = pd_test[y_output]
pd_test['precipitacion_mm'] = pd_model_id[pd_model_id.index>trainind_pd.periodo.max()][y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
pd_test['prediction_precipitacion_mm'] = pd_test[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )
pd_test['precip_t'] = pd_model_id[pd_model_id.index>trainind_pd.periodo.max()][y_output]

# Validacion entrenamiento
test_metrics = metrics(observado=pd_test.precipitacion_mm,
                        prediccion=pd_test.prediction_precipitacion_mm)

# Resultados del modelo
dict_metrics = {'epocas':[len(history.epoch)],
                'prediction_order':[prediction_order],
                'auto_order':[auto_order],
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

# Pronostico
data_predict = pd_model_id[[y_output,exogena]]
data_exogena = pd_sst_pron[pd_sst_pron.index>data_predict.index.max()][[exogena]]

pd_prediction = predict_one_stap_narx(model,data_predict,data_exogena,exog_order,auto_order,exog_delay,prediction_order)

pd_prediction = pd_prediction[pd_prediction.type=='data_out']
pd_prediction['type'] = 'prediction'

pd_prediction['precipitacion_mm'] = np.nan
pd_prediction['prediction_precipitacion_mm'] = pd_prediction[y_output].apply(lambda x: transformacion.inversa(x) if np.isnan(x)==False else np.nan )

pd_prediction['prediction'] = pd_prediction[y_output]
pd_prediction['precip_t'] = np.nan

# Uniendo informacion
pd_summary = pd.concat([trainind_pd, 
                        pd_test.reset_index(drop=False), 
                        pd_prediction.reset_index(drop=False)])
pd_summary.index = pd.to_datetime(pd_summary.periodo)

pd_summary.to_pickle(f'{DIR}/{experimento}/predicciones.pkl')