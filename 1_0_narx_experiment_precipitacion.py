

import subprocess
import pandas as pd
from datetime import datetime

dates = pd.date_range('2012-01-01','2022-09-01',freq='M')
meses = list(map(lambda x: x.strftime('%Y-%m') , dates))


for point in [1,2,3,4,5,
             6,7,8,9,10,
             11,12,13,14,15]:

    parametros = {  'park':'cerro_saroche',
                    'id_point':str(point),
                    'y_output':'precip_t',
                    'exogena':'oni',
                    'prediction_order':str(12),
                    'auto_order':str(20*12),
                    'exog_order':str(12),
                    'exog_delay':str(0),
                    'activation':'tanh'
                    }

    response = subprocess.run(["python3", "./1_experimento_narx_precipitacion.py",
                                parametros.get('park'),
                                parametros.get('id_point'),
                                parametros.get('y_output'),
                                parametros.get('exogena'),
                                parametros.get('prediction_order'),
                                parametros.get('auto_order'),
                                parametros.get('exog_order'),
                                parametros.get('exog_delay'),
                                parametros.get('activation')
                                ])