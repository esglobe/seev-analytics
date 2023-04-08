import subprocess
import pandas as pd
from datetime import datetime


for point in [1,2,3,
            4,5,6,
            7,8,9]:

    parametros = {  'park':'terepaima',
                    'id_point':str(point),
                    'y_output':'precip_t',
                    'exogena':'oni',
                    'prediction_order':str(1*12),
                    'auto_order':str(25*12),
                    'exog_order':str(12),
                    'exog_delay':str(3),
                    'activation':'sigmoid'
                    }

    response = subprocess.run(["python3", "./1_narx_precipitacion.py",
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