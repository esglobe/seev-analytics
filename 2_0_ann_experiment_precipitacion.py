import subprocess
import pandas as pd
from datetime import datetime


parametros = [{  'park':'cerro_saroche',
                    'test_size':str(0.2),
                    'random_state':str(0),
                    'activation':'sigmoid',
                    },
                    {  'park':'cerro_saroche',
                    'test_size':str(0.2),
                    'random_state':str(0),
                    'activation':'softplus',
                    }]

for parametro in parametros:

    response = subprocess.run(["python3", "./2_ann_precipitacion.py",
                                parametro.get('park'),
                                parametro.get('test_size'),
                                parametro.get('random_state'),
                                parametro.get('activation')
                                ])