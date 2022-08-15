import yaml 

# Definiendo variables
with open('./config.yml') as stream:
    config = yaml.safe_load(stream)


# Objeto para la conexión
class CONEXION:
    """
    Calse para la conexión a mongo DB
    """
    username = config['MONGO_USER']
    password = config['MONGO_PASSWORD']
    cluster = config['MONGO_CLUSTER']

    @classmethod
    def conexion(cls):
      
      import pymongo

      conn_str = f"mongodb+srv://{CONEXION.username}:{CONEXION.password}@{CONEXION.cluster}.wsg1gnp.mongodb.net/?retryWrites=true&w=majority"
      cliente = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=5000)

      return cliente['SSEV']

