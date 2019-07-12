import sys
sys.path.append('API/model')
from app import create_app
server = create_app()
server.run()