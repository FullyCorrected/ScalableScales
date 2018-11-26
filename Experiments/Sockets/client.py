import socket
from time import sleep

def client():
  host = socket.gethostname()  # get local machine name
  port = 8080  # Make sure it's within the > 1024 $$ <65535 range
  
  s = socket.socket()
  s.connect((host, port))
  
  message = 'Brödrost'
  while True:
    s.send(message.encode('utf-8'))
    data = s.recv(1024).decode('utf-8')
    print('Received: ' + data)
    sleep(3)
    
  s.close()

if __name__ == '__main__':
  client()