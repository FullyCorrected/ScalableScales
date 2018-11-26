import socket

def server():
  host = socket.gethostname()   # get local machine name
  port = 8080  # Make sure it's within the > 1024 $$ <65535 range
  
  s = socket.socket()
  s.bind((host, port))
  
  s.listen(1)
  c, addr = s.accept()
  print("Connection from: " + str(addr))
  while True:
    recieved_data = c.recv(1024).decode('utf-8')
    if not recieved_data:
      break
  
    print('Recieved: ' + recieved_data)
    if recieved_data.lower() == 'brödrost':
        data_to_send = 'Jag är trött på din jävla brödrost'
    else:
        data_to_send = 'Ge mig en brödrost istället'
        
    c.send(data_to_send.encode('utf-8'))
  c.close()

if __name__ == '__main__':
    server()