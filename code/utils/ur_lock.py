import time
import socket

if __name__ == "__main__":
    # Connect to the robot's dashboard interface and save the log
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.1.100',29999))

    s.send("setUserRole locked\n".encode('ascii'))
    s.close()
    time.sleep(2)
