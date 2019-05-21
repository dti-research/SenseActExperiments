import time
import socket

if __name__ == "__main__":
    # Connect to the robot
    res = socket.getaddrinfo('192.168.1.100','29999', socket.AF_UNSPEC, socket.SOCK_STREAM)
    afam, socktype, proto, canonname, sock_addr = res[0]
    s = socket.socket(afam, socktype, proto)
    s.connect(sock_addr)

    # Power off the robot arm
    s.send("power off\n".encode('ascii'))
    print(s.recv(100))

    # Let the robot shutdown
    time.sleep(5)

    # Power the robot arn up
    s.send("power on\n".encode('ascii'))
    print(s.recv(100))

    time.sleep(10)

    s.send("brake release\n".encode('ascii'))
    print(s.recv(100))

    time.sleep(5)

    s.close()