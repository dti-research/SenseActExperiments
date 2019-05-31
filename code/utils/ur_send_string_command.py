import time
import socket

import argparse

# Setup argparser
parser = argparse.ArgumentParser(description='Sends a string (URScript) command to the UR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ip", "--robot-ip",
                    dest="robot_ip",
                    help="robot ip",
                    metavar="192.168.1.100",
                    required=True)
parser.add_argument("-p", "--robot-port",
                    dest="robot_port",
                    help="robot port",
                    metavar="29999",
                    required=True)
parser.add_argument("-c", "--command",
                    dest="command",
                    type=str,
                    required=True)
args = parser.parse_args()

if __name__ == "__main__":
    # Connect to the robot's dashboard interface and save the log
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.robot_ip, int(args.robot_port)))
    print(s.recv(100))

    s.send((args.command+"\n").encode('ascii'))
    time.sleep(0.1)
    print(s.recv(100))
    s.close()
