import time
import socket

import argparse

# Setup argparser
parser = argparse.ArgumentParser(description='Kills the URControl process via SSH',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--robot-ip",
                    help="robot ip",
                    metavar="192.168.1.100",
                    required=True)
args = parser.parse_args()

if __name__ == "__main__":
    # Connect to the robot's dashboard interface and save the log
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.robot_ip,29999))

    s.send("setUserRole locked\n".encode('ascii'))
    s.close()
    time.sleep(2)
