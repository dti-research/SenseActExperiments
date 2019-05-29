import time
import socket
import paramiko
import argparse

# Setup argparser
parser = argparse.ArgumentParser(description='Kills the URControl process via SSH',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--robot-ip",
                    help="robot ip",
                    metavar="192.168.1.100",
                    required=True)
parser.add_argument('--username', dest='username',
                    default="root",
                    type=str,
                    help='name of the user at the robot')
args = parser.parse_args()

if __name__ == "__main__":
    # Connect to the robot's dashboard interface and save the log
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.robot_ip,29999))

    s.send("robotmode\n".encode('ascii'))
    time.sleep(.5)
    response = s.recv(100)

    s.close()

    if "RUNNING" not in str(response):
        print("Some error occurred. UR Robotmode: {}".format(response))
        print("UR's Polyscope prob. lost connection to its controller. Rebooting robot...")
    else:
        exit(0)
    
    # Reboot robot controller
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())   # This script doesn't work for me unless this line is added!
    p.connect(args.robot_ip, username=args.username, password="easybot")
    #stdin, stdout, stderr = p.exec_command("/sbin/reboot")
    print("Rebooting PolyScope (pkill java)!")
    stdin, stdout, stderr = p.exec_command("pkill java")
    opt = stdout.readlines()
    opt = "".join(opt)
    print(opt)

    # Wait for the robot to come online
    for x in range(45):
        print('Reconnecting in {0}s   '.format(45-x), end="\r")
        time.sleep(1)
    print('')

    # Connect to robot's dashboard interface again
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.robot_ip,29999))

    # Close the "Go To Initialize Screen" popup
    s.send("close popup\n".encode('ascii'))
    s.recv(100)

    time.sleep(0.5)

    # Power on robot arm
    s.send("power on\n".encode('ascii'))
    s.recv(100)

    time.sleep(5)

    # Brake release
    s.send("brake release\n".encode('ascii'))
    s.recv(100)

    time.sleep(5)

    s.send("robotmode\n".encode('ascii'))
    time.sleep(.5)
    response = s.recv(100)

    s.close()
    time.sleep(2)

    if "RUNNING" not in str(response):
        print("Some unknown error occurred after rebooting the robot... Need human assistance!")
        exit(1)
    else:
        exit(0)
