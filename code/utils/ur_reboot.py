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
parser.add_argument('--sim', dest='sim',
                    default=False,
                    type=bool,
                    help='Using UR simulator?')
parser.add_argument('--username', dest='username',
                    default="root",
                    type=str,
                    help='name of the user at the robot')
args = parser.parse_args()

if __name__ == "__main__":
    # Connect to the robot's dashboard interface (port 29999)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.robot_ip,29999))

    # Get state of the robot
    s.send("robotmode\n".encode('ascii'))
    time.sleep(.5)
    response = s.recv(100)

    # Close TCP/IP connection
    s.close()

    if "RUNNING" not in str(response):
        print("Some error occurred. UR Robotmode: {}".format(response))
        print("UR's Polyscope prob. lost connection to its controller. Rebooting robot...")
    else:
        exit(0)
    
    # Reboot UR's PolyScope via SSH
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect(args.robot_ip, username=args.username, password="easybot")
    print("Rebooting PolyScope (pkill java)!")
    p.exec_command("pkill java")

    if args.sim:
        # UR simulator does not automatically reboot when killed. Manual rebooting:
        p.exec_command("export DISPLAY=:0; nohup bash ursim-current/start-ursim.sh UR5 &>/dev/null &")
        time.sleep(2)
    
    p.close()

    # Wait for the robot to come online
    # - Robot controller: 100s
    # - PolyScope: 45s
    if args.sim:
        wait = 15
    else:
        wait = 45

    for x in range(wait):
        print('Reconnecting in {0}s   '.format(wait-x), end="\r")
        time.sleep(1)
    print('')
    
    if not args.sim:
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

        # Get the robot state
        s.send("robotmode\n".encode('ascii'))
        response = s.recv(100)
        time.sleep(2)

        # Close TCP/IP connection
        s.close()

        if "RUNNING" not in str(response):
            print("Some unknown error occurred after rebooting the robot... Need human assistance!")
            exit(1)
        else:
            print("Rebooted succesfully")
            exit(0)
    else:
        print("Rebooted URSim succesfully")
        exit(0)
