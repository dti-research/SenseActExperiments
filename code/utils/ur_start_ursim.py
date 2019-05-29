import time
import paramiko
import argparse

# Setup argparser
parser = argparse.ArgumentParser(description='Starts the UR simulator via SSH',
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
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect(args.robot_ip, username=args.username, password="easybot")
    p.exec_command("export DISPLAY=:0; nohup bash ursim-current/start-ursim.sh UR5 &>/dev/null &")
    time.sleep(2)
