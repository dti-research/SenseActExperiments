import time
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
    # Connect to the robot's dashboard interface and kill the URControl process
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())   # This script doesn't work for me unless this line is added!
    p.connect(args.robot_ip, username=args.username, password="easybot")
    print("HACK: Killing robot controller")
    stdin, stdout, stderr = p.exec_command("pkill URControl")
    opt = stdout.readlines()
    opt = "".join(opt)
    print(opt)
    time.sleep(2)