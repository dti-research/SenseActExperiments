import time
import paramiko

if __name__ == "__main__":
    # Connect to the robot's dashboard interface and kill the URControl process
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())   # This script doesn't work for me unless this line is added!
    p.connect("192.168.1.100", username="root", password="easybot")
    print("HACK: Killing robot controller")
    stdin, stdout, stderr = p.exec_command("pkill URControl")
    opt = stdout.readlines()
    opt = "".join(opt)
    print(opt)
    time.sleep(2)