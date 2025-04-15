from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import csv
import time

def load_waypoints(csv_file):
    waypoints = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x, y, z = map(float, row)
            waypoints.append((x,y,z))
        
    return waypoints

def main():
    bot = InterbotixManipulatorXS("vx300s", "arm", "gripper")
    waypoints = load_waypoints("waypoints.csv")
    bot.arm.go_to_home_pose()

    for point in waypoints:
        x, y, z = point
        bot.arm.set_ee_pose_components(x = x, y = y, z = z)
        #time.sleep(0.2)
    
    bot.arm.go_to_sleep_pose()

if __name__ == '__main__':
    main()