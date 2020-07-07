imu = select(bag, 'time', [tStart tEnd], 'Topic', '/imu');
gps = select(bag, 'time', [tStart tEnd], 'Topic', '/gps');
speed = select(bag, 'time', [tStart tEnd], 'Topic', '/optical_speed_sensor');
lidar = select(bag, 'time', [tStart tEnd], 'Topic', '/velodyne_points');
wspeed = select(bag, 'time', [tStart tEnd], 'Topic', '/wheel_rpm');