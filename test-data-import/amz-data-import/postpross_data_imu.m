close all;
figure('Name','IMU Data');

%% Angular velocity
subplot(3,3,1) 
plot(wXsens)
xlabel('Time [s]')
ylabel('Angular Velocity [rad/s]')
title('wXsens')
legend('X axis')

subplot(3,3,2) 
plot(wYsens)
xlabel('Time [s]')
ylabel('Angular Velocity [rad/s]')
title('wYsens')
legend('Y axis')

subplot(3,3,3) 
plot(wZsens)
xlabel('Time [s]')
ylabel('Angular Velocity [rad/s]')
title('wZsens')
legend('Z axis')

%% Linear acceleration
subplot(3,3,4) 
plot(a_x)
xlabel('Time [s]')
ylabel('Linear Acceleration [m/s^2]')
title('a_x')
legend('X axis')

subplot(3,3,5) 
plot(a_y)
xlabel('Time [s]')
ylabel('Linear Acceleration [m/s^2]')
title('a_y')
legend('Y axis')

subplot(3,3,6) 
plot(a_z)
xlabel('Time [s]')
ylabel('Linear Acceleration [m/s^2]')
title('a_z')
legend('Z axis')

%% Orientation
subplot(3,3,7) 
plot(ori_x)
xlabel('Time [s]')
ylabel('Orientation')
title('ori_x')
legend('X axis')

subplot(3,3,8) 
plot(ori_y)
xlabel('Time [s]')
ylabel('Orientation')
title('ori_y')
legend('Y axis')

subplot(3,3,9) 
plot(ori_z)
xlabel('Time [s]')
ylabel('Orientation')
title('ori_z')
legend('Z axis')