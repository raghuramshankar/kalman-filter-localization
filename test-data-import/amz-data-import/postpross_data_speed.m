close all;
figure('Name','Speed');

%% Angular velocity
subplot(2,3,1) 
plot(ang_v_x)
xlabel('Time [s]')
ylabel('Angular Velocity [rad/s]')
title('ang_v_x')
legend('X axis')

subplot(2,3,2) 
plot(ang_v_y)
xlabel('Time [s]')
ylabel('Angular Velocity [rad/s]')
title('ang_v_y')
legend('Y axis')

subplot(2,3,3) 
plot(ang_v_z)
xlabel('Time [s]')
ylabel('Angular Velocity [rad/s]')
title('ang_v_z')
legend('Z axis')

%% Linear velocity
subplot(2,3,4) 
plot(lin_v_x)
xlabel('Time [s]')
ylabel('Linear Velocity [m/s]')
title('lin_v_x')
legend('X axis')

subplot(2,3,5) 
plot(lin_v_y)
xlabel('Time [s]')
ylabel('Linear Velocity [m/s]')
title('lin_v_y')
legend('Y axis')

subplot(2,3,6) 
plot(lin_v_z)
xlabel('Time [s]')
ylabel('Linear Velocity [m/s]')
title('lin_v_z')
legend('Z axis')