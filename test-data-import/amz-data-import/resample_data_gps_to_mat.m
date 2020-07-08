clear;
clc;
close all;

%% load data
load('AMZ_GPS.mat');
load('AMZ_IMU.mat');
load('AMZ_speed.mat');
load('AMZ_wspeed.mat');

%% initialize times [sec]
gps_time = gps_time - gps_time(1);
imu_time = imu_time - imu_time(1);
speed_time = speed_time - speed_time(1);
wspeed_time = wspeed_time - wspeed_time(1);

%% fix gps data
lat = [0; lat];
long = [0; long];

%% find sample times [sec] and frequencies [Hz]
gps_sample_time = mean(diff(gps_time));
imu_sample_time = mean(diff(imu_time));
speed_sample_time = mean(diff(speed_time));
wspeed_sample_time = mean(diff(wspeed_time));

gps_sample_freq = round(1/gps_sample_time);
imu_sample_freq = round(1/imu_sample_time);
speed_sample_freq = round(1/speed_sample_time);
wspeed_sample_freq = round(1/wspeed_sample_time);

%% resample to gps frequency [Hz]
a_x = resample(a_x.Data(:,1), gps_sample_freq, imu_sample_freq);
a_y = resample(a_y.Data(:,1), gps_sample_freq, imu_sample_freq);
a_z = resample(a_z.Data(:,1), gps_sample_freq, imu_sample_freq);

wXsens = resample(wXsens.Data(:,1), gps_sample_freq, imu_sample_freq);
wYsens = resample(wYsens.Data(:,1), gps_sample_freq, imu_sample_freq);
wZsens = resample(wZsens.Data(:,1), gps_sample_freq, imu_sample_freq);

lin_v_x = resample(lin_v_x.Data(:,1), gps_sample_freq, speed_sample_freq);
lin_v_y = resample(lin_v_y.Data(:,1), gps_sample_freq, speed_sample_freq);

n_FL = resample(n_FL.Data(:,1), gps_sample_freq, wspeed_sample_freq);
n_FR = resample(n_FR.Data(:,1), gps_sample_freq, wspeed_sample_freq);
n_RL = resample(n_RL.Data(:,1), gps_sample_freq, wspeed_sample_freq);
n_RR = resample(n_RR.Data(:,1), gps_sample_freq, wspeed_sample_freq);

%% save to mat
% save('AMZ_data_resample_gps.mat', 'lat', 'long', 'a_x', 'a_y', 'a_z', 'lin_v_x', 'lin_v_y', 'wXsens', 'wYsens', 'wZsens', 'n_FL', 'n_FR', 'n_RL', 'n_RR');