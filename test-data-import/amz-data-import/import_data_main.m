clear;
clc;
close all;

%% Bag import
bag = rosbag('amz.bag');

%% Time
tStart = bag.StartTime;
tEnd = bag.EndTime;
time = tEnd - tStart

%% Message import
import_msg_all;

%% Data import
% import_data_imu;
% import_data_gps;
% import_data_speed;
% import_data_lidar;
% import_data_wspeed;

%% Postprocess
% postpross_data_imu;
% postpross_data_gps;
% postpross_data_speed;
% postpross_data_lidar;
% postpross_data_wspeed;