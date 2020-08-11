clear;
clc;
close all;

%% save to csv
d = load('AMZ_data_resample_gps.mat');
headers = {'lat', 'long', 'a_x', 'a_y', 'a_z', 'wXsens', 'wYsens', 'wZsens', 'lin_v_x', 'lin_v_y', 'n_FL', 'n_FR', 'n_RL', 'n_RR'};
data = [d.lat, d.long, d.a_x, d.a_y, d.a_z, d.wXsens, d.wYsens, d.wZsens, d.lin_v_x, d.lin_v_y, d.n_FL, d.n_FR, d.n_RL, d.n_RR];
csvwrite_with_headers('amz_resample_gps.csv', data, headers);