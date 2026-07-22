clc;
clear;
close all;

pos0 = [12000, 1400, 1400];
theta = 4.5175; %%
dir = [cos(theta), sin(theta), 0];
vel = 122.8233; %%
release_time = 8.26619; %%
detonation_delay = 6.69533; %%
g = 9.8;

theta * 180 / pi
release_pos = pos0 + dir * vel * release_time
detonation_pos = release_pos + dir * vel * detonation_delay - [0, 0, 0.5 * g * detonation_delay^2]