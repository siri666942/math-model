clc;
clear;
close all;

obj_func( ...
    [3.138978, 5.136728, 1.599195, 4.715715, 2.082462, ...
    139.9796, 130.0138, 126.8076, 137.6076, 122.8233, ...
    ...
    0             2.408497      2.567866, ...
    6.138287      1.002069      5.473934, ...
    19.08704      1.114563      2.779414, ...
    0.7719681     1.794311      3.107894, ...
    13.25147      1.228234       6.67763, ...
    ...
    4.187053e-06  5.000162      5.887084, ...
    1.994981      4.601763        2.7422, ...
    3.735751      4.165216      2.876455, ...
    10.85204      11.72386      11.36387, ...
    1.760724      4.244666      1.002012])

function y = obj_func(x)
    y = -cal_t( ...
        x(1:5), ...
        x(6:10), ...
        [ ...
        x(11), sum(x(11:12)), sum(x(11:13)); ...
        x(14), sum(x(14:15)), sum(x(14:16)); ...
        x(17), sum(x(17:18)), sum(x(17:19)); ...
        x(20), sum(x(20:21)), sum(x(20:22)); ...
        x(23), sum(x(23:24)), sum(x(23:25)); ...
        ], ...
        [ ...
        x(26:28);
        x(29:31);
        x(32:34);
        x(35:37);
        x(38:40)
        ]);
end


function total_effective_time = cal_t(FY_theta, FY_vel, release_time, detonation_delay)
    %% 参数定义
    % 导弹M初始参数
    M_pos0 = [
        20000, 0, 2000;
        19000, 600, 2100;
        18000, -600, 1900
    ]; % 初始位置(m)
    M_vel = 300;               % 速度大小(m/s)
    M_dir = [
        -M_pos0(1, :)/norm(M_pos0(1, :));
        -M_pos0(2, :)/norm(M_pos0(2, :));
        -M_pos0(3, :)/norm(M_pos0(3, :));
    ]; % 飞行方向(朝向原点)
    
    % 目标参数
    target_center = [0, 200, 0]; % 目标圆柱下底面圆心
    target_radius = 7;          % 圆柱半径(m)
    target_height = 10;         % 圆柱高度(m)
    
    % 无人机FY参数
    FY_pos0 = [
        17800, 0, 1800;
        12000, 1400, 1400;
        6000, -3000, 700;
        11000, 2000, 1800;
        13000, -2000, 1300
    ]; % 初始位置(m)
    FY_dir = [
        cos(FY_theta(1)), sin(FY_theta(1)), 0;
        cos(FY_theta(2)), sin(FY_theta(2)), 0;
        cos(FY_theta(3)), sin(FY_theta(3)), 0;
        cos(FY_theta(4)), sin(FY_theta(4)), 0;
        cos(FY_theta(5)), sin(FY_theta(5)), 0
    ]; % 飞行方向(水平朝向原点)
    
    % 烟幕干扰弹参数
    smoke_sink_speed = 3;       % 云团下沉速度(m/s)
    effective_radius = 10;      % 有效遮蔽半径(m)
    effective_duration = 20;    % 有效遮蔽持续时间(s)
    
    g = 9.8;                    % 重力加速度(m/s^2)
    
    %% 计算
    dt = 0.01;                  % 时间步长(s)
    t_total = 40;               % 总模拟时间(s)
    total_effective_time = 0;   % 总有效遮蔽时长

    smoke_pos = zeros(5, 3, 3);

    %% figure
    %{
    figure;
    axis([0 20000 -3000 2000 0 2100]);
    hold on;
    grid on;
    view(3); % 3D视角
    M1 = plot3(M_pos0(1, 1), M_pos0(1, 2), M_pos0(1, 3), 'ro');
    M2 = plot3(M_pos0(2, 1), M_pos0(2, 2), M_pos0(2, 3), 'ro');
    M3 = plot3(M_pos0(3, 1), M_pos0(3, 2), M_pos0(3, 3), 'ro');
    FY1 = plot3(FY_pos0(1, 1), FY_pos0(1, 2), FY_pos0(1, 3), 'bo');
    FY2 = plot3(FY_pos0(2, 1), FY_pos0(2, 2), FY_pos0(2, 3), 'bo');
    FY3 = plot3(FY_pos0(3, 1), FY_pos0(3, 2), FY_pos0(3, 3), 'bo');
    FY4 = plot3(FY_pos0(4, 1), FY_pos0(4, 2), FY_pos0(4, 3), 'bo');
    FY5 = plot3(FY_pos0(5, 1), FY_pos0(5, 2), FY_pos0(5, 3), 'bo');
    line1 = plot3([M_pos0(1, 1), target_center(1)], [M_pos0(1, 2), target_center(2)], [M_pos0(1, 3), target_center(3)], 'r-');
    line2 = plot3([M_pos0(2, 1), target_center(1)], [M_pos0(2, 2), target_center(2)], [M_pos0(2, 3), target_center(3)], 'r-');
    line3 = plot3([M_pos0(3, 1), target_center(1)], [M_pos0(3, 2), target_center(2)], [M_pos0(3, 3), target_center(3)], 'r-');
    timeText = text(19500, 0, 2010, '', 'FontSize', 12); % 添加文本对象显示时间
    smoke = [];
    flag = zeros(5, 3);
    spf = 0.1;
    %}
    %%
    t_eff = zeros(5, 3);
    bomb_h = zeros(5, 3);
    for t_current = 0:dt:t_total
        % total_cur = total_effective_time;
        % 计算导弹位置
        M_pos = M_pos0 + M_vel * M_dir * t_current;
        
        % 计算无人机位置
        FY_pos = FY_pos0 + FY_dir .* FY_vel' * t_current;
        
        % 计算烟幕弹位置
        iseff = zeros(5, 3, 3);
        
        for i = 1:5
            for j = 1:3
                if t_current < release_time(i, j)
                    smoke_pos(i, j, :) = FY_pos(i, :);
                else
                    if t_current <= release_time(i, j) + detonation_delay(i, j)
                        % 平抛运动阶段
                        delta_t = t_current - release_time(i, j);
                        vertical_dist = -0.5 * g * delta_t^2;
                        smoke_pos(i, j, :) = [FY_pos(i, 1:2), FY_pos(i, 3) + vertical_dist];
                    else
                        if bomb_h(i, j) == 0
                            bomb_h(i, j) = FY_pos(i, 3)  - 0.5 * g * detonation_delay(i, j)^2;
                            smoke_pos(i, j, 1:2) = FY_pos0(i, 1:2) + FY_dir(i, 1:2) * FY_vel(i) * (release_time(i, j) + detonation_delay(i, j));
                        end
                        % 爆炸后云团下沉
                        smoke_pos(i, j, 3) = bomb_h(i, j) - smoke_sink_speed * (t_current - release_time(i, j) - detonation_delay(i, j));
                        
                        % 判断遮蔽是否有效
                        for k = 1:3
                            if t_current <= release_time(i, j) + detonation_delay(i, j) + effective_duration
                                if norm(M_pos(k, :)' - squeeze(smoke_pos(i, j, :))) < effective_radius
                                    iseff(i, j, k) = 1;
                                elseif norm(M_pos(k, :) - target_center) > norm(squeeze(smoke_pos(i, j, :)) - target_center')
                                    iseff(i, j, k) = 1;
                                    central_axis = squeeze(smoke_pos(i, j, :))' - M_pos(k, :); 
                                    sin_alpha = effective_radius / norm(central_axis);
                                    cos_alpha = sqrt(1 - sin_alpha ^ 2);
                                    nvex = 8;
                                    for theta = 1:nvex
                                        tar_vex = target_center + [target_radius * cos(theta * 2 * pi / nvex), target_radius * sin(theta * 2 * pi / nvex), target_height * mod(theta, 2)];
                                        tar_axis = tar_vex - M_pos(k, :);
                                        dot_product = dot(central_axis, tar_axis);
                                        cos_gamma = dot_product / (norm(central_axis) * norm(tar_axis));
                                        if cos_alpha > cos_gamma
                                            iseff(i, j, k) = 0;
                                            break;	
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        for k = 1:3
            current_slice = iseff(:, :, k);
            if any(current_slice(:))
                total_effective_time = total_effective_time + dt;
                t_eff = t_eff + iseff(:, :, k) * dt;
            end
        end
        %% figure
        %{
        if mod(t_current, spf) == 0
            for i = 1:5
                for j = 1:3
                    if t_current >= release_time(i, j) && t_current <= release_time(i, j) + detonation_delay(i, j) + effective_duration
                        if flag(i, j) == 0
                            smoke(i, j) = plot3(FY_pos0(i, 1), FY_pos0(i, 2), FY_pos0(i, 3), 'go');
                            flag(i, j) = 1;
                        end
                        if t_current >= release_time(i, j) + detonation_delay(i, j) && sum(iseff(i, j, :)) == 1
                            set(smoke(i, j), 'XData', smoke_pos(i, j, 1), 'YData', smoke_pos(i, j, 2), 'ZData', smoke_pos(i, j, 3), 'Color', 'm');
                        else
                            set(smoke(i, j), 'XData', smoke_pos(i, j, 1), 'YData', smoke_pos(i, j, 2), 'ZData', smoke_pos(i, j, 3), 'Color', 'g');
                        end
                    end
                end
            end
            set(M1, 'XData', M_pos(1, 1), 'YData', M_pos(1, 2), 'ZData', M_pos(1, 3));
            set(M2, 'XData', M_pos(2, 1), 'YData', M_pos(2, 2), 'ZData', M_pos(2, 3));
            set(M3, 'XData', M_pos(3, 1), 'YData', M_pos(3, 2), 'ZData', M_pos(3, 3));
            set(FY1, 'XData', FY_pos(1, 1), 'YData', FY_pos(1, 2), 'ZData', FY_pos(1, 3));
            set(FY2, 'XData', FY_pos(2, 1), 'YData', FY_pos(2, 2), 'ZData', FY_pos(2, 3));
            set(FY3, 'XData', FY_pos(3, 1), 'YData', FY_pos(3, 2), 'ZData', FY_pos(3, 3));
            set(FY4, 'XData', FY_pos(4, 1), 'YData', FY_pos(4, 2), 'ZData', FY_pos(4, 3));
            set(FY5, 'XData', FY_pos(5, 1), 'YData', FY_pos(5, 2), 'ZData', FY_pos(5, 3));
            set(timeText, 'String', sprintf('%.2f s', t_current));
            set(line1, 'XData', [M_pos(1, 1), target_center(1)], 'YData', [M_pos(1, 2), target_center(2)], 'ZData', [M_pos(1, 3), target_center(3)]);
            set(line2, 'XData', [M_pos(2, 1), target_center(1)], 'YData', [M_pos(2, 2), target_center(2)], 'ZData', [M_pos(2, 3), target_center(3)]);
            set(line3, 'XData', [M_pos(3, 1), target_center(1)], 'YData', [M_pos(3, 2), target_center(2)], 'ZData', [M_pos(3, 3), target_center(3)]);
            drawnow;
            pause(0.1);
        end
        %}
        %%
    end
    t_eff
    sum(t_eff(:))
end