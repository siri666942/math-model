clc;
clear;
close all;

obj_func([5.183279           133.4141             6.293838      1.000364      5.525586     1.829352       4.41929       2.67778])

function y = obj_func(x)
    y = -cal_t(x(1), x(2), [x(3), sum(x(3:4)), sum(x(3:5))], x(6:8));
end
%}

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
    FY_pos0 = [12000, 1400, 1400]; % 初始位置(m)
    FY_dir = [cos(FY_theta), sin(FY_theta), 0]; % 飞行方向
    
    % 烟幕干扰弹参数
    smoke_sink_speed = 3;       % 云团下沉速度(m/s)
    effective_radius = 10;      % 有效遮蔽半径(m)
    effective_duration = 20;    % 有效遮蔽持续时间(s)
    
    g = 9.8;                    % 重力加速度(m/s^2)
    
    %% 计算
    dt = 0.1;                  % 时间步长(s)
    t_total = 60;               % 总模拟时间(s)
    total_effective_time = 0;   % 总有效遮蔽时长

    smoke_pos = zeros(3, 3);

    %% figure
    %{}
    figure;
    axis([0 20000 -3000 2000 0 2100]);
    hold on;
    grid on;
    view(3); % 3D视角
    M1 = plot3(M_pos0(1, 1), M_pos0(1, 2), M_pos0(1, 3), 'ro');
    M2 = plot3(M_pos0(2, 1), M_pos0(2, 2), M_pos0(2, 3), 'ro');
    M3 = plot3(M_pos0(3, 1), M_pos0(3, 2), M_pos0(3, 3), 'ro');
    FY = plot3(FY_pos0(1), FY_pos0(2), FY_pos0(3), 'bo');
    line1 = plot3([M_pos0(1, 1), target_center(1)], [M_pos0(1, 2), target_center(2)], [M_pos0(1, 3), target_center(3)], 'r-');
    line2 = plot3([M_pos0(2, 1), target_center(1)], [M_pos0(2, 2), target_center(2)], [M_pos0(2, 3), target_center(3)], 'r-');
    line3 = plot3([M_pos0(3, 1), target_center(1)], [M_pos0(3, 2), target_center(2)], [M_pos0(3, 3), target_center(3)], 'r-');
    timeText = text(19500, 0, 2010, '', 'FontSize', 12); % 添加文本对象显示时间
    smoke = [];
    flag = zeros(1, 3);
    spf = 0.1;
    %}
    %%
    
    for t_current = 0:dt:t_total
        % total_cur = total_effective_time;
        % 计算导弹位置
        M_pos = M_pos0 + M_vel * M_dir * t_current;
        
        % 计算无人机位置
        FY_pos = FY_pos0 + FY_dir .* FY_vel' * t_current;
        
        % 计算烟幕弹位置
        iseff = zeros(1, 3);
        sort = [1, 1, 1];
        for i = 1:3
            if t_current < release_time(i)
                smoke_pos(i, :) = FY_pos;
            else
                if t_current <= release_time(i) + detonation_delay(i)
                    % 平抛运动阶段
                    delta_t = t_current - release_time(i);
                    vertical_dist = -0.5 * g * delta_t^2;
                    smoke_pos(i, :) = [FY_pos(1:2), FY_pos(3) + vertical_dist];
                else
                    % 爆炸后云团下沉
                    smoke_pos(i, 3) = smoke_pos(i, 3) - smoke_sink_speed * dt;
                    
                    % 判断遮蔽是否有效
                    k = sort(i);
                    if t_current <= release_time(i) + detonation_delay(i) + effective_duration
                        if norm(M_pos(k, :) - smoke_pos(i, :)) < effective_radius
                            iseff(i) = 1;
                        elseif norm(M_pos(k, :) - target_center) > norm(smoke_pos(i, :) - target_center)
                            iseff(i) = 1;
                            central_axis = smoke_pos(i, :) - M_pos(k, :); 
                            sin_alpha = effective_radius / norm(central_axis);
                            cos_alpha = sqrt(1 - sin_alpha ^ 2);
                            nvex = 8;
                            for theta = 1:nvex
                                tar_vex = target_center + [target_radius * cos(theta * 2 * pi / nvex), target_radius * sin(theta * 2 * pi / nvex), target_height * mod(theta, 2)];
                                tar_axis = tar_vex - M_pos(k, :);
                                dot_product = dot(central_axis, tar_axis);
                                cos_gamma = dot_product / (norm(central_axis) * norm(tar_axis));
                                if cos_alpha > cos_gamma
                                    iseff(i) = 0;
                                    break;
                                end
                            end
                        end
                    end
                end
            end
        end
        if sum(iseff(:)) == 1
            total_effective_time = total_effective_time + dt;
        end
        %% figure
        %{}
        if mod(t_current, spf) == 0
            for i = 1:3
                if t_current >= release_time(i) && t_current <= release_time(i) + detonation_delay(i) + effective_duration
                    if flag(i) == 0
                        smoke(i) = plot3(FY_pos0(1), FY_pos0(2), FY_pos0(3), 'go');
                        flag(i) = 1;
                    end
                    if t_current >= release_time(i) + detonation_delay(i) && iseff(i) == 1
                        set(smoke(i), 'XData', smoke_pos(i, 1), 'YData', smoke_pos(i, 2), 'ZData', smoke_pos(i, 3), 'Color', 'm');
                    else
                        set(smoke(i), 'XData', smoke_pos(i, 1), 'YData', smoke_pos(i, 2), 'ZData', smoke_pos(i, 3), 'Color', 'g');
                    end
                end
            end
            set(M1, 'XData', M_pos(1, 1), 'YData', M_pos(1, 2), 'ZData', M_pos(1, 3));
            set(M2, 'XData', M_pos(2, 1), 'YData', M_pos(2, 2), 'ZData', M_pos(2, 3));
            set(M3, 'XData', M_pos(3, 1), 'YData', M_pos(3, 2), 'ZData', M_pos(3, 3));
            set(FY, 'XData', FY_pos(1), 'YData', FY_pos(2), 'ZData', FY_pos(3));
            set(timeText, 'String', sprintf('%.2f s', t_current));
            set(line1, 'XData', [M_pos(1, 1), target_center(1)], 'YData', [M_pos(1, 2), target_center(2)], 'ZData', [M_pos(1, 3), target_center(3)]);
            set(line2, 'XData', [M_pos(2, 1), target_center(1)], 'YData', [M_pos(2, 2), target_center(2)], 'ZData', [M_pos(2, 3), target_center(3)]);
            set(line3, 'XData', [M_pos(3, 1), target_center(1)], 'YData', [M_pos(3, 2), target_center(2)], 'ZData', [M_pos(3, 3), target_center(3)]);
            drawnow;
            pause(0.1);
        end
        %}
    end
end