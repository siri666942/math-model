clc;
clear;
close all;

obj_func([pi, 120, 1.5, 3.6])

function y = obj_func(x)
    y = cal_t(x(1), x(2), x(3), x(4));
end



function total_effective_time = cal_t(FY1_theta, FY1_vel, release_time, detonation_delay)
    %% 参数定义
    % 导弹M1初始参数
    M1_pos0 = [20000, 0, 2000]; % 初始位置(m)
    M1_vel = 300;               % 速度大小(m/s)
    M1_dir = -M1_pos0/norm(M1_pos0); % 飞行方向(朝向原点)
    
    % 目标参数
    target_center = [0, 200, 0]; % 目标圆柱下底面圆心
    target_radius = 7;          % 圆柱半径(m)
    target_height = 10;         % 圆柱高度(m)
    
    % 无人机FY1参数
    FY1_pos0 = [17800, 0, 1800]; % 初始位置(m)
    FY1_dir = [cos(FY1_theta), sin(FY1_theta), 0]; % 飞行方向(水平朝向原点)
    
    % 烟幕干扰弹参数
    smoke_sink_speed = 3;       % 云团下沉速度(m/s)
    effective_radius = 10;      % 有效遮蔽半径(m)
    effective_duration = 20;    % 有效遮蔽持续时间(s)
    
    g = 9.8;                    % 重力加速度(m/s^2)
    
    %% 计算
    dt = 0.00001;                  % 时间步长(s)
    t_total = 30;               % 总模拟时间(s)
    total_effective_time = 0;   % 总有效遮蔽时长

    key_points_count_range = 16:16:1440;
    shielding_times = zeros(size(key_points_count_range));
    
    
    %% figure
    %{
    figure;
    axis([15000 20000 1500 2000]);
    hold on;
    M1 = plot(M1_pos0(1), M1_pos0(3), 'ro');
    FY1 = plot(FY1_pos0(1), FY1_pos0(3), 'bo');
    plot([M1_pos0(1), 0], [M1_pos0(3), 0], 'r-');
    plot([FY1_pos0(1), 0], [FY1_pos0(3), FY1_pos0(3)], 'b-');
    timeText = text(19500, 2010, '', 'FontSize', 12); % 添加文本对象显示时间
    flag = false;
    spf = 0.1;
    %}
    %%
    
    for i = 1:length(key_points_count_range)
        nvex=key_points_count_range(i);


        for t_current = 0:dt:t_total
            total_cur = total_effective_time;
            % 计算导弹位置
            M1_pos = M1_pos0 + M1_vel * M1_dir * t_current;

            % 计算无人机位置
            FY1_pos = FY1_pos0 + FY1_vel * FY1_dir * t_current;

            % 计算烟幕弹位置
            if t_current >= release_time
                if t_current <= release_time + detonation_delay
                    % 平抛运动阶段
                    delta_t = t_current - release_time;
                    vertical_dist = -0.5 * g * delta_t^2;
                    smoke_pos = [FY1_pos(1:2), FY1_pos(3) + vertical_dist];
                else
                    % 爆炸后云团下沉
                    smoke_pos = [smoke_pos(1:2), smoke_pos(3) - smoke_sink_speed * dt];

                    %% 判断遮蔽是否有效
                    if t_current <= release_time + detonation_delay + effective_duration
                        if norm(M1_pos - smoke_pos) < effective_radius
                            total_effective_time = total_effective_time + dt;
                        elseif norm(M1_pos - target_center) > norm(smoke_pos - target_center)
                            iseff = true;
                            central_axis = smoke_pos - M1_pos;
                            sin_alpha = effective_radius / norm(central_axis);
                            cos_alpha = sqrt(1 - sin_alpha ^ 2);
                            for theta = 1:nvex
                                tar_vex = target_center + [target_radius * cos(theta * pi / nvex), target_radius * sin(theta * pi / nvex), target_height * mod(theta, 2)];
                                tar_axis = tar_vex - M1_pos;
                                dot_product = dot(central_axis, tar_axis);
                                cos_gamma = dot_product / (norm(central_axis) * norm(tar_axis));
                                if cos_alpha > cos_gamma
                                    iseff = false;
                                    break;
                                end
                            end
                            if iseff
                                total_effective_time = total_effective_time + dt;
                            end
                        end
                    end
                end
            end
            %% figure
            %{
        if mod(t_current, spf) == 0
            if t_current >= release_time && t_current <= release_time + detonation_delay + effective_duration
                if ~flag
                    smoke = plot(FY1_pos0(1), FY1_pos0(3), 'go');
                    flag = true;
                end
                if total_cur == total_effective_time
                    set(smoke, 'XData', smoke_pos(1), 'YData', smoke_pos(3), 'Color', 'g');
                else
                    set(smoke, 'XData', smoke_pos(1), 'YData', smoke_pos(3), 'Color', 'm');
                end
            end
            set(M1, 'XData', M1_pos(1), 'YData', M1_pos(3));
            set(FY1, 'XData', FY1_pos(1), 'YData', FY1_pos(3));
            set(timeText, 'String', sprintf('%.2f s', t_current));
            drawnow;
        end
            %}
            %%
        end
        shielding_times(i)=total_effective_time;
        total_effective_time=0;
    end

    figure('Name', '目标离散化粒度敏感性分析', 'NumberTitle', 'off');
    plot(key_points_count_range, shielding_times, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    xlabel('用于表示真目标的关键点总数', 'FontSize', 12);
    ylabel('有效遮蔽时长 (s)', 'FontSize', 12);
    title('有效遮蔽时长对目标表示精度的敏感性分析', 'FontSize', 14);
    grid on;
    set(gca, 'FontSize', 11);
    xlim([min(key_points_count_range)-5, max(key_points_count_range)+5]);
    for k = 1:length(key_points_count_range)
    fprintf('关键点数: %d, 有效遮蔽时长: %.8f s\n', key_points_count_range(k), shielding_times(k));
    end
end