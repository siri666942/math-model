clc;
clear;
close all;

% 设置变量范围和维度
nvars = 12;
lb = [0, 0, 0, 70, 70, 70, 0, 1, 1, 0, 0, 0];
ub = [2 * pi, 2 * pi, 2 * pi, 140, 140, 140, 20, 20, 20, 20, 20, 20];

% 运行PSO
options = optimoptions('particleswarm','SwarmSize',10000,'HybridFcn',@fmincon,'MaxIterations',2000) 
[x, fval] = particleswarm(@obj_func, nvars, lb, ub,options);
disp(['最优解: ', num2str(x)]);
disp(['最优值: ', num2str(fval)]);

function y = obj_func(x)
    y = -cal_t(x(1:3), x(4:6), [x(7), sum(x(7:8)), sum(x(7:9))], x(10:12));
end

function total_effective_time = cal_t(FY_theta, FY_vel, release_time, detonation_delay)
    %% 参数定义
    % 导弹M1初始参数
    M1_pos0 = [20000, 0, 2000]; % 初始位置(m)
    M1_vel = 300;               % 速度大小(m/s)
    M1_dir = -M1_pos0/norm(M1_pos0); % 飞行方向(朝向原点)
    
    % 目标参数
    target_center = [0, 200, 0]; % 目标圆柱下底面圆心
    target_radius = 7;          % 圆柱半径(m)
    target_height = 10;         % 圆柱高度(m)
    
    % 无人机FY参数
    FY_pos0 = [
        17800, 0, 1800;
        12000, 1400, 1400;
        6000, -3000, 700
    ]; % 初始位置(m)
    FY_dir = [
        cos(FY_theta(1)), sin(FY_theta(1)), 0;
        cos(FY_theta(2)), sin(FY_theta(2)), 0;
        cos(FY_theta(3)), sin(FY_theta(3)), 0
    ]; % 飞行方向(水平朝向原点)
    
    % 烟幕干扰弹参数
    smoke_sink_speed = 3;       % 云团下沉速度(m/s)
    effective_radius = 10;      % 有效遮蔽半径(m)
    effective_duration = 20;    % 有效遮蔽持续时间(s)
    
    g = 9.8;                    % 重力加速度(m/s^2)
    
    %% 计算
    dt = 0.01;                  % 时间步长(s)
    t_total = 20;               % 总模拟时间(s)
    total_effective_time = 0;   % 总有效遮蔽时长

    smoke_pos = zeros(3, 3);
    for t_current = 0:dt:t_total
        % total_cur = total_effective_time;
        % 计算导弹位置
        M1_pos = M1_pos0 + M1_vel * M1_dir * t_current;
        
        % 计算无人机位置
        FY_pos = FY_pos0 + FY_dir .* FY_vel' * t_current;
        
        % 计算烟幕弹位置
        glo_iseff = false;
        for i = 1:3
            if t_current >= release_time(i)
                if t_current <= release_time(i) + detonation_delay(i)
                    % 平抛运动阶段
                    delta_t = t_current - release_time(i);
                    vertical_dist = -0.5 * g * delta_t^2;
                    smoke_pos(i, :) = [FY_pos(i, 1:2), FY_pos(i, 3) + vertical_dist];
                else
                    % 爆炸后云团下沉
                    smoke_pos(i, :) = [smoke_pos(i, 1:2), smoke_pos(i, 3) - smoke_sink_speed * dt];
                    
                    % 判断遮蔽是否有效
                    if t_current <= release_time(i) + detonation_delay(i) + effective_duration
                        if norm(M1_pos - smoke_pos(i, :)) < effective_radius
                            total_effective_time = total_effective_time + dt;
                        elseif norm(M1_pos - target_center) > norm(smoke_pos(i, :) - target_center)
                            iseff = true;
                            central_axis = smoke_pos(i, :) - M1_pos; 
                            sin_alpha = effective_radius / norm(central_axis);
                            cos_alpha = sqrt(1 - sin_alpha ^ 2);
                            nvex = 8;
                            for theta = 1:nvex
                                tar_vex = target_center + [target_radius * cos(theta * 2 * pi / nvex), target_radius * sin(theta * 2 * pi / nvex), target_height * mod(theta, 2)];
                                tar_axis = tar_vex - M1_pos;
                                dot_product = dot(central_axis, tar_axis);
                                cos_gamma = dot_product / (norm(central_axis) * norm(tar_axis));
                                if cos_alpha > cos_gamma
                                    iseff = false;
                                    break;	
                                end
                            end
                            if iseff
                                glo_iseff = true;
                            end
                        end
                    end
                end
            end
        end
        if glo_iseff
            total_effective_time = total_effective_time + dt;
        end
    end
end