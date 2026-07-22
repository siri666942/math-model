function y = Obj_fun(x)
    y = cal_t(x(1), x(2), x(3), x(4));
end

function t = cal_t(FY1_theta, FY1_vel, release_time, detonation_delay)
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
    FY1_pos0 = [6000, -3000, 700]; % 初始位置(m)
    FY1_dir = [cos(FY1_theta), sin(FY1_theta), 0]; % 飞行方向(水平朝向原点)
    
    % 烟幕干扰弹参数
    smoke_sink_speed = 3;       % 云团下沉速度(m/s)
    effective_radius = 10;      % 有效遮蔽半径(m)
    effective_duration = 20;    % 有效遮蔽持续时间(s)
    
    g = 9.8;                    % 重力加速度(m/s^2)
    
    %% 计算
    dt = 0.005;                  % 时间步长(s)
    t_total = 30;               % 总模拟时间(s)
    total_effective_time = 0;   % 总有效遮蔽时长
    smoke_pos = zeros(1, 3);
    for t_current = 0:dt:t_total
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
                        for theta = 1:360
                            tar_vex = target_center + [target_radius * cos(theta * pi / 180), target_radius * sin(theta * pi / 180), target_height * mod(theta, 2)];
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
    end
    t = -total_effective_time;
end