% 圆柱参数
center = [0, 200, 0];  % 下底面圆心坐标 (x, y, z)
radius = 7;            % 半径 (m)
height = 10;           % 高度 (m)

% 创建圆柱网格数据
[x, y, z] = cylinder(radius, 50);  % 50个分段

% 调整圆柱高度和位置
z = z * height;        % 缩放z坐标到指定高度
x = x + center(1);     % 平移x坐标
y = y + center(2);     % 平移y坐标
z = z + center(3);     % 平移z坐标

% 绘制圆柱
figure('Color', 'white', 'Position', [100, 100, 1200, 800])
surf(x, y, z, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', [0.7 0.7 0.9])
hold on

% 绘制上下底面
fill3(x(1,:), y(1,:), z(1,:), 'b', 'FaceAlpha', 0.3)
fill3(x(2,:), y(2,:), z(2,:), 'b', 'FaceAlpha', 0.3)

% 导弹位置
missiles = [
    20000, 0, 2000;      % M1
    19000, 600, 2100;    % M2
    18000, -600, 1900    % M3
];

% 无人机位置
drones = [
    17800, 0, 1800;      % FY1
    12000, 1400, 1400;   % FY2
    6000, -3000, 700;    % FY3
    11000, 2000, 1800;   % FY4
    13000, -2000, 1300   % FY5
];

% 绘制导弹（红色大点）
missile_plot = plot3(missiles(:,1), missiles(:,2), missiles(:,3), ...
    'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 2);

% 绘制无人机（蓝色小点）
drone_plot = plot3(drones(:,1), drones(:,2), drones(:,3), ...
    'b^', 'MarkerSize', 10, 'MarkerFaceColor', 'c', 'LineWidth', 2);

% 添加标签
text(missiles(:,1)+500, missiles(:,2), missiles(:,3), {'M1', 'M2', 'M3'}, ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', 'r');

text(drones(:,1)+500, drones(:,2), drones(:,3), {'FY1', 'FY2', 'FY3', 'FY4', 'FY5'}, ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', 'b');

% 设置图形属性
axis equal
grid on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
title('圆柱与导弹、无人机分布图', 'FontSize', 14)
view(3)  % 3D视角

% 添加参考点
center_plot = plot3(center(1), center(2), center(3), 'go', 'MarkerSize', 8, 'LineWidth', 2);

% 设置坐标轴范围以适应所有点
all_points = [missiles; drones; center];
padding = 1000;
xlim([min(all_points(:,1))-padding, max(all_points(:,1))+padding])
ylim([min(all_points(:,2))-padding, max(all_points(:,2))+padding])
zlim([min(all_points(:,3))-padding, max(all_points(:,3))+padding])

% 添加图例
legend([center_plot, missile_plot, drone_plot], ...
    {'圆柱圆心', '导弹', '无人机'}, 'Location', 'best')

% 添加比例尺参考
fprintf('注意：圆柱尺寸(半径7m,高10m)相对于坐标尺度非常小，可能难以直接看到。\n');
fprintf('导弹和无人机坐标范围：X[%.0f,%.0f], Y[%.0f,%.0f], Z[%.0f,%.0f]\n', ...
    min(all_points(:,1)), max(all_points(:,1)), ...
    min(all_points(:,2)), max(all_points(:,2)), ...
    min(all_points(:,3)), max(all_points(:,3)));