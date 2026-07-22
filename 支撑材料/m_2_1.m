% % 定义参数范围
% z = linspace(0, 20000, 100);
% y = linspace(0, 400, 100);
% [Z, Y] = meshgrid(z, y);
% 
% % 计算x的值
% X = 10 * Z;
% 
% % 绘制平面
% surf(X, Y, Z);
% xlabel('x');
% ylabel('y');
% zlabel('z');
% title('过点A(0,207,0)、B(0,0,0)和C(20000,0,2000)的平面');
% 
% 
% 
% 
% 
% % 定义x和y的范围
% x = linspace(10000, 20000, 10000);
% y = linspace(0, 1000, 1000);
% 
% % 生成网格
% [X, Y] = meshgrid(x, y);
% 
% % 定义函数
% f = @(x,y) sqrt((17800 - x).^2 + y.^2)/140 - (20000 - x)/298.5;
% 
% % 计算函数值
% F = sqrt((17800 - X).^2 + Y.^2)/140 - (20000 - X)/298.5;
% 
% % 绘制隐函数曲线
% fimplicit(f, 'Color', 'k');
% 
% % 填充满足F < 0的区域
% contourf(X, Y, F, [0 0], 'LineStyle', 'none');
% 
% % 添加标签和标题
% xlabel('x');
% ylabel('y');
% title('区域：(sqrt((17800-x)^2+y^2)/140) < ((20000-x)/298.5)');
% 
% % 调整坐标轴比例
% axis equal;
% 
% % 添加网格
% grid on;




% 定义参数范围
z = linspace(200, 2000, 1000);
y = linspace(0, 1000, 1000);
[Z, Y] = meshgrid(z, y);

% 计算对应的x值
X = 10 .* Z;

% 计算不等式条件
F = sqrt((17800 - X).^2 + Y.^2)/140 - (20000 - X)/298.5;
condition = F < 0;

% 提取满足条件的点
idx = find(condition);
x_points = X(idx);
y_points = Y(idx);
z_points = Z(idx);

% 创建图形
figure;
scatter3(x_points, y_points, z_points, 10, 'red', 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('x');
ylabel('y');
zlabel('z');
title('x=10z平面上满足条件的区域');
grid on;
view(30, 30);
xlim([10000, 20000]);
ylim([0, 1000]);
zlim([200, 2000]);







% 定义参数范围
z = linspace(200, 2000, 1000);
y = linspace(0, 1000, 1000);
[Z, Y] = meshgrid(z, y);

% 计算对应的x值
X = 10 .* Z-(100/207).*Y;

% 计算不等式条件
F = sqrt((17800 - X).^2 + Y.^2)/140 - (20000 - X)/298.5;
condition = F < 0;

% 提取满足条件的点
idx = find(condition);
x_points = X(idx);
y_points = Y(idx);
z_points = Z(idx);

% 创建图形
figure;
scatter3(x_points, y_points, z_points, 10, 'red', 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('x');
ylabel('y');
zlabel('z');
title(' -207x - 100y + 2070z = 0平面上满足条件的区域');
grid on;
view(30, 30);
xlim([10000, 20000]);
ylim([0, 1000]);
zlim([200, 2000]);





% 定义参数范围
z = linspace(200, 2000, 500);
y = linspace(0, 1000, 500);
[Z, Y] = meshgrid(z, y);

% 创建图形
figure;
hold on;

% ===== 第一个平面：x = 10z =====
% 计算对应的x值
X1 = 10 .* Z;

% 计算不等式条件
F1 = sqrt((17800 - X1).^2 + Y.^2)/140 - (20000 - X1)/298.5;
condition1 = F1 < 0;

% 提取满足条件的点
idx1 = find(condition1);
x_points1 = X1(idx1);
y_points1 = Y(idx1);
z_points1 = Z(idx1);

% 绘制第一个平面的区域（蓝色）
scatter3(x_points1, y_points1, z_points1, 10, 'red', 'filled', 'MarkerFaceAlpha', 0.6);

% ===== 第二个平面：-207x - 100y + 2070z = 0 =====
% 计算对应的x值 (从平面方程推导：x = (2070z - 100y)/207)
X2 = (2070 .* Z - 100 .* Y) ./ 207;

% 计算不等式条件
F2 = sqrt((17800 - X2).^2 + Y.^2)/140 - (20000 - X2)/298.5;
condition2 = F2 < 0;

% 提取满足条件的点
idx2 = find(condition2);
x_points2 = X2(idx2);
y_points2 = Y(idx2);
z_points2 = Z(idx2);

% 绘制第二个平面的区域（红色）
scatter3(x_points2, y_points2, z_points2, 10, 'red', 'filled', 'MarkerFaceAlpha', 0.6);

% ===== 设置图形属性 =====
xlabel('x');
ylabel('y');
zlabel('z');
title('两个平面上满足条件的区域比较');
grid on;
view(30, 30);
xlim([10000, 20000]);
ylim([0, 1000]);
zlim([200, 2000]);

% 添加图例
legend({'x = 10z 平面上的区域', '-207x - 100y + 2070z = 0 平面上的区域'}, 'Location', 'best');

hold off;

% 显示统计信息
fprintf('x = 10z 平面上的点数: %d\n', length(idx1));
fprintf('-207x - 100y + 2070z = 0 平面上的点数: %d\n', length(idx2));













% 定义参数范围
z = linspace(200, 2000, 300);
y = linspace(0, 1000, 300);
[Z, Y] = meshgrid(z, y);

% ===== 第一个平面：x = 10z =====
X1 = 10 .* Z;
F1 = sqrt((17800 - X1).^2 + Y.^2)/140 - (20000 - X1)/298.5;
condition1 = F1 < 0;
idx1 = find(condition1);
points1 = [X1(idx1), Y(idx1), Z(idx1)];

% ===== 第二个平面：-207x - 100y + 2070z = 0 =====
X2 = (2070 .* Z - 100 .* Y) ./ 207;
F2 = sqrt((17800 - X2).^2 + Y.^2)/140 - (20000 - X2)/298.5;
condition2 = F2 < 0;
idx2 = find(condition2);
points2 = [X2(idx2), Y(idx2), Z(idx2)];

% 合并两个平面的点
all_points = [points1; points2];

% 使用 alphashape 创建三维几何体
shp = alphaShape(all_points(:,1), all_points(:,2), all_points(:,3), 1000);

% 创建图形
figure;
hold on;

% 绘制原始点（可选）
scatter3(points1(:,1), points1(:,2), points1(:,3), 10, 'red', 'filled', 'MarkerFaceAlpha', 0.3);
scatter3(points2(:,1), points2(:,2), points2(:,3), 10, 'red', 'filled', 'MarkerFaceAlpha', 0.3);

% 绘制填充的几何体
plot(shp, 'FaceColor', 'red', 'FaceAlpha', 0.9, 'EdgeColor', 'k');

% 设置图形属性
xlabel('x');
ylabel('y');
zlabel('z');
title('合并后的三维几何体');
grid on;
view(30, 30);
xlim([10000, 20000]);
ylim([0, 1000]);
zlim([200, 2000]);

legend({'x=10z平面点', '斜平面点', '合并几何体'}, 'Location', 'best');
hold off;

% 显示统计信息
fprintf('总点数: %d\n', size(all_points, 1));
fprintf('几何体体积: %.2f\n', volume(shp));







% 定义参数范围
z = linspace(200, 2000, 300);
y = linspace(0, 1000, 300);
[Z, Y] = meshgrid(z, y);

% ===== 第一个平面：x = 10z =====
X1 = 10 .* Z;
F1 = sqrt((17800 - X1).^2 + Y.^2)/140 - (20000 - X1)/298.5;
condition1 = F1 < 0;
idx1 = find(condition1);
points1 = [X1(idx1), Y(idx1), Z(idx1)];

% ===== 第二个平面：-207x - 100y + 2070z = 0 =====
X2 = (2070 .* Z - 100 .* Y) ./ 207;
F2 = sqrt((17800 - X2).^2 + Y.^2)/140 - (20000 - X2)/298.5;
condition2 = F2 < 0;
idx2 = find(condition2);
points2 = [X2(idx2), Y(idx2), Z(idx2)];

% 合并两个平面的点
all_points = [points1; points2];

% 使用 alphashape 创建三维几何体
shp = alphaShape(all_points(:,1), all_points(:,2), all_points(:,3), 1000);

% 创建图形
figure;
hold on;

% 绘制原始点（可选）
scatter3(points1(:,1), points1(:,2), points1(:,3), 10, 'red', 'filled', 'MarkerFaceAlpha', 0.3);
scatter3(points2(:,1), points2(:,2), points2(:,3), 10, 'red', 'filled', 'MarkerFaceAlpha', 0.3);

% 绘制填充的几何体 - 使用更立体的效果
plot(shp, 'FaceColor', [0.8, 0.2, 0.2], 'FaceAlpha', 0.9, 'EdgeColor', 'none', 'LineWidth', 1.5);

% 添加光照效果增强立体感
light('Position', [30000, 500, 1000], 'Style', 'infinite');
lighting gouraud;
material([0.6, 0.8, 0.2, 10, 0.5]);

% 设置图形属性
xlabel('x');
ylabel('y');
zlabel('z');
title('合并后的三维几何体');
grid on;
view(45, 30); % 调整视角以获得更好的立体效果
xlim([10000, 20000]);
ylim([0, 1000]);
zlim([200, 2000]);

% 设置坐标轴属性以获得更好的视觉效果
set(gca, 'LineWidth', 1.5, 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XColor', [0.3, 0.3, 0.3], 'YColor', [0.3, 0.3, 0.3], 'ZColor', [0.3, 0.3, 0.3]);

legend({'x=10z平面点', '斜平面点', '合并几何体'}, 'Location', 'best', 'FontSize', 10);
hold off;

% 显示统计信息
fprintf('总点数: %d\n', size(all_points, 1));
fprintf('几何体体积: %.2f\n', volume(shp));