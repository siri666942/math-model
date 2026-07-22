clear;clc

narvs = 4; % 变量个数
x_lb = [1    70  15  0]; % x的下界(长度等于变量的个数，每个变量对应一个下界约束)
x_ub = [1.5 140  30  3]; % x的上界
options = optimoptions('particleswarm','SwarmSize',4000,'MaxIterations',1000) 
[x,fval,exitflag,output] = particleswarm(@Obj_fun,narvs,x_lb,x_ub,options)   