function [] = partII()

    % generate the data

    rng(1); 
    r = sqrt(rand(100,1)); 
    t = 2*pi*rand(100,1);  
    data1 = [r.*cos(t), r.*sin(t)]; 
    
    r2 = sqrt(3*rand(100,1)+1); 
    t2 = 2*pi*rand(100,1);      
    data2 = [r2.*cos(t2), r2.*sin(t2)]; 

    % plot the data

    figure;
    plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
    hold on
    plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
    axis equal
    hold on

    % work on class 1
    [a1, R1] = calcRandCentre(data1);

    % work on class 2
    [a2, R2] = calcRandCentre(data2);
    
    disp(['a1(1),a1(2) ',num2str(a1(1)), ' ',num2str(a1(2)), ' radius1: ', num2str(R1)]);
    % plot centre and radius for class 1
    plot(a1(1), a1(2), 'rx', 'MarkerSize', 15);
    viscircles(a1', R1, 'Color', 'r', 'LineWidth', 1);
    hold on

    disp(['a2(1),a2(2) ',num2str(a2(1)),' ', num2str(a2(2)), ' radius2: ', num2str(R2)]);
    % plot centre and radius for class 2
    plot(a2(1), a2(2), 'bx', 'MarkerSize', 15);
    viscircles(a2', R2, 'Color', 'b', 'LineWidth', 1);

end

function [a, R] = calcRandCentre(data)


% Initialise values needed for quadprog.
% Arbitrary C value.
C = 0.4;

% Apply linear kernel to data points for each class
K_x = data * data';
% y1 = ones(1, 100);
% y2 = -ones(1, 100);
% y = [y1 y2]';

% Identify parameters to quadprog function.
H = 2 * K_x;
f = -(diag(K_x))';

A = zeros(1, 100);

c = 0;
A_e = ones(1, 100);
c_e = 1;
g_l = zeros(100,1);
g_u = C * ones(100,1);

% Find optimal Lagrangian multipliers for each class.
lambda = quadprog(H, f, A, c, A_e, c_e, g_l, g_u);


% Substitute optimal lambda to Lagrangian. Solution is equal to the optimal
% solution for primal problem (-d* = p*). We can then solve for R to find
% optimal radius of circle.
opt = -diag(K_x)' * lambda + lambda' * K_x * lambda;
opt_Radius = sqrt(-opt);


% Find vector a (given by sum of x and lambda, see report) for each class.
a = zeros(2, 1);
for j = 1 : 100
    a = a + lambda(j) * data(j);
end

R = opt_Radius;



end