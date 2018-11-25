rng(1); % For reproducibility

% Class 1 - red points
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1); % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
data1 = data1';

% Class 2 - blue points
r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1); % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points
data2 = data2';

% Plot the data points only
figure;
plot(data1(1,:),data1(2,:),'r.','MarkerSize',15)
hold on
plot(data2(1,:),data2(2,:),'b.','MarkerSize',15)
%@ezpolar(@(x)1);ezpolar(@(x)2);
axis equal

% Initialise values needed for quadprog.
% Arbitrary C value.
C = 0.4;

% Apply linear kernel to data points for each class
K_x_C1 = data1' * data1;
K_x_C2 = data2' * data2;
y1 = ones(1, 100);
y2 = -ones(1, 100);
y = [y1 y2]';

% Identify parameters to quadprog function.
f_C1 = -(diag(K_x_C1))';
f_C2 = -(diag(K_x_C2))';
A = zeros(1, 100);
H_C1 = 2 * K_x_C1;
H_C2 = 2 * K_x_C2;
c = 0;
A_e = ones(1, 100);
c_e = 1;
g_l = zeros(100,1);
g_u = C * ones(100,1);

% Find optimal Lagrangian multipliers for each class.
lambda_C1 = quadprog(H_C1, f_C1, A, c, A_e, c_e, g_l, g_u);
lambda_C2 = quadprog(H_C2, f_C2, A, c, A_e, c_e, g_l, g_u);

% Substitute optimal lambda to Lagrangian. Solution is equal to the optimal
% solution for primal problem (-d* = p*). We can then solve for R to find
% optimal radius of circle.
opt = -diag(K_x_C1)' * lambda_C1 + lambda_C1' * K_x_C1 * lambda_C1;
optR_C1 = sqrt(-opt);

opt = -diag(K_x_C2)' * lambda_C2 + lambda_C2' * K_x_C2 * lambda_C2;
optR_C2 = sqrt(-opt);

% Find vector a (given by sum of x and lambda, see report) for each class.
a_C1 = zeros(2, 1);
for j = 1 : 100
    a_C1 = a_C1 + lambda_C1(j) * data1(:, j);
end

a_C2 = zeros(2, 1);
for j = 1 : 100
    a_C2 = a_C2 + lambda_C2(j) * data2(:, j);
end

% Plot optimal hypersphere - plot circle using center (vector a) and rad.
centerX = a_C1(1); centerY = a_C1(2); radius = optR_C1;
theta = 0 : (pi / 50) : (2 * pi);
xunit = radius * cos(theta) + centerX;
yunit = radius * sin(theta) + centerY;
plot(xunit, yunit);

centerX = a_C2(1); centerY = a_C2(2); radius = optR_C2;
theta = 0 : (pi / 50) : (2 * pi);
xunit = radius * cos(theta) + centerX;
yunit = radius * sin(theta) + centerY;
plot(xunit, yunit);

title('Graph of optimal hypersphere enclosing class 1 and class 2', ...
          'fontsize', 15);

hold off;