%% Problem Set 1

%% ex 1
% we set the seed for replicability purposes
rng(42)

N = 500;
phi = 0.4;
sigma_2 = sqrt(0.2); % given variance (sigma_squared = 0.2)
epsilon = randn(N,1)*sigma_2;

%%% ex 1.a
% we don't have a starting condition for AR(1) process thus we choose the
% most likely value ExpValue(ya) = 0 for ya_1
ya_1 = 0; 
ya = phi*ya_1 + epsilon(1);

for t = 2:N
    ya(t, 1) = phi*ya(t-1, 1) + epsilon(t);
end

%%% ex 1.b
% we set numerator = 1 and denominator = [1 -phi] since it is AR process
yb = filter(1, [1 -phi], epsilon, 0);

%%% ex 1.c

% check that the two outputs coincide with same forcing variables
disp("The two processes are the same:")
disp(isequal(ya, yb))

% plot ya and yb on top
tiledlayout(2,2);

nexttile;
plot(ya, 'r');
legend('Y_t');
title('AR(1) process using "for loop"');

nexttile;
plot(yb, 'b');
legend('X_t');
title('AR(1) process using "filter"');

% plot both process together on the same plot
nexttile([1 2]);
plot(ya, 'r');
hold on;
plot(yb, 'b');
hold off
legend('Y_t', 'X_t')
title('Overlapping of AR(1) processes with same forcing variables')

%% ex. 2
% we set the seed for replicability purposes
rng(42)

% supposing to use the same number of observations, phi and sigma
% as in the previous exercise:
N = 500;
T = 200;
phi = 0.4;
sigma_2 = sqrt(0.2); % given variance (sigma_squared = 0.2)
epsilon = randn(N + T,1)*sigma_2;

% we have expected value of te process equal to 3
% the only way is to impose a drift (delta)
Exp_Value_yc = 3;
delta = (1-phi)*Exp_Value_yc;

% starting condition of the process
yc_1 = 10;

yc = delta + phi*yc_1 + epsilon(1);
for t = 2:N+T
    yc(t,1) = delta + phi*yc(t-1,1) + epsilon(t);
end

% we apply the filter function as before but we recall there is delta 
yd = filter(1, [1 -phi], epsilon + delta, phi*yc_1); % we use the starting condition for the process which is different from zero

y_trimmed = yc(end-N: end, 1);

disp("The two processes are the same:")
disp(isequal(yc(T:end), y_trimmed))

tiledlayout(2, 1);

x = linspace(0, N+T);
x1 = linspace(0, N);

nexttile;
plot(yc, 'r');
hold on
yline(Exp_Value_yc, "--", "LineWidth", 2);
legend('Y_t', 'Uncoditional Exp Value')
hold off
title('AR(1) process with starting condition 10');

nexttile;
plot(y_trimmed, 'b');
hold on
yline(Exp_Value_yc, "--", "LineWidth", 2);
hold off
legend("Y_t'", 'Uncoditional Exp Value');
title('trimmed AR(1) process from the starting condition');

%% ex. 3
% we set the seed for replicability purposes
rng(42)

%%% ex 3.a
N = 500;
theta = 0.3;
sigma_2 = sqrt(0.3); % given variance (sigma_squared = 0.3)
epsilon = randn(N,1)*sigma_2;

ye = epsilon(1); %most likely value for epsilon starting condition is 0
for t = 2:N
    ye(t, 1) = epsilon(t) + theta*epsilon(t-1);
end

%%% ex 3.b
yf = filter([1 +theta], 1, epsilon); 

disp("The two processes are the same:")
disp(isequal(ye, yf))

tiledlayout(2,2);

nexttile;
plot(ye, 'r');
legend('Y_t');
title('MA(1) process using "for loop"');

nexttile;
plot(yf, 'b');
legend('X_t');
title('MA(1) process using "filter"');

% plot both process together on the same plot
nexttile([1 2]);
plot(ye, 'r');
hold on;
plot(yf, 'b');
hold off
legend('Y_t', 'X_T')
title('Overlapping of MA(1) processes with same forcing variables')

%% ex 4
clear
rng(42);
ar = [1, 0.1, 0.5];
ma = [1, 0.3, 0.7];
sigma_2 = 0.6;
T = 400;
roots = 0;

[y_ARMA, epsilon_ARMA] = arma_pq(ar, ma, sigma_2, T, roots);

%% ex 5
% ex 5.a
clear;
rng(42)
phi = 0.4;
T = 250;
simulations = 10000;
sigma_2 = sqrt(0.2); %we chose the evolution variance of the process to be = 0.2

for t=1:simulations
    epsilon = randn(T, 1)*sigma_2;
    AR_1 = filter(1, [1 -phi], epsilon, 0);
    X = AR_1(1:end-1);
    Y = AR_1(2:end);
    [beta, ~, ~] = OLS_regression(Y, [ones(length(X), 1), X]);
    betas(t,:) = beta(2);
end

histogram(betas);
title('Empirical Distribution of OLS estimator');

% ex 5.b
rng(42);
T = 200;
phi_null = 0;

for t=1:simulations
    epsilon = randn(T, 1)*sigma_2;
    AR_1 = filter(1, [1 -phi], epsilon, 0);
    X = AR_1(1:end-1);
    Y = AR_1(2:end);
    [beta, ~, cov_beta] = OLS_regression(Y, [ones(length(X), 1) X]);
    betas(t,:) = beta(2);

    t_test(t, :) = (beta(2) - phi_null)/sqrt(cov_beta(2,2));
end

N_95_critical_level_two_sided = 1.96;
rejected = sum(abs(t_test) > N_95_critical_level_two_sided);
prob_rejected = rejected/simulations;
display(prob_rejected);

%% ex 6
clear;
rng(42);
phi = 0.9;
sigma_2 = sqrt(0.2);
simulations = 10000;
T_ARs = [50 100 200 1000];

c = 0;
for T=T_ARs
    c = c + 1;
    for t=1:simulations
        epsilon = randn(T, 1)*sigma_2;
        AR_1 = filter(1, [1 -phi], epsilon, 0);
        X = AR_1(1:end-1);
        Y = AR_1(2:end);
        [beta, ~, cov_beta] = OLS_regression(Y, [ones(length(X), 1) X]);
        betas(t,:) = beta(2);
    end
    betas_ARs(:, c) = betas;
end

subplot(2,2,1);histogram(betas_ARs(:,1));title('T=50');
subplot(2,2,2);histogram(betas_ARs(:,2));title('T=100');
subplot(2,2,3);histogram(betas_ARs(:,3));title('T=200');
subplot(2,2,4);histogram(betas_ARs(:,4));title('T=1000');
sgtitle('Different OLS empirical distributions for different AR(1) length');

%% ex.7
clear;
rng(42);
theta = 0.6;
sigma_2 = sqrt(0.2);
T = 250;
T_infinity = 10000;
simulations = 10000;

for t = 1:simulations
    epsilon = randn(T, 1)*sigma_2;
    MA_1 = filter([1 +theta], 1, epsilon); 
    X = MA_1(1:end-1);
    Y = MA_1(2:end);
    [beta, ~, ~] = OLS_regression(Y, X);
    betas(t, :) = beta;
    
    epsilon = randn(T_infinity, 1)*sigma_2;
    MA_1 = filter([1 +theta], 1, epsilon); 
    X = MA_1(1:end-1);
    Y = MA_1(2:end);
    [beta, ~, ~] = OLS_regression(Y, X);
    betas_infinity(t, :) = beta;
end   

subplot(2,1,1);
histogram(betas);
title('T=250');

subplot(2,1,2);
histogram(betas_infinity);
title('T=10000');

sgtitle("Empirical distributions of OLS estimator, same process different time length");

disp("The mean of Beta_OLS when T=250 :");
disp(mean(betas));

disp("The mean of Beta_OLS when T=10000 :");
disp(mean(betas_infinity));

%check the true beta of the regression beta_true = theta/(1 - theta^2)
disp("The convergence value of beta :");
disp(theta/(1 + theta^2))

%% ex 8
clear;

rng(42)

T = 250;
phi = 1;
sigma_2 = sqrt(0.2);
simulations = 10000;

% under the null hypothesis of the first difference regression,
% we want to test if rho is statistically different from 0.
rho_null = 0;

% clearly the simulation must be the same for empirical distributuion of
% betas and the one-sided test of hypothesis
for t=1:simulations
    epsilon = randn(T, 1)*sigma_2;
    AR_1 = filter(1, [1 -phi], epsilon, 0);
    X = AR_1(1: end-1);
    Y = AR_1(2: end);
    [beta, ~, ~] = OLS_regression(Y, [ones(length(X), 1) X]);
    betas1(t,:) = beta(2);

    first_diff = Y - X;
    [beta_fd, ~, cov_beta_fd] = OLS_regression(first_diff, [ones(length(X), 1) X]);
    betas_fd(t,:) = beta_fd(2);
    t_test_fd(t, :) = (betas_fd(t) - rho_null)/sqrt(cov_beta_fd(2, 2));
end

% PLOTS
tiledlayout(2,2);

nexttile;
histogram(betas1);
title('distr of OLS coeff, AR(1) process');
nexttile;
histogram(betas_fd);
title('distr of OLS coeff, first diff reg of AR(1) process');
nexttile([1 2]);
histogram(t_test_fd);
title('Distibution of t-statistics values using a standard normal distribution');


% the null hypothesis is rejected with the 95% critical value of a standard
% normal distribution for a one-sided test of hypothesis
N_95_critical_one_sided = -1.6449;
rejection = sum(t_test_fd < N_95_critical_one_sided); 
disp("The probability of rejecting the null hypothesis using a one sided test:");
disp(rejection/simulations);


%plot few percentiles of the empirical distribution of the t test
disp("Empirical distribution percentiles 1 - 2.5 - 5 - 10")
disp(prctile(t_test_fd, [1 2.5 5 10]));

disp("Dickey-Fuller distribution percentiles 1 - 2.5 - 5 - 10")
disp([-3.45  -3.14 -2.88 -2.58]);

%% ex 9

% ex 9 ab
clear;

rng(42)
T = 250;
delta = 0.3;
phi = 1;
sigma = sqrt(0.7);
time_variation = (2:T)'; % time parameter, unrestricted model
simulations = 10000;

for k = 1:simulations
    epsilon = randn(T, 1)*sigma;
    % we generate data using a random walk plus drift
    rw = filter(1, [1 -phi], epsilon+delta, 0);
    rw_t = rw(2:end);
    rw_t1 = rw(1:end-1);
    delta_rw = rw_t - rw_t1;
    
    % regressor of the restricted model, constant vector
    X_1 = [ones(length(delta_rw), 1)];
    [~, res, ~] = OLS_regression(delta_rw, X_1);
    
    %regressor of the unrestricted model, (constant vector, time vartiation, lagged variable)
    X_2 = [ones(length(delta_rw), 1) time_variation rw_t1]; 
    [~, res2, ~] = OLS_regression(delta_rw, X_2);

    res_squared = res.^2;
    res2_squared = res2.^2;

    RSS_1 = sum(res_squared);
    RSS_2 = sum(res2_squared);
    
    v1 = (size(X_2, 2) - size(X_1, 2));
    v2 = (length(X_2) - size(X_2, 2));
    
    F_stat(k, 1) = ((RSS_1 - RSS_2)/v1)/(RSS_2/v2);
    p_value_F(k, 1) = 1 - fcdf(F_stat(k, 1), v1, v2);

    Chi_stat(k, 1) = v1*F_stat(k, 1);
    p_value_Chi(k, 1) = 1 - chi2cdf(Chi_stat(k, 1), v1);
end

subplot(2,1,1);
histogram(Chi_stat);
title(['Empirical Distribution of Chi^2 statistics, ' ...
    'random walk plus drift process']);

subplot(2,1,2);
histogram(F_stat);
title(['Empirical Distribution of F-statistics, ' ...
    'random walk plus drift process']);

%plot few percentiles of the empirical distribution of the F test
disp("Empirical distribution percentiles 90 - 95 - 97.5 - 99")
disp(prctile(F_stat, [90 95 97.5 99]));

disp("Dickey-Fuller distribution percentiles 90 - 95 - 97.5 - 99")
disp([5.39 6.34 7.25 8.43]);

 % the value tabulated by Dickey Fuller using a sample size of the time series of 250 observations
DF_95_critical_value = 6.34;
disp("The probability of rejecting H0 " + ...
    "at 95% confidence interval using the critical value tabulated by Dickey Fuller is:");
disp(sum(F_stat > DF_95_critical_value )/simulations);

disp("The probability of rejecting H0 " + ...
    "at 95% confidence interval (p_value < 0.05) using an F distribution:");
disp(sum(p_value_F < 0.05)/simulations);

disp("The probability of rejecting H0 " + ...
    "at 95% confidence interval (p_value < 0.05) using an Chi_Squared distribution:");
disp(sum(p_value_Chi < 0.05)/simulations);


% ex 9.c
clear;

rng(42)
T = 250;
delta = 2;
gamma = 0.5;
sigma = sqrt(0.7);
time_variation = (2:T)'; % time paraneter, unrestricted model
simulations = 10000;


for k = 1:simulations
    epsilon = randn(T, 1)*sigma;
    % we generate data from a deterministic time trend (Xtt =  time trend)
    for t=1:T
        Xtt(t, 1) = delta + gamma*t + epsilon(t);
    end

    Xtt_t1 = Xtt(1:end-1);
    Xtt_t = Xtt(2:end);
    delta_Xtt = Xtt_t - Xtt_t1;

    X_1 = [ones(length(delta_Xtt), 1)];
    [~, res, ~] = OLS_regression(delta_Xtt, X_1);

    X_2 = [ones(length(delta_Xtt), 1) time_variation Xtt_t1];
    [~, res2, ~] = OLS_regression(delta_Xtt, X_2);

    res_squared = res.^2;
    res2_squared = res2.^2;

    RSS_1 = sum(res_squared);
    RSS_2 = sum(res2_squared);
    
    v1 = (size(X_2, 2) - size(X_1, 2));
    v2 = (length(X_2) - size(X_2, 2));

    F_stat(k, 1) = ((RSS_1 - RSS_2)/v1)/(RSS_2/v2);
    p_value_F(k, 1) = 1 - fcdf(F_stat(k, 1), v1, v2);

    Chi_stat(k, 1) = v1*F_stat(k, 1);
    p_value_Chi(k, 1) = 1 - chi2cdf(Chi_stat(k, 1), v1);

end


histogram(Chi_stat);
title('Empirical Distribution of F-statistics, determinist time trend model');

DF_95_critical_value = 6.34; % the value tabulated by Dickey Fuller using a sample size of the time series of 250 observations

disp("The probability of rejecting H0 " + ...
    "at 95% confidence interval using the value tabulated by Dickey Fuller is:");
disp(sum(Chi_stat > DF_95_critical_value )/simulations);

