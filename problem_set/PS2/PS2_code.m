% Problem Set 2
%% ex.1

%1.a-b
clear
rng(42)

T = 1000;
simulations = 10000;
alpha = 0.2;
sigma = sqrt(0.1);
rho_null = 1;
for t = 1:simulations
    % simulate the AR(1) with constant parameter
    epsilon = randn(T,1)*sigma;
    y = filter(1, [1, -1], epsilon+alpha);

    % estimate the regression
    X_1 = y(1:end-1);
    X = [ones(length(X_1), 1), X_1];
    Y = y(2:end);

    [beta, res, cov_beta] = OLS_regression(Y, X); %% 
    
    t_stat(t) = (beta(2) - rho_null)/sqrt(cov_beta(2,2));
end

disp('Probability of rejecting rho_null=1:');
disp(sum(abs(t_stat)>1.96)/simulations);

disp('Mean of the t_stat:')
disp(mean(t_stat))

[t_test_func, values_range_t_test]=ksdensity(t_stat);
norm=randn(simulations*T,1);
[normal_func, values_range_normal]=ksdensity(norm);
hold on
plot(values_range_t_test, t_test_func, 'b')
plot(values_range_normal, normal_func, 'r')
hold off
legend('Test statistic distribution','Standard normal distribution');

%1.c
clear
rng(42)

T = 1000;
simulations = 10000;
alpha = 0.2;
delta = 0.5;
time_variation = (2:T)'; %for the regressor X
sigma = sqrt(0.1);
rho_null = 1;

for t = 1:simulations

    y = filter(1,[1 -1],randn(T,1)+alpha+delta*(1:T)');

    %esimate the regression
    X_1 = y(1:end-1);
    X = [ones(length(X_1), 1), time_variation, X_1];
    Y = y(2:end);
    
    [beta, res, cov_beta] = OLS_regression(Y, X); 
    
    t_stat(t) = (beta(3) - rho_null)/sqrt(cov_beta(3,3));

end

disp('Probability of rejecting rho_null=1:');
disp(sum(abs(t_stat)>1.96)/simulations);

disp('Mean of the t_stat:')
disp(mean(t_stat))

[t_test_func, values_range_t_test]=ksdensity(t_stat);
norm=randn(simulations*T,1);
[normal_func, values_range_normal]=ksdensity(norm);
hold on
plot(values_range_t_test, t_test_func, 'b')
plot(values_range_normal, normal_func, 'r')
hold off
legend('Test statistic distribution','Standard normal distribution');


%% ex.2

% case 1
clear
rng(42)

T = 1000;
simulations = 10000;
phi_y = 0.1;
phi_z = 0.3;
sigma_y = sqrt(0.2);
sigma_z = sqrt(0.3);
beta_null = 0;

for t = 1:simulations
    
    epsilon_y = randn(T,1)*sigma_y;
    epsilon_z = randn(T,1)*sigma_z;
    y = filter(1,[1 -phi_y],epsilon_y);
    z = filter(1,[1 -phi_z],epsilon_z);
    
    Y = y;
    X = [ones(T,1) z];

    [beta,res,cov_beta] = OLS_regression(Y,X);
    
    t_stat_1(t, 1) = (beta(2) - beta_null)/sqrt(cov_beta(2,2));
    
    RSS = sum(res.^2);
    TSS = sum((y - mean(y)).^2);
    R2(t, 1) = 1 - RSS/TSS;

end

% plot the kernenl density of the t statistics
[t_test_func_1, values_range_t_test_1]=ksdensity(t_stat_1);
plot(values_range_t_test_1, t_test_func_1, 'b')
legend('Test statistic distribution');

disp('Case 1: prob of rej beta_null=0');
disp(sum(abs(t_stat_1)>1.96)/simulations);

disp('Result of R2:');
disp(mean(R2))

% case 2
rng(42)

T = 1000;
simulations = 10000;
phi_y = 1;  % y is not stationary
phi_z = 0.2; % z is stationary as suggested by Enders
sigma_y = sqrt(0.2);
sigma_z = sqrt(0.3);
beta_null = 0;

for t = 1:simulations
    epsilon_y = randn(T,1)*sigma_y;
    epsilon_z = randn(T,1)*sigma_z;
    y = filter(1,[1 -phi_y],epsilon_y);
    z = filter(1,[1 -phi_z],epsilon_z);

    Y = y;
    X = [ones(T,1) z];

    [beta,res,cov_beta] = OLS_regression(Y,X);
    
    t_stat_2(t, 1) = (beta(2) - beta_null)/sqrt(cov_beta(2,2));
    
    RSS = sum(res.^2);
    TSS = sum((y - mean(y)).^2);
    R2(t, 1) = 1 - RSS/TSS;
end


% plot the kernenl density of the t statistics
[t_test_func_2, values_range_t_test_2]=ksdensity(t_stat_2);
plot(values_range_t_test_2, t_test_func_2, 'b')
legend('Test statistic distribution');

disp('Case 2: prob of rej beta_null=0');
disp(sum(abs(t_stat_2)>1.96)/simulations);

disp('Result of R2:');
disp(mean(R2))

% case 3
rng(42)

T = 1000;
simulations = 10000;
phi_y = 1;  % y is not stationary
phi_z = 1;  % z in not stationary
sigma_y = sqrt(0.2);
sigma_z = sqrt(0.3);

beta_null = 0;

for t = 1:simulations
    epsilon_y = randn(T,1)*sigma_y;
    epsilon_z = randn(T,1)*sigma_z;
    y = filter(1,[1 -phi_y],epsilon_y);
    z = filter(1,[1 -phi_z],epsilon_z);

    Y = y;
    X = [ones(T,1) z];

    [beta,res,cov_beta] = OLS_regression(Y,X);

    t_stat_3(t, 1) = (beta(2) - beta_null)/sqrt(cov_beta(2,2));
    
    RSS = sum(res.^2);
    TSS = sum((y - mean(y)).^2);
    R2(t, 1) = 1 - RSS/TSS;
end

% plot the kernenl density of the t statistics
[t_test_func_3, values_range_t_test_3]=ksdensity(t_stat_3);
plot(values_range_t_test_3, t_test_func_3, 'b')
legend('Test statistic distribution');

disp('Case 3: prob of rej beta_null=0');
disp(sum(abs(t_stat_3)>1.96)/simulations);

disp('Result of R2:');
disp(mean(R2))

% case 4

rng(42)

T = 1000;
simulations = 10000;
phi_y = 1;  % y is not stationary
phi_z = 1;  % z in not stationary
sigma_y = sqrt(0.2);
sigma_z = sqrt(0.3);
sigma_mu = sqrt(0.2);

beta_null = 0;

for t = 1:simulations
    epsilon_y = randn(T,1)*sigma_y;
    epsilon_z = randn(T,1)*sigma_z;
    epsilon_mu = randn(T,1)*sigma_mu;
    mu = filter(1, [1 -1], epsilon_mu);
    y = filter(1, 1,epsilon_y+mu);
    z = filter(1, 1,epsilon_z+mu);

    Y = y;
    X = [ones(T,1) z];

    [beta,res,cov_beta] = OLS_regression(Y,X);

    t_stat_4(t, 1) = (beta(2) - beta_null)/sqrt(cov_beta(2,2));
    
    RSS = sum(res.^2);
    TSS = sum((y - mean(y)).^2);
    R2(t, 1) = 1 - RSS/TSS;
end

% plot the kernenl density of the t statistics
[t_test_func_4, values_range_t_test_4]=ksdensity(t_stat_4);
plot(values_range_t_test_4, t_test_func_4, 'b')
legend('Test statistic distribution');

disp('Case 4: prob of rej beta_null=0');
disp(sum(abs(t_stat_4)>1.96)/simulations);

disp('Result of R2:');
disp(mean(R2))


%plots
subplot(2,2,1);plot(values_range_t_test_1, t_test_func_1, 'b');title('case 1');
subplot(2,2,2);plot(values_range_t_test_2, t_test_func_2, 'b');title('case 2');
subplot(2,2,3);plot(values_range_t_test_3, t_test_func_3, 'b');title('case 3');
subplot(2,2,4);plot(values_range_t_test_4, t_test_func_4, 'b');title('case 4');

%% ex.3
clear
rng(42)

df = xlsread('Romer_Romer.xlsx');

[T, M] = size(df);
n_lags = 4; % required by the exercise

var_model = varm(M-1, n_lags); % M-1 first column is date column

est_var_model = estimate(var_model, df(:, 2:M));

X_T_regressor = T - n_lags; % number of observations in X matrix
X_vars = n_lags*(M-1)+1;    % number of variables in X matrix (4 lags of 4 variables and a constant term)
X = ones(X_T_regressor, X_vars);

% loop to construct X for Wald Test
counter = 0;
for j = 1:n_lags:(X_vars-n_lags)
    X(:, 1+j:j+n_lags) = df(n_lags-counter:end-1-counter, 2:M);
    counter = counter + 1;
end

% We have to create the Wald Statistics (Chi squared with 4 degrees of 
% freedom, 4 joint restrictions to test)

% We first need to create the matrix R that yields the coefficients of the
% Romer and Romer variables (4 lags in time)

R = zeros(n_lags, X_vars); % 4 rows (lags) and 17 columns (variables of X)
for j = 1:n_lags
    R(j, 1+j*n_lags) = 1;
end


q = zeros(n_lags,1); % number of joint restrictions to test

for j = 1:M-1-1
    beta_hat = [est_var_model.Constant(j)];
    for k = 1:n_lags
        coeff = est_var_model.AR{1, k}(j,:);
        beta_hat = [beta_hat, coeff];
    end
    beta_hat = beta_hat';
    sigma_2 = est_var_model.Covariance(j,j);
    
    %finally construct the statistics after the praeludium
    left = (R*beta_hat - q)';
    center = inv(sigma_2*R*inv(X'*X)*R');
    right = (R*beta_hat - q);
    chi_stat(j, 1) = left * center * right;
    p_value_chi_stat(j,1) = 1 - chi2cdf(chi_stat(j, 1), n_lags);
end

disp('Chi^2 Statistics p-value of Inflation:');
disp(p_value_chi_stat(1, 1))

disp('Chi^2 Statistics p-value of Unemployment:');
disp(p_value_chi_stat(2, 1))

disp('Chi^2 Statistics p-value of Federal Fund Rate:');
disp(p_value_chi_stat(3, 1))

%% ex.4

clear
rng(42)

T = 500;
beta = 0.6;
n_lags = 4;
sigma_eta = sqrt(1);
sigma_eps = sqrt(0.8);

shocks = 2;
n_vars = 2;

% simulation parameters
simulations = 500;
N = simulations;

% simulate the process of the impulse response

% the impulse response will tend to zero very rapidly, 
% there is not much sense in generating 500 observations.

% the theoretical impulse response's shock persists only three periods
% therefore we will use 12 periods where we store the empirical impulse
% response estimated from the VAR
T_ir = 12;
irf = zeros(T_ir+n_lags-1, n_vars, N, shocks);

for n = 1:N
    %generate the data at every iteration of the simulation
    eta_ir = randn(T, 1)*sigma_eta;
    eps_ir = randn(T+2, 1)*sigma_eps;
    
    y(:, 1) = filter(1, 1, eta_ir(1:T)+eps_ir(1:T));
    y(:, 2) = filter(1, 1, beta/(1-beta)*eta_ir(1:T,1) + beta.^2/(1-beta)*eps_ir(3:T+2,1) + beta*eps_ir(2:T+1,1));

    % estimate the VAR model with four lags, for every iteration of the
    % simulation
    var_model = varm(size(y,2), n_lags);
    est_var_model = estimate(var_model, y);
    
    G = chol(est_var_model.Covariance);

    %set the value for the epsilon shock equal to G to see the evolution of
    %the impulse response function
    
    for shock = 1:shocks
    
        irf(n_lags, :, n, shock) = G(shock,:);
        
        for t = n_lags+1:T_ir+n_lags-1
            for lag = 1:n_lags
                irf(t, :, n, shock) = irf(t, :, n, shock)' + ...
                    est_var_model.AR{1, lag}*irf(t-lag, :, n, shock)';      
            end
        end
    end
end

irf = irf(4:end, :, :, :);

% calculate the metrics required of the empirical impulse response function
irf_mean(:, :, 1, :) = mean(irf(:, :, :, :), 3);
irf_5_perc(:, :, 1, :) = prctile(irf, 5, 3);
irf_95_perc(:, :, 1, :) = prctile(irf, 95, 3);


% calculate the teoretical impulse response function
irf_th = zeros(T_ir, n_vars, shocks);
irf_th(1, 1, 1) = 1;
irf_th(1, 2, 1) = beta/(1 - beta);
irf_th(1, 2, 2) = beta.^2 / (1 - beta);
irf_th(2, 2, 2) = beta;
irf_th(3, 1, 2) = 1;

% plotting the difference between theoretical and empirical impulse
% response functions

counter = 1;
for shock = 1:shocks
    for var = 1:n_vars
        subplot(n_vars, shocks, counter);
        hold on
        % patch([(1:T_ir) fliplr((1:T_ir))], [irf_5_perc(:, var, :, shock)' fliplr(irf_95_perc(:, var, :, shock)')], [0.3010 0.7450 0.9330])
        plot(irf_th(:, var, shock), 'Color', 'r', 'LineWidth', 1);
        plot(irf_mean(:, var, :, shock), 'Color', 'b');
        plot(irf_95_perc(:, var, :, shock), 'Color', 'b', 'LineStyle','--');
        plot(irf_5_perc(:, var, :, shock), 'Color', 'b', 'LineStyle','--');
        
        title(sprintf('IRF of shock %d for variable %d', shock, var));
        hold off
        legend('Th. IRF', 'Emp. IRF', '95% quantile')
        counter = counter + 1;
    end

end











