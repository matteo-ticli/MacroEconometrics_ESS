%% Exercise 4
clear;
rng(42);

T_th_IRF = 20; % aribitrarily chosen
n_vars = 2; % given by the exercises
n_shocks = 2; % given by the exercise
n_lags = 2; % model presents two lags (y_{t-1} and y_{t-1})

% set up the theoretical model

%identity matrix to identify the shock of the system. need to be done for
%both variables
I = [1, 0; 0, 1];

%matrix coefficient that have been found through resolution of the
%thoretical points of the problem set
A1 = [0, 0; 0.5, 7/12];
A2 = [0.4, 0.5; 0, 0];

% store the vector for the theoretical IRF
y_IRF = zeros(T_th_IRF, n_shocks, n_shocks);

% compute the theoretical impulse responses
for shock = 1:n_shocks
    y_IRF(n_lags, :, shock) = I(:, shock);
    for t = n_lags+1: T_th_IRF+1
        y_IRF(t, :, shock) = A1*y_IRF(t-1, :, shock)' + ...
                             A2*y_IRF(t-2, :, shock)';
    end
end

%update the IRF function starting from the realization of the third
%observation due to presence of two lags within the model

y_IRF = y_IRF(n_lags:end, :, :); %first row as temp for the second lag

% plot the shocks
clf
variables = ["Y1", "Y2"];
shocks=["Shock 1", "Shock 2"];

counter = 1;
for var = 1:n_vars 
    for s = 1:n_shocks
        subplot(n_vars, n_vars, counter);
        hold on
        plot(y_IRF(:, var, s), 'Color', 'black');
        title(shocks(s)+', theoretical IRF for var: ' + variables(var));
        hold off
        legend('IRF th')
        counter = counter + 1;
    end
end


%% Exercise 5

% we impose parameter restriction for the simulation

T = 250;
MC_sim = 1000;
T_burn = 1000;
% generate 1000 obs for each sample. Discard (T_burn-T) observations to
% have unbiased long-run observations extracted from the model.

% store the vector of the MC simulation
y_MC_sim = zeros(T_burn, n_shocks, MC_sim);

for j = 1:MC_sim
    epsilon = randn(T_burn, n_shocks);
    for t = n_lags+1:T_burn
        y_MC_sim(t, :, j) = y_MC_sim(t-1, :, j)*A1' + ...
                         y_MC_sim(t-2, :, j)*A2' + ...
                         epsilon(t, :);
    end
end

% discard the first obs... sample from a proper realisation of the model
y_MC_sim = y_MC_sim(T_burn-T:end-1, :, :);

%% point (5.a)

% We estimate the VAR in levels and compute the empirical IRF

T_emp_IRF = 20; % arbitrarily chosen

% store the vector for the empirical IRF
y_emp_IRF = zeros(T_emp_IRF+n_lags, n_vars, n_shocks, MC_sim);

% Estimate a two lag var model on each sample of the MC simulation
% specification of the VAR model
var_model = varm(n_vars, n_lags);

% calculation of the empirical VMA (IRF)
for j=1:MC_sim
    est_var = estimate(var_model, y_MC_sim(:, :, j));
    for s=1:n_shocks
        y_emp_IRF(n_lags, :, s, j) = I(:, s);
        for t = n_lags+1:T_emp_IRF+n_lags
            y_emp_IRF(t,:,s,j)=est_var.AR{1, 1}*y_emp_IRF(t-1,:,s,j)' + ...
                               est_var.AR{1, 2}* y_emp_IRF(t-2,:,s,j)';            
        end
    end
end

% burn the first two observations due to presence of lags
y_emp_IRF = y_emp_IRF(n_lags:end-1, :, :, :);

% calculate the required statistics
y_emp_IRF_mean = mean(y_emp_IRF, 4);
y_emp_IRF_per1 = prctile(y_emp_IRF, 2.5, 4);
y_emp_IRF_per2 = prctile(y_emp_IRF, 97.5, 4);

% plot empirical statistics of IRF for the MC simulation
clf
variables = ["Y1", "Y2"];
shocks=["Shock 1", "Shock 2"];

counter = 1;
for var = 1:n_vars
    for s = 1:n_shocks
        subplot(n_vars, n_shocks, counter);
        hold on
        plot(y_emp_IRF_mean(:, var, s), 'Color', 'black');
        plot(y_emp_IRF_per1(:, var, s), 'Color', 'red', 'LineStyle','--');
        plot(y_emp_IRF_per2(:, var, s), 'Color', 'red', 'LineStyle','--');
        
        title(shocks(s)+', empirical IRF for variable: ' + variables(var));
        hold off
        legend('IRF emp.', '95% conf. int.')

        counter = counter + 1;
    end
end

%% point (5.b)

%estimate a misspecified VAR in difference

diff = 1; % first differening

% store the vector for the empirical first differencing model IRF
delta_y_emp_IRF = zeros(T_emp_IRF+diff-n_lags, n_vars, n_shocks, MC_sim);

% data generated from the MC simulation taken in first difference
delta_y_sim = y_MC_sim(diff+diff:end, :, :) - y_MC_sim(diff:end-diff, :, :);

% specification of the VAR model
var_model = varm(n_vars, diff);

% calculation of the empirical VMA (IRF)
for j=1:MC_sim
    est_var = estimate(var_model, delta_y_sim(:, :, j));
    for s=1:n_shocks
        delta_y_emp_IRF(diff, :, s, j) = I(:, s);
        for t = diff+1:T_emp_IRF+1
            delta_y_emp_IRF(t,:,s,j)=...
                est_var.AR{1, diff}*delta_y_emp_IRF(t-diff,:,s,j)';                                    
        end
    end
end

levels_y_emp_IRF = delta_y_emp_IRF(diff:end-diff,:,:,:);

% compute the IRF in levels, sum the differences
for t=diff+1:T_emp_IRF
    levels_y_emp_IRF(t,:,:,:) = levels_y_emp_IRF(t,:,:,:) + ...
                                levels_y_emp_IRF(t-diff,:,:,:);
end

% calculate the required statistics
levels_y_emp_IRF_mean = mean(levels_y_emp_IRF, 4);
levels_y_emp_IRF_per1 = prctile(levels_y_emp_IRF, 2.5, 4);
levels_y_emp_IRF_per2 = prctile(levels_y_emp_IRF, 97.5, 4);

% plot empirical statistics of IRF for the MC simulation specified in diff
clf
variables = ["Y1", "Y2"];
shocks=["Shock 1", "Shock 2"];

counter = 1;
for var = 1:n_vars
    for s = 1:n_shocks
        subplot(n_vars, n_shocks, counter);
        hold on
        plot(levels_y_emp_IRF_mean(:, var, s), 'Color', 'black');
        plot(levels_y_emp_IRF_per1(:, var, s), 'Color', 'red', ...
                                                'LineStyle','--');
        plot(levels_y_emp_IRF_per2(:, var, s), 'Color', 'red', ...
                                                'LineStyle','--');
        
        title(shocks(s)+', empirical IRF for variable: ' ...
            + variables(var));
        hold off
        legend('IRF emp.', '95% conf. int.')

        counter = counter + 1;
    end
end

%% point (5.c)

% new lags specification
n_lags_new = 4;

diff = 1; % first differening

% store the vector for the empirical first differencing model IRF
delta_y_emp_IRF_new = zeros(T_emp_IRF+n_lags_new-n_lags,...
                            n_vars, n_shocks, MC_sim);

% data generated from the MC simulation taken in first difference
delta_y_sim = y_MC_sim(diff+diff:end, :, :) - y_MC_sim(diff:end-diff, :, :);

% specification of the VAR model
var_model = varm(n_vars, n_lags_new);

% calculation of the empirical VMA (IRF)
for j=1:MC_sim
    est_var = estimate(var_model, delta_y_sim(:, :, j));
    for s=1:n_shocks
        delta_y_emp_IRF_new(n_lags_new, :, s, j) = I(:, s);
        for t = n_lags_new+1:T_emp_IRF+n_lags_new
            delta_y_emp_IRF_new(t,:,s,j)= ... 
                     est_var.AR{1, 1}*delta_y_emp_IRF_new(t-1,:,s,j)'+...
                     est_var.AR{1, 2}*delta_y_emp_IRF_new(t-2,:,s,j)'+...
                     est_var.AR{1, 3}*delta_y_emp_IRF_new(t-3,:,s,j)'+...
                     est_var.AR{1, 4}*delta_y_emp_IRF_new(t-4,:,s,j)';                                  
        end
    end
end

levels_y_emp_IRF_new = delta_y_emp_IRF_new(n_lags_new:end-1,:,:,:);

% compute the IRF in levels, sum the differences
for t=diff+1:T_emp_IRF
    levels_y_emp_IRF_new(t,:,:,:) = levels_y_emp_IRF_new(t,:,:,:) + ...
                                levels_y_emp_IRF_new(t-1,:,:,:);
end

% calculate the required statistics
levels_y_emp_IRF_mean_new = mean(levels_y_emp_IRF_new, 4);
levels_y_emp_IRF_per1_new = prctile(levels_y_emp_IRF_new, 2.5, 4);
levels_y_emp_IRF_per2_new = prctile(levels_y_emp_IRF_new, 97.5, 4);

% plot empirical statistics of IRF for the MC simulation specified in diff
clf
variables = ["Y1", "Y2"];
shocks=["Shock 1", "Shock 2"];

counter = 1;
for var = 1:n_vars
    for s = 1:n_shocks
        subplot(n_vars, n_shocks, counter);
        hold on
        plot(levels_y_emp_IRF_mean_new(:, var, s), 'Color', 'black');
        plot(levels_y_emp_IRF_per1_new(:, var, s), 'Color', 'red', ...
                                                   'LineStyle','--');
        plot(levels_y_emp_IRF_per2_new(:, var, s), 'Color', 'red', ...
                                                   'LineStyle','--');
        
        title(shocks(s)+', empirical IRF for variable: ' ...
            + variables(var));
        hold off
        legend('IRF emp.', '95% conf. int.')

        counter = counter + 1;
    end
end

%% point(5.d)h

% we implement the Johansen procedure

for j = 1:MC_sim
    % select the data for the current Johansen test
    df_sample = y_MC_sim(:,:,j);
    % compute the Johansen test and store mles
    [~,~,~,~,mles] = jcitest(df_sample,Model='H2',lags=1, display='off');
    % convert the VEC model from Johansen to a VAR model that can be
    % estimated using the standard module
    VEC = {mles.r1.paramVals.B1};
    var_model = vec2var(VEC, mles.r1.paramVals.A*mles.r1.paramVals.B');
    %estimate the VAR model obtained using Jhonasen procedure
    var_model_jo = varm(AR=var_model, Constant=[0;0], Covariance=eye(2));
    %compute the IRF
    y_jo_IRF(:,:,:,j) = var_model_jo.irf();
end

% calculate the required statistics
y_jo_IRF_mean = mean(y_jo_IRF, 4);
y_jo_IRF_per1 = prctile(y_jo_IRF, 2.5, 4);
y_jo_IRF_per2 = prctile(y_jo_IRF, 97.5, 4);

% plot statistics of IRF obtained using the Johansen procedure
clf
variables = ["Y1", "Y2"];
shocks=["Shock 1", "Shock 2"];

counter = 1;
for s = 1:n_shocks
    for var = 1:n_vars 
        subplot(n_vars, n_shocks, counter);
        hold on
        plot(y_jo_IRF_mean(:, var, s), 'Color', 'black');
        plot(y_jo_IRF_per1(:, var, s), 'Color', 'red', ...
                                                   'LineStyle','--');
        plot(y_jo_IRF_per2(:, var, s), 'Color', 'red', ...
                                                   'LineStyle','--');
        
        title(shocks(var)+', Johansen IRF for variable: ' ...
            + variables(s));
        hold off
        legend('IRF emp.', '95% conf. int.')

        counter = counter + 1;
    end
end



