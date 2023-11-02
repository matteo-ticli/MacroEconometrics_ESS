% problem 1
clear
rng(42)
df = xlsread('data_ps3.xlsx', 'monetary_shock');
[T, M] = size(df);

% estimate the VAR model
n_lags = 1;
var_model = varm(M, n_lags);
est_var_model = estimate(var_model, df);

A_hat = est_var_model.AR{1,n_lags};
Eps_hat = est_var_model.infer(df);

IRF_x = irf(est_var_model, NumObs=50);

sim = 1000;
for h = 1:sim
%permute the residual (innovation) vector
% do not use permute because it switches (permute) dimensions of the array

    PER = randi([1,T-1],1, T-1);
    
    Eps_tilda = Eps_hat(PER, :);
    
    % generate the new series y_tilda
    
    y_tilda = zeros(T-1, M);
    
    y_t1 = df(1, :);

    for j = 2:T
        y_t = y_t1*A_hat' + Eps_tilda(j-1, :);
        y_tilda(j-1, :) = y_t;
        y_t1 = y_t;
    end
    
    % estimate VAR on y_tilda
    
    var_model_y_tilda = varm(M, n_lags);
    est_var_model_y_tilda = estimate(var_model_y_tilda, y_tilda);
    
    IRF_y_tilda(:, :, :, h) = irf(est_var_model_y_tilda, NumObs=50);
end

% compute the 95% conf int of the Bootstrap distribution
irf_5_perc(:, :, :) = prctile(IRF_y_tilda, 2.5, 4);
irf_95_perc(:, :, :) = prctile(IRF_y_tilda, 97.5, 4);

% compute the FEVD

mon_shock = 3; % variables that refers to the monetary shock

G = chol(est_var_model.Covariance)';
omega = G*G';

len_irf = size(IRF_x, 1);

FEVD_mon_shock = zeros(len_irf, M);

for j = 1:len_irf
    if j == 1
        FE = G(:, mon_shock)*G(:, mon_shock)';
        tot_VAR = omega;
        FEVD_mon_shock(j, :) = diag(FE)./diag(omega);
    
    elseif j ~= 1
        phi = squeeze(IRF_x(j-1, :, :))';
        FE = FE +  phi*G(:, mon_shock)*G(:, mon_shock)'*phi';
        tot_VAR = tot_VAR + phi*omega*phi';
        FEVD_mon_shock(j, :) = diag(FE)./diag(tot_VAR);
    end
end


% plot the shocks 

n_vars = 3;
variables = ["log(GDP)", "log(P)", "FFR"];
shocks=["Y", "P", "Monetary"];

counter = 1;
for var = 1:n_vars
    for var2 = 1:n_vars
        subplot(n_vars, n_vars, counter);
        hold on
        plot(IRF_x(:, var2, var), 'Color', 'black');
        plot(irf_95_perc(:, var2, var), 'Color', 'red', 'LineStyle','--');
        plot(irf_5_perc(:, var2, var), 'Color', 'red', 'LineStyle','--');
        
        title(shocks(var2)+' Shock, IRF for var: ' + variables(var));
        hold off
        legend('IRF', '95% conf. int.')
        counter = counter + 1;
    end
end

%% ex.2
clear
rng(42)

df = xlsread("data_ps3.xlsx", "technology_shock");

% Estimate Structural VAR-IRF

A = df(:, 1);
N = df(:, 2);
Y = A.*N;

% multiply the log vars by 100 to make the effect more visible 
log_A = log(A)*100;
log_N = log(N)*100;
log_Y = log(Y)*100;

delta_log_A = diff(log_A);
delta_log_N = diff(log_N);

n_lags = 4;
[T, M] = size(df);

var_model_main = varm(M, n_lags);
est_var_model_main = estimate(var_model_main, [delta_log_A delta_log_N]);

A_hat = est_var_model_main.AR;
Eps_hat = est_var_model_main.infer([delta_log_A delta_log_N]);

IRF_var = irf(est_var_model_main,  NumObs=50);

%reference at page 73 of lecture notes
% long run identification

D_11 = sum(IRF_var(:, 1, 1));
D_12 = sum(IRF_var(:, 2, 1));
D_21 = sum(IRF_var(:, 1, 2));
D_22 = sum(IRF_var(:, 2, 2));

D_L_1 = [D_11 D_12; D_21 D_22];

omega = est_var_model_main.Covariance;

SS = D_L_1*omega*D_L_1';

S = chol(SS)';

K = inv(D_L_1)*S;

inv_K = inv(K);

% calculate the impulse response of the reduced form for both shocks
n_shocks = M;
n_vars = M;
% there are 4 lags that have to be zeros when summed together
irf_red_form_main = zeros(size(IRF_var, 1)+n_lags, n_shocks, n_vars);

for shock = 1:n_shocks
    irf_red_form_main(n_lags, shock, :) = K(:, shock);

    for t = 1:size(IRF_var, 1)
        for lag = 1:n_lags
            irf_red_form_main(t+n_lags, shock, :) = squeeze(irf_red_form_main(t+n_lags, shock, :)) + ...
            est_var_model_main.AR{1, lag}*squeeze(irf_red_form_main(t + n_lags-lag, shock, :));
        end

    end
end

irf_red_form_main = irf_red_form_main(n_lags:end-1, :, :);

irf_levels_main = irf_red_form_main(1, :, :);
for t = 2:size(IRF_var, 1)
    irf_levels_main(t, :, :) = irf_red_form_main(t, :, :) + irf_levels_main(t-1, :, :);
end

% calculate the IRF of log(GDP)
% sum the IRF of the irf_reduced_form (estimation done in log terms)
irf_levels_main(:, :, 3) = irf_levels_main(:, :, 1) + irf_levels_main(:, :, 2);


% Bootstrap

sim = 1000;

T_Eps_hat = size(Eps_hat, 1);

irf_red_form = zeros(size(IRF_var, 1) + n_lags, M, M+1, sim);
irf_levels = zeros(size(IRF_var, 1) + n_lags, M, M+1, sim);

for h = 1:sim 
    PER = randi([1 T_Eps_hat], T_Eps_hat, 1);
    Eps_tilda = Eps_hat(PER, :);
    
    data_delta_log = [delta_log_A delta_log_N];
    y_t1 = data_delta_log(1:n_lags, :);
    y_tilda(1:n_lags, :) = y_t1;

    for t = n_lags+1:T_Eps_hat+n_lags
        y_t = [0; 0];
        for lag = 1:n_lags
            y_t = y_t + A_hat{1, lag}*y_t1(n_lags+1-lag, :)';
        end
           
        y_t = y_t' + Eps_tilda(t-n_lags, :);
        
        % array that stores the process
        y_tilda(t, :) = y_t;
        
        % update new "past observations" 
        y_t1 = y_tilda(t+1-n_lags: t, :);
    end
    
    % complete process for the iteration
    y_tilda = y_tilda(n_lags+1: end, :);

    % estimate a VAR on the bootstrapped process
    var_model_y_tilda = varm(M, n_lags);
    est_var_model_y_tilda = estimate(var_model_y_tilda, y_tilda);
    
    IRF_y_tilda(:, :, :, h) = irf(est_var_model_y_tilda,  NumObs=50);

    %reference at page 73 of lecture notes
    % long run identification
    
    D_11 = sum(IRF_y_tilda(:, 1, 1, h));
    D_12 = sum(IRF_y_tilda(:, 2, 1, h));
    D_21 = sum(IRF_y_tilda(:, 1, 2, h));
    D_22 = sum(IRF_y_tilda(:, 2, 2, h));
    
    D_L_1 = [D_11 D_12; D_21 D_22];
    
    omega = est_var_model_y_tilda.Covariance;
    
    SS = D_L_1*omega*D_L_1';
    
    S = chol(SS)';
    
    K = inv(D_L_1)*S;
    
    inv_K = inv(K);
    
    % calculate the impulse response of the reduced form for both shocks
    n_shocks = size(Eps_tilda, 2);
    
    for shock = 1:n_shocks
        irf_red_form(n_lags, shock, 1:n_shocks, h) = K(:, shock);
        for t = 1:size(IRF_y_tilda, 1)
            for lag = 1:n_lags
                irf_red_form(t+n_lags, shock, 1:n_shocks, h) = squeeze(irf_red_form(t+n_lags, shock, 1:n_shocks, h)) + ...
                est_var_model_y_tilda.AR{1, lag}*squeeze(irf_red_form(t + n_lags-lag, shock, 1:n_shocks, h));
            end
        end
    end
    
    irf_levels(n_lags, :, :, h) = irf_red_form(n_lags, :, :, h);
    for t = n_lags+1:size(irf_red_form, 1)
        irf_levels(t, :, :, h) = irf_red_form(t, :, :, h) + irf_levels(t-1, :, :, h);
    end
    
    % calculate the IRF of log(GDP)
    % sum the IRF of the irf_reduced_form (estimation done in log terms)
    irf_levels(:, :, 3, h) = irf_levels(:, :, 1, h) + irf_levels(:, :, 2, h);

end

% remove the first observations from the IRF vector
% used as placeholder

irf_levels = irf_levels(n_lags:end-1, :, :, :);

% compute 95% conf int of the Bootsrap distribution
irf_levels_5_perc(:, :, :) = prctile(irf_levels, 2.5, 4);
irf_levels_95_perc(:, :, :) = prctile(irf_levels, 97.5, 4);


% create new array to match Gali's Figure 2 in the paper..

% substitute the main IRF of the initial VAR on the actual data
irf_levels_main_2(:, :, 1) = irf_levels_main(:, :, 1);
irf_levels_main_2(:, :, 2) = irf_levels_main(:, :, 3);
irf_levels_main_2(:, :, 3) = irf_levels_main(:, :, 2);

% substitute the IRF for the bootsrapped 95% confidence interval
irf_levels_5_perc_2(:, :, 1) = irf_levels_5_perc(:, :, 1);
irf_levels_5_perc_2(:, :, 2) = irf_levels_5_perc(:, :, 3);
irf_levels_5_perc_2(:, :, 3) = irf_levels_5_perc(:, :, 2);

irf_levels_95_perc_2(:, :, 1) = irf_levels_95_perc(:, :, 1);
irf_levels_95_perc_2(:, :, 2) = irf_levels_95_perc(:, :, 3);
irf_levels_95_perc_2(:, :, 3) = irf_levels_95_perc(:, :, 2);

% plot
n_vars = 3;
variables = ["Productivity", "GDP", "Hours",];
shocks=["Tech", "Non-Tech"];

counter = 1;
for var = 1:n_vars
    for s = 1:n_shocks
        subplot(n_vars, n_shocks, counter);
        hold on
        plot(irf_levels_main_2(1:10, s, var), 'Color', 'black');
        plot(irf_levels_95_perc_2(1:10, s, var), 'Color', 'red', 'LineStyle','--');
        plot(irf_levels_5_perc_2(1:10, s, var), 'Color', 'red', 'LineStyle','--');
        
        title(shocks(s)+' Shock, IRF for variable: ' + variables(var));
        hold off
        legend('IRF', '95% conf. int.')

        counter = counter + 1;
    end
end




