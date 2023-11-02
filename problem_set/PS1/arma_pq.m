function [y_ARMA, epsi_ARMA] = arma_pq(ar, ma, var, T, roots)
% we expect to receive to vectors of coefficients for AR and MA processes with 1 as first element
% we expect to receive AR = [1 , x, y, z], where x,y, z will be the p coefficients of the AR part of the ARMA ==> in this case p=3
% we expect to receive MA = [1 , a, b], where a, b will be the q coefficients of the MA part of the ARMA ==> in this case q=2

AR_coeff = ar * (-1);  % from LHS to RHS
MA_coeff = ma;   % just renaming

% when the users wants to provide the roots instead, the user must give as input "roots=1" and provide the roots (the input may not have 1 as first element in this case).
if roots == 1
    AR_coeff = -poly(1./ar); % from LHS to RHS
    MA_coeff = poly(1./ma);  % we take the inverse of the roots by definition of the polynomial
end

% lenght of the AR & MA
p=length(AR_coeff)-1; % as the first element is 1, the number of lags of y is equal to the lenght of the the AR coefficient array minus 1
q=length(MA_coeff)-1;  % as the first element is 1, the number of lags of epsilon is equal to the lenght of the MA coefficient array minus 1
            
% generating the vector of y_ARMA and the corresponding white noise (considering sigma)
y_ARMA = zeros(T,1);
epsi_ARMA=randn(T,1)*sqrt(var);

% generating some temporary storage variables
y_temp=zeros(p,1);
epsi_temp = zeros(q+1,1);
p1=zeros(T,1);
p2=zeros(T,1);

epsi_temp = [epsi_ARMA(1); epsi_temp(1:q)]; %useful for 1st iteration


% generating ARMA(p,q)
for t=1:T   
    p1(t,1) = AR_coeff(2:end)*y_temp(1:p,1); % AR part, note that we start from the 2nd value of the AR_coeff as the first coefficient = 1
    p2(t,1) = MA_coeff*epsi_temp(1:q+1,1);   % MA part

    y_ARMA(t,1) = p1(t,1) + p2(t,1);   % summing AR and MA parts
    
    if t==T % if this is the last iteration, we don't need to update our temporary vectors
        break
    end

    epsi_temp= [epsi_ARMA(t+1); epsi_temp(1:q,1)];  % updating temporary vector of errors     
    y_temp= [y_ARMA(t) ; y_temp(1:p-1,1)];      % updating temporary vector of y  

end