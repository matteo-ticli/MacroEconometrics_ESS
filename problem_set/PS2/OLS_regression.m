function [b,res,cov_b] = OLS_regression(y,x)

b = x\y;
res = y-x*b;
cov_b = inv(x'*x)*cov(res);
