function [X,Y,A,B] = generatenoise(d, t, n)
        A = abs(randn(d-t,d-t));
        B = randn(d-t,d-t);
        X = randn(d-t,n);
        Y = B * (max(A * X,0) + X);
end