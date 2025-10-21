function d = JSdiv(P,Q)
    epsVal = 1e-12;  % smoothing to avoid log(0)
    P = P + epsVal;
    Q = Q + epsVal;
    P = P / sum(P);
    Q = Q / sum(Q);
    M = 0.5*(P+Q);
    d = 0.5*sum(P .* log2(P./M)) + 0.5*sum(Q .* log2(Q./M));
end