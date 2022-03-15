function x = wedge(X)
% wedge: se(3) -> R^6
x = [unskew(X(1:3,1:3)); X(1:3,4)];
end