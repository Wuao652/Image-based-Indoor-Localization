function X = hat(x)
% hat: R^6 -> se(3)
X = [skew(x(1:3)), x(4:6); 0 0 0 0];
end
