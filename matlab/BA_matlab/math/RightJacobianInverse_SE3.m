function output = RightJacobianInverse_SE3(xi)
Jr = RightJacobian_SE3(xi);
output = Jr \ eye(size(Jr));
end