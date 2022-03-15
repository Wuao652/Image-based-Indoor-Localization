function output = RightJacobian_SE3(xi)
output = Adjoint_SE3(expm(hat(-xi))) * LeftJacobian_SE3(xi);
end

