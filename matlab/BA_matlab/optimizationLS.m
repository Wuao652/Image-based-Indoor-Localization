function [ Orient, robotpose,reproerror2,angle_var,position_var] = optimizationLS( observe,param )
%OPTIMIZATION Summary of this function goes here



%   State: Robot pose + number of features
x0 = zeros(6,1);
if(1)
    Orient = observe.orientation';
    x0(4:6) = observe.robotpose;
end

%   Detailed explanation goes here
[F,P] = CalculateF_LS(observe,param,x0,Orient,false);
J = JacobianF_LS(observe,param,x0,Orient);


reproerror1 = 0;
for i = 1 : param.num_features
    reproerror1 = reproerror1 + sqrt(F(2*i-1)^2 + F(2*i)^2);
end
reproerror1 = reproerror1 / param.num_features;
% m = size(F,1);
% v_diag = 1*ones(m,1);
% ind = 2*sum(sum(param.validity==true));
% for i = 1 : param.num_step
%     v_diag(ind+6*(i-1)+1:ind+6*(i-1)+3) = 1000;%   Orientation
%     v_diag(ind+6*(i-1)+4:ind+6*(i-1)+6) = 1;    %   Position
% end


% ============================  No outlier =========================%
% P = spdiags(v_diag,0,m,m);
M = 0.000001;
tic  %%%%timing
min_FX_old = 10000000000000;
k=0;
x = x0;    % initialize x
%   Without huber loss:
I = speye(size(J,2));

while ((F'*P*F)>M&&abs(F'*P*F-min_FX_old)>M&&k<50)
%     while (k<20)
    min_FX_old = F'*P*F;
%     J = sparse(J);
%     F = sparse(F);
%     P = sparse(P);
    u = 0.0000001;
    d = -(J'*P*J+u*I)\(J'*P*F);
    x_new = x + d;
    Orient_new = Orient*expm(skew(d(1:3)));
%     for i = 1 :param.num_step
%         Orient_new(3*(i-1)+1:3*i,:) = Orient(3*(i-1)+1:3*i,:)*expm(skew(d(6*(i-1)+1:6*(i-1)+3)));
%     end
    
    
    [F_new,P_new] = CalculateF_LS(observe,param,x_new,Orient_new,false);
    min_FX = F_new'*P_new*F_new;
    t = 1;
    while(min_FX > min_FX_old&&t<6)
        % LM algorithm
        d = -(J'*P_new*J+u*I)\(J'*P_new*F);
        x_new = x + d;
        Orient_new = Orient*expm(skew(d(1:3)));
%         for i = 1 :param.num_step
%             Orient_new(3*(i-1)+1:3*i,:) = Orient(3*(i-1)+1:3*i,:)*expm(skew(d(6*(i-1)+1:6*(i-1)+3)));
%         end
        [F_new,P_new] = CalculateF_LS(observe,param,x_new,Orient_new,1);
        min_FX = F_new'*P_new*F_new;
        u  = u * 100;
        t =  t + 1;
    end
    if(t==6)
        break
    end
    Orient = Orient*expm(skew(d(1:3)));
    x = x + d;
%     for i = 1 :param.num_step
%         Orient(3*(i-1)+1:3*i,:) = Orient(3*(i-1)+1:3*i,:)*expm(skew(d(6*(i-1)+1:6*(i-1)+3)));
%     end
    k=k+1;
    if(k > 5)
        [F,P] = CalculateF_LS(observe,param,x,Orient,1);
    else
        [F,P] = CalculateF_LS(observe,param,x,Orient,false);
    end
    J = JacobianF_LS(observe,param,x,Orient);
end

var=((F'*P*F)/(2*param.num_features))*inv(J'*P*J);
x_std = sqrt(diag(var));
angle_std = angleDifference_so3(x_std(1:3));
angle_var = (angle_std)^2;
position_var = norm(x_std(4:6))^2;
robotpose = x(4:6)';
F = CalculateF_LS(observe,param,x,Orient,0);
reproerror2 = 0;
for i = 1 : param.num_features
    reproerror2 = reproerror2 + sqrt(F(2*i-1)^2 + F(2*i)^2);
end
reproerror2 = reproerror2 / param.num_features;
disp(['Average reprojection error: ', num2str(reproerror1),' --->' num2str(reproerror2)]);
Orient = Orient';
% % ============================  Outlier detection =========================%
% m = size(F,1);
% v_diag = 1*ones(m,1);
% ind = 2*sum(sum(param.validity==true));
% for i = 1 : param.num_step
%     v_diag(ind+6*(i-1)+1:ind+6*(i-1)+3) = 0.001;%   Orientation
%     v_diag(ind+6*(i-1)+4:ind+6*(i-1)+6) = 0.000001;    %   Position
% end
% P = spdiags(v_diag,0,m,m);
% %   With huber loss:
% F = CalculateF_LS(observe,param,x,Orient,true);
% J = JacobianF_LS(observe,param,x,Orient);
% 
% min_FX_old = 1000000;
% k=0;
% I = speye(size(J,2));
% Orient_new = Orient;
% while ((F'*P*F)>M&&abs(F'*P*F-min_FX_old)>M&&k<50)
% %     while (k<20)
%     min_FX_old = F'*P*F
% %     J = sparse(J);
% %     F = sparse(F);
% %     P = sparse(P);
%     u = 0.001;
%     d = -(J'*P*J+u*I)\(J'*P*F);
%     x_new = x + d;
%     
%     for i = 1 :param.num_step
%         Orient_new(3*(i-1)+1:3*i,:) = Orient(3*(i-1)+1:3*i,:)*expm(skew(d(6*(i-1)+1:6*(i-1)+3)));
%     end
%     
%     
%     F_new = CalculateF_LS(observe,param,x_new,Orient_new,true);
%     min_FX = F_new'*P*F_new;
%     t = 1;
%     while(min_FX > min_FX_old&&t<6)
%         % LM algorithm
%         d = -(J'*P*J+u*I)\(J'*P*F);
%         x_new = x + d;
%         for i = 1 :param.num_step
%             Orient_new(3*(i-1)+1:3*i,:) = Orient(3*(i-1)+1:3*i,:)*expm(skew(d(6*(i-1)+1:6*(i-1)+3)));
%         end
%         F_new = CalculateF_LS(observe,param,x_new,Orient_new,true);
%         min_FX = F_new'*P*F_new;
%         u  = u * 100;
%         t =  t + 1;
%     end
%     if(t==6)
%         break
%     end
%     x = x + d;
%     for i = 1 :param.num_step
%         Orient(3*(i-1)+1:3*i,:) = Orient(3*(i-1)+1:3*i,:)*expm(skew(d(6*(i-1)+1:6*(i-1)+3)));
%     end
%     k=k+1;
%     F = CalculateF_LS(observe,param,x,Orient,true);
%     J = JacobianF_LS(observe,param,x,Orient);
% end
% disp('Gauss-Newton algorithm');
% min_FX = F'*P*F
% iteration=k

% %   Convert state to estimation
% estimation.robotpose   = zeros(param.num_step,3);
% estimation.featurepose = zeros(param.num_features,3);
% for i = 1 : param.num_step
%     estimation.robotpose(i,:) = x(6*(i-1)+4:6*(i-1)+6);
% end
% for i = 1 : param.num_features
%     estimation.featurepose(i,:) = ...
%         x(6*param.num_step+3*(i-1)+1:6*param.num_step+3*i);
% end
% 
% %   Covariance matrix
% cov_mat = inv(J'*J);

