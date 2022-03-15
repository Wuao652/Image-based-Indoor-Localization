function [Jl,phiv] = calc_Jl(phi)


% Modified by Jingwei
if(norm(phi)==0)
    phi = [1 1 1];
end


a = phi'/norm(phi);
phiv = norm(phi);
%Jl = sin(phiv)/phiv*eye(3,3)+(1-sin(phiv))/cos(phiv)*a*a'+...
%    (1-cos(phiv))/phiv*LieGroup.getskew(a);
% edit 2017-12-8;
Jl = sin(phiv)/phiv*eye(3,3)+(1-sin(phiv)/phiv)*a*a'+...
    (1-cos(phiv))/phiv*getskew(a);
%根据p225；公�?.71�?% 经过验证，改了上面这个bug之后，r-r�?��就能收敛了；随便给初始�?�?
end

