function [Jl,phiv] = calc_Jr(phi)


% Modified by Jingwei
if(norm(phi)==0)
    phi = [1 1 1];
end


a = phi'/norm(phi);
phiv = norm(phi);
%Jl = sin(phiv)/phiv*eye(3,3)+(1-sin(phiv))/cos(phiv)*a*a'+...
%    (1-cos(phiv))/phiv*LieGroup.getskew(a);
% edit 2017-12-8;
Jl = sin(phiv)/phiv*eye(3,3)+(1-sin(phiv)/phiv)*a*a'-...
    (1-cos(phiv))/phiv*getskew(a);
%根�?�p225；公�?.71�?% �?过验�?，改了上�?�这个bug之�?�，r-r�?��就能收敛了；�?便给�?始�?�?
end

