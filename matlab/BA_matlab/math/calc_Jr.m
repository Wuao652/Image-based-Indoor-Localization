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
%æ ¹æ?®p225ï¼›å…¬å¼?.71ï¼?% ç»?è¿‡éªŒè¯?ï¼Œæ”¹äº†ä¸Šé?¢è¿™ä¸ªbugä¹‹å?Žï¼Œr-rä¸?­¥å°±èƒ½æ”¶æ•›äº†ï¼›éš?ä¾¿ç»™åˆ?å§‹å?ï¼?
end

