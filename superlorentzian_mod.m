function [ SL ] = superlorentzian_mod( offsetppms, pos_ppm, T2b_us, B0, interp_ppm, NORMALISE)
%CALCULATESLVALUE Summary of this function goes here

% THIS USED TO BE superlorentzian_mod3, now renamed to preserve
% functionality in other scripts.
% the previous version of superlorentzian_mod was renamed to
% superlorentzian_mod_old

%   Detailed explanation goes here
% trying to fix asymmetry introduced by interpolating 
% NOTE: the interp_ppm variable is the ppm distance from the CENTRE OF THE
% SUPERLORENTZIAN within which the values will be interpolated. It is NOT
% the absolute ppm values of interpolation (unless pos_ppm==0). This must
% be so to allow for symmetrical interpolation - otherwise you get
% aymmetric interpolation results in the central region.

% T2b is provided in microseconds - convert to seconds here
T2b_s = T2b_us*10^-6;

% Separate out inner and outer ppm regions
ppm_inner = offsetppms(abs(offsetppms)<interp_ppm);
ppm_outer = offsetppms(abs(offsetppms)>interp_ppm);

% Convert ppms to Hz
Hz_to_calculate = ppm_outer*128*(B0/3); % for 3T
posHz = pos_ppm*128*(B0/3); % for 3T

% Intialise intermediate output vector for calculated offsets
SL_outer = nan(numel(Hz_to_calculate),1); % stores initial calculated values for outer ppm range (including dummy ppms, which will be dropped before output)

% Calculate SL value for each outer offset
for i = 1:numel(Hz_to_calculate);
    offsetHzi = Hz_to_calculate(i);
    SLfun = @(x,T2b_s,offsetHzi) sqrt(2/pi).*sin(x).*(T2b_s./(abs(3*((cos(x)).^2)-1))).*exp((-2)*(((2*pi*(offsetHzi)*T2b_s)./(abs(3*((cos(x)).^2)-1))).^2));
    q = integral(@(x)SLfun(x,T2b_s,offsetHzi),0,pi/2);
    SL_outer(i) = q;
end;

%interpolate middle bit
SL_all = nan(size(offsetppms));
SL_all_shifted = nan(size(offsetppms));

if numel(ppm_outer)>2 & numel(SL_outer)>2 & numel(offsetppms)>2 & numel(SL_all)>2; % had to put this if clause in to 'trick' the 'fittype' errors into going away
    SL_all = interp1(ppm_outer',SL_outer,offsetppms,'spline');
    %apply offset
    SL_all_shifted = interp1(offsetppms,SL_all,offsetppms-pos_ppm,'pchip'); %pchip is shape preserving and stops weird interpolated values appearing at faraway offsets quite nicely

end;

% return interpolated lineshape
SL = SL_all_shifted;

% normalise to 1;
if NORMALISE == 1;
    SL = SL/max(SL);
end;
% figure, plot(offsetppms,SL)
% hold on;
% plot(offsetppms,SL_all);
% hold off;
% legend('SL','SL-mod');