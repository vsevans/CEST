function [ g ] = gaussian_line_norm( offsets_ppm, pos_ppm, T2_us, B0_T )
%GAUSSIAN_LINE Summary of this function goes here
%   Generates a Gaussian lineshape based on eqn 4 of Morrison and
%   Henkelmann 'A model for magnetization transfer in tissues' 1995

% convert offsets to Hz based on field strength, and apply 'position' to
% shift lineshape
offsets_Hz = (offsets_ppm-pos_ppm)*128*B0_T/3;
offsets_Hz_fine = linspace(min(offsets_Hz),max(offsets_Hz),1001); % for normalization

% convert T2_us to seconds
T2_s = T2_us*10^-6;


g = (T2_s/(sqrt(2*pi)))*exp((-(2*pi*offsets_Hz*T2_s).^2)/2);
g_fine = (T2_s/(sqrt(2*pi)))*exp((-(2*pi*offsets_Hz_fine*T2_s).^2)/2);

% Normalize to 1
g = g/max(g_fine);

end

