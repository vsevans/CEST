function [lor] = lorentzian(offsets,pos,fwhm)

lor = (1/pi)*(fwhm/2)./(((offsets-pos).^2)+(fwhm/2).^2);
    
% normalise to 1
lor = lor/max(lor);

end