% Start assuming the following objects exist in the current workspace:
% (they can be loaded from the Matlab file provided)

% (1) CESTToFitNorm
% Description: Normlized and B0-corrected CEST images from 1 or more files,
% with 1 or more offsets (typically tens of offsets) in ascending order
% (-ve to +ve), 1 or more slices, and 1 or more rows/columns
% Dimensions: [row, col, slice, offsets, files]

% (2) PPMToFit
% Description: A single-dimension vector containing the list of saturation
% offset frequencies expressed in ppm ordered from -ve to +ve
% Dimensions: [Noffsets,1]

% (3) RefDataRaw
% Description: A set of unsaturated reference images.
% Dimensions: The same as CESTToFitNorm but with a size of only 1 in the
% 'file' dimension.

%% For testing:
% Phantom data - 6 falcon tubes containing various concentrations of 
% nicotinamide solution, within a large circular container filled with 
% water. Multiple files, each corresponding to a different saturation power.

%load('/Users/Vincent/Google Drive/MATLAB/Repeatability/Codes_for_Git/PhantomData.mat','CESTToFitNorm','PPMToFit','RefDataRaw');
load('/Your/Directory/Location/.../.../PhantomData.mat','CESTToFitNorm','PPMToFit','RefDataRaw');

CESTToFitNorm = CESTToFitNorm(:,:,:,:,[2,5,8]); % just take data from 3 files for this example
CESTToFitNormFlipped = 1 - CESTToFitNorm;
%% Script parameters (user-specified)
pDrawFitMaskFlag = 1; % Draw a mask on the initial datasets enclosing only the voxels that should be considered for fitting? 0=no, 1=yes
pPropagateFitMasks = 1; % Do you want to automatically propagate the fit-mask from the first file to all subsequent files? (shortcut to avoid having to re-fraw them on every file)? 0=no, 1=yes
pMTEqn = 1; % Which lineshape for the MT pool? 1=Lorentzian, 2=super-Lorentzian, 3=Gaussian, 4=Modified super-Lorentzian, 5=straight line
pDrawROIFlag = 1; % Do you want to do ROI analysis?

%% Script parameters (calculated)
pNumCESTFilesPostB1 = size(CESTToFitNorm,5); % How many CEST files are we about to process?

%% Fitting masks - if using the supplied data, recommend drawing a polygon that encircles all the falcon tubes and excludes the outside of the image
if pDrawFitMaskFlag == 0;
    % If you don't want to draw them skip this step
    fitmasks = ones(size(RefDataRaw,1),size(RefDataRaw,2),size(RefDataRaw,3),size(RefDataRaw,5));
    disp('NOTE: No fit masks applied.');
    
else;
    % For figure closing admin
    if pPropagateFitMasks == 0;
        n_mask_figures = pNumCESTFilesPostB1*size(CESTToFitNorm,3);
    else
        n_mask_figures = size(CESTToFitNorm,3);
    end;
    mask_figure_numbers = [1:n_mask_figures]+1000;
    
    % Draw fit masks
    fitmasks = nan(size(RefDataRaw,1),size(RefDataRaw,2),size(RefDataRaw,3),size(RefDataRaw,5));
    for i = 1:pNumCESTFilesPostB1; % for each CEST dataset
        for j = 1:size(CESTToFitNorm,3); % for each slice
            if i == 1 && j == 1;
                counter = 1;
            else;
                counter = counter+1;
            end;
            
            if pPropagateFitMasks == 0;
                fig_num = mask_figure_numbers(counter);
                RefDataRaw_temp = RefDataRaw(:,:,j,i);
                figure(fig_num),
                imagesc(RefDataRaw_temp);
                title(strcat({'Draw fit mask within slice '},num2str(j),{' (of '},num2str(size(CESTToFitNorm,3)),{') for file '},num2str(i),{' (of '},num2str(pNumCESTFilesPostB1),{')'}));
                aamask=roipoly;
                aamask=double(aamask);
                aamask(aamask==0)=nan;
                fitmasks(:,:,j,i) = aamask;
                RefDataRaw_temp = RefDataRaw_temp.*isnan(aamask);
                imagesc(RefDataRaw_temp);
            else
                if i == 1;
                    fig_num = mask_figure_numbers(counter);
                    
                    RefDataRaw_temp = RefDataRaw(:,:,j,i);
                    figure(fig_num),
                    imagesc(RefDataRaw_temp);
                    title(strcat({'Draw fit mask within slice '},num2str(j),{' (of '},num2str(size(CESTToFitNorm,3)),{') for file '},num2str(i),{' (of '},num2str(pNumCESTFilesPostB1),{')'}));
                    aamask=roipoly;
                    aamask=double(aamask);
                    aamask(aamask==0)=nan;
                    fitmasks(:,:,j,i) = aamask;
                    RefDataRaw_temp = RefDataRaw_temp.*isnan(aamask);
                    imagesc(RefDataRaw_temp);
                else;
                    fitmasks(:,:,j,i) = fitmasks(:,:,1,1);
                    i
                    j
                end;
            end;
        end;
    end;
    % Close all fit mask figures
    close(mask_figure_numbers);
    disp('NOTE: Fit masks created.');
end;

%% Lorentzian Fitting 1: Prepare Fit Parameters
% A: Amplitude (scaling factor: height of Lorentzian)
% x0: Offset (ppm)
% w0: FWHM (ppm)

% 1 - Water
A1min = 0.1;
A1max = 1;
A1initial = 0.8;
w01min = 0.5;
w01max = 12;
w01initial = 1;
x01min = -0.0001;
x01max = 0.0001;
x01initial = 0;

% 2 - Amide
A2min = 0;
A2max = 1;
A2initial = 0.1;
w02min = 0.5;
w02max = 12;
w02initial = 2;
x02min = 2.5;
x02max = 4;
x02initial = 3.5;

% 3 - Amine
A3min = 0;
A3max = 1;
A3initial = 0.1;
w03min = 0.5;
w03max = 12;
w03initial = 1;
x03min = 1.899;
x03max = 1.9;
x03initial = 1.901;

% 4 - Hydroxyl
A4min = 0;
A4max = 1;
A4initial = 0.1;
w04min = 0.5;
w04max = 8;
w04initial = 1;
x04min = 1.5;
x04max = 1.7;
x04initial = 1.6;

% 5 - NOE
A5min = 0;
A5max = 1;
A5initial = 0.1;
w05min = 0.5;
w05max = 12;
w05initial = 2;
x05min = -4;
x05max = -2.5;
x05initial = -3.5;

% 6 - MT
if pMTEqn == 1;
    % if fitting MT as Lorentzian
    A6min = 0.001;
    A6max = 0.7;
    A6initial = 0.1;
    w06min = 15;
    w06max = 100;
    w06initial = 20;
    x06min = -0.5;
    x06max = 0.5;
    x06initial = 0;
elseif pMTEqn == 2;
    % if fitting MT as Super-Lorentzian
    A6min = 0.01;
    A6max = 0.8;
    A6initial = 0.3;
    w06min = 1; %T2b in us
    w06max = 20;
    w06initial = 5.1;
    x06min = -2;
    x06max = 0.01;
    x06initial = 0;
elseif pMTEqn == 3;
    % if fitting MT as gaussian
    A6min = 0.01;
    A6max = 0.8;
    A6initial = 0.2;
    w06min = 1; %T2 in us
    w06max = 10000;
    w06initial = 1000;
    x06min = -2;
    x06max = 0.01;
    x06initial = 0;
elseif pMTEqn == 4;
    % if fitting MT as modified Super-Lorentzian (interpolated between
    % +/-XXppm)
    A6min = 0.001;
    A6max = 1;
    A6initial = 0.5;
    w06min = 1; %T2b in us
    w06max = 50;
    w06initial = 10;
    x06min = -1.27001;
    x06max = -1.26999;
    x06initial = -1.27;
elseif pMTEqn == 5;
    % if fitting a straight line
    A6min = 0.00;
    A6max = 1;
    A6initial = 0.1;
    w06min = 1; % these parameters won't get used
    w06max = 1; % these parameters won't get used
    w06initial = 1; % these parameters won't get used
    x06min = -0.1; % these parameters won't get used
    x06max = -0.01; % these parameters won't get used
    x06initial = 0; % these parameters won't get used
end;

% Vertical offset 'v' - use with caution if you only have data from low ppm
% ranges as the MT-contribution can become confused with a vertical offset 
% if MT-points aren't present.
vmin = -0.05;
vmax = 0.05;
vinitial = 0;

% Horizontal offset 'h'
hmin = -0.3;
hmax = 0.3;
hinitial = 0;

Aallmin = [A1min,A2min,A3min,A4min,A5min,A6min]';
Aallmax = [A1max,A2max,A3max,A4max,A5max,A6max]';
Aallinitial = [A1initial,A2initial,A3initial,A4initial,A5initial,A6initial]';
wallmin = [w01min,w02min,w03min,w04min,w05min,w06min]';
wallmax = [w01max,w02max,w03max,w04max,w05max,w06max]';
wallinitial = [w01initial,w02initial,w03initial,w04initial,w05initial,w06initial]';
xallmin = [x01min,x02min,x03min,x04min,x05min,x06min]';
xallmax = [x01max,x02max,x03max,x04max,x05max,x06max]';
xallinitial = [x01initial,x02initial,x03initial,x04initial,x05initial,x06initial]';

disp('NOTE: Fitting parameters set');

%% Lorentzian Fitting 2: Prepare fit options
% Which pools to include in fit? (0=no, 1=yes)
fitwater = 1;
fitamide = 1;
fitamine = 0;
fithydroxyl = 0;
fitNOE = 1;
fitMT = 1;

fitlist = [fitwater;fitamide;fitamine;fithydroxyl;fitNOE;fitMT];
allpeaknames = {'Water','Amide','Amine','Hydroxyl','NOE','MT'};
NLorFits = numel(fitlist);
fittedpeaknames = allpeaknames(fitlist==1);

water_eqn =     {'A1.*lorentzian_norm(x-h,x01,w01)'};
amide_eqn =     {'A2.*lorentzian_norm(x-h,x02,w02)'};
amine_eqn =     {'A3.*lorentzian_norm(x-h,x03,w03)'};
hydroxyl_eqn =  {'A4.*lorentzian_norm(x-h,x04,w04)'};
NOE_eqn =       {'A5.*lorentzian_norm(x-h,x05,w05)'};
if pMTEqn == 1;
    MT_eqn =    {'A6.*lorentzian_norm(x-h,x06,w06)'};
elseif pMTEqn == 2;
    MT_eqn =    {'A6.*superlorentzian_norm(x-h,x06,w06,3)'};
elseif pMTEqn == 3;
    MT_eqn =    {'A6.*gaussian_line_norm(x-h,x06,w06,3)'};
elseif pMTEqn == 4;
    MT_eqn =    {'A6.*superlorentzian_mod(x-h,x06,w06,3,19,1)'};
end;

all_eqns =      [water_eqn;amide_eqn;amine_eqn;hydroxyl_eqn;NOE_eqn;MT_eqn];
fit_eqns =      all_eqns(fitlist==1);

% Generate full fit equation
fit_eqn = 'v';
for i=1:sum(fitlist);
    fit_eqn = strcat(fit_eqn,'+',fit_eqns(i));
end;

% Set fittype model
ft = fittype( fit_eqn{1,1} , 'independent', 'x', 'dependent', 'y' );

% NOTE: Parameter orderings are as follows:
% All 'A' values in consecutive order
% 'h'
% 'v'
% All 'w' values in consecutive order
% All 'x' values in consecutive order
% 'x' (independent variable)

% Bounds and initial values
lowerBounds = [Aallmin(fitlist==1);hmin;vmin;wallmin(fitlist==1);xallmin(fitlist==1)];
upperBounds = [Aallmax(fitlist==1);hmax;vmax;wallmax(fitlist==1);xallmax(fitlist==1)];
initialVals = [Aallinitial(fitlist==1);hinitial;vinitial;wallinitial(fitlist==1);xallinitial(fitlist==1)];

opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.MaxIter = 100000;
opts.MaxFunEvals = 100000;
opts.Lower = lowerBounds;
opts.Upper = upperBounds;
opts.StartPoint = initialVals;

disp('NOTE: Fit options set');

%% Lorentzian Fitting 3: Initialise matrices
% Initialise fit result matrices
% Lorentzians
fitresultLorA = nan([size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5),NLorFits]);
fitresultLorw = nan([size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5),NLorFits]);
fitresultLorx = nan([size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5),NLorFits]);

% Offsets
fitresulth = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5)); % horizontal offset (x-h)
fitresultv = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5)); % vertical offset (+c)

% Initialise goodness-of-fit (GOF) matrices
gof_sse = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5));
gof_rsquare = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5));
gof_dfe = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5));
gof_adjrsquare = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5));
gof_rmse = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5));

% Initialise lineshape matrices
LorentzianLines = nan([size(CESTToFitNorm),NLorFits]);

% Initialise peak height matrices
peakHeights = nan([size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5),NLorFits]);
peakAreas = nan([size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5),NLorFits]);

% Initialise re-normalisation scaling factor matrix
sfac = nan(size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5));

% Fitted-flag matrix (1 if fit has been done, 0 if not)
fitflag = nan([size(CESTToFitNorm,1),size(CESTToFitNorm,2),size(CESTToFitNorm,3),size(CESTToFitNorm,5)]);

disp('NOTE: Matrices created ready for voxel-wise fitting');
%% Lorentzian Fitting 4: Do the fitting for each voxel

% Suppress warning for removing nans and infs
warning('off','curvefit:prepareFittingData:removingNaNAndInf');
fineoffsetrange = linspace(min(PPMToFit),max(PPMToFit),10001);

% Commence fitting
for b = 1:pNumCESTFilesPostB1;
    for i = 1:size(CESTToFitNorm,1);
        for j = 1:size(CESTToFitNorm,2);
            for k = 1:size(CESTToFitNorm,3);
                if isnan(fitflag(i,j,k,b)) && fitmasks(i,j,k,b)==1;
                    zToFit = double(squeeze(CESTToFitNormFlipped(i,j,k,:,b)));
                    [xData, yData] = prepareCurveData( PPMToFit, zToFit);
                    
                    % Only fit if there are at least as many points as variables
                    if numel(xData) > numel(initialVals);
                        % All points weighted equally
                        weights = ones(numel(xData),1);
                        
                        % Want to use some other weighting? (0=ignore this data point, 1=full-weighting)
                        % weights = zeros(numel(xData),1);
                        % weights(1:5:end-4:end)=1; % only fit MT points
                        
                        opts.Weights = weights;
                        
                        [fitresulti,gofi,outputi] = fit(xData,yData,ft,opts);
                        % Store fit results
                        % For each of the 6 possible Lorentzians
                        for l = 1:NLorFits;
                            if fitlist(l) == 1;
                                A_l = eval(strcat('fitresulti.A',num2str(l)));
                                w_l = eval(strcat('fitresulti.w0',num2str(l)));
                                x_l = eval(strcat('fitresulti.x0',num2str(l)));
                                h_l = fitresulti.h;
                                line_l = A_l*lorentzian_norm(PPMToFit-h_l,x_l,w_l); % NOTE 19th April 2017: this assumes a lorentzian so is not correct for MT - this functionality may be removed soon as it is not as useful as storing the parameters themselves and generating lineshapes later for whatever ppm range we want
                                
                                % Store Lorentzian fit parameters
                                fitresultLorA(i,j,k,b,l) = A_l;
                                fitresultLorw(i,j,k,b,l) = w_l;
                                fitresultLorx(i,j,k,b,l) = x_l;
                                
                                % Store a version of the lineshape for
                                % quick plotting - same ppm range as xData
                                LorentzianLines(i,j,k,:,b,l) = line_l;
                                
                                % Calculate heights and areas using a more
                                % finely sampled Lorentzian to ensure more
                                % accurate height/area measurements
                                if l < 6;
                                    peakHeights(i,j,k,b,l) = max(A_l*lorentzian_norm(fineoffsetrange-h_l,x_l,w_l));
                                    peakAreas(i,j,k,b,l) = trapz(fineoffsetrange,A_l*lorentzian_norm(fineoffsetrange-h_l,x_l,w_l));
                                elseif l == 6;
                                    if pMTEqn == 4;
                                        peakHeights(i,j,k,b,l) = max(A_l*superlorentzian_mod(fineoffsetrange-h_l,x_l,w_l,3,19,1));
                                        peakAreas(i,j,k,b,l) = trapz(fineoffsetrange,A_l*superlorentzian_mod(fineoffsetrange-h_l,x_l,w_l,3,19,1));
                                    end;
                                end;
                            end;
                        end;
                        % For vertical and horizontal offsets
                        fitresulth(i,j,k,b) = fitresulti.h;
                        fitresultv(i,j,k,b) = fitresulti.v;
                        
                        % Store goodness-of-fit (GOF) measures
                        gof_sse(i,j,k,b) = gofi.sse;
                        gof_rsquare(i,j,k,b) = gofi.rsquare;
                        gof_dfe(i,j,k,b) = gofi.dfe;
                        gof_adjrsquare(i,j,k,b) = gofi.adjrsquare;
                        gof_rmse(i,j,k,b) = gofi.rmse;
                        
                    end;
                    fitflag(i,j,k,b) = 1; % Mark this spectrum as having been fitted
                end;
            end;
        end;
        fprintf('NOTE: %s Completed Lorentzian fit for row %d of %d (over all columns and slices), from file %d of %d \n', datestr(datetime('now')),i,size(CESTToFitNormFlipped,1),b,size(CESTToFitNorm,5));
    end;
    %    % Optional: Save data after each file has been processed (useful if
    %    running a very large batch and want to save after each loop to avoid
    %    losing all data if an error occurs)
    
    %    mat_file_location = '/Users/XXX/MATLAB/xxx/xxxx/xxxx.mat');
    %    save(mat_file_location);
end;
disp('NOTE: Voxel-by-voxel fitting is complete');

%% Re-normalise data and fit results
% Re-normalise CESTToFitNorm
CESTToFitRenorm = CESTToFitNorm./repmat((1-reshape(fitresultv,size(fitresultv,1),size(fitresultv,2),size(fitresultv,3),1,size(fitresultv,4))),[1,1,1,numel(PPMToFit),1]);
CESTToFitRenormFlipped = 1 - CESTToFitRenorm;

% Re-normalise fit results
peakHeightsRenorm = peakHeights./repmat((1-fitresultv),[1,1,1,1,NLorFits]);
peakAreasRenorm = peakAreas./repmat((1-fitresultv),[1,1,1,1,NLorFits]);
fitresultLorARenorm = fitresultLorA./repmat(1-reshape(fitresultv,size(fitresultv,1),size(fitresultv,2),size(fitresultv,3),size(fitresultv,4)),[1,1,1,1,NLorFits]);

disp('NOTE: CESTToFitNorm has been re-normalised --> CESTToFitRenorm');
disp('NOTE: CESTToFitNormFlipped has been re-normalised --> CESTToFitRenormFlipped');

%% Generate MTRasym data
MTRasym = flip(CESTToFitRenorm(:,:,:,1:((end-1)/2),:),4)-CESTToFitRenorm(:,:,:,((end-1)/2)+2:end,:);
MTRasym = cat(4,zeros(size(CESTToFitRenorm,1),size(CESTToFitRenorm,2),size(CESTToFitRenorm,3),(size(CESTToFitRenorm,4)+1)/2,pNumCESTFilesPostB1),MTRasym);
disp('NOTE: Generated MTRasym data');

%% Plot parameter histograms
% Which files to include in these histograms?
hist_files = [1:3];

fitresultLorw(fitresultLorw==0)=nan;
fitresultLorx(fitresultLorx==0)=nan;
fitresultLorARenorm(fitresultLorA==0)=nan;

figure,
for i = 1:6;
    subplot(5,6,i);
    histogram(fitresultLorx(:,:,:,hist_files,i)); title(strcat(allpeaknames(i),{ ' offset'}));
    
    subplot(5,6,i+6);
    histogram(fitresultLorw(:,:,:,hist_files,i)); title(strcat(allpeaknames(i),{ ' FWHM/T2b'}));
    
    subplot(5,6,i+12);
    histogram(fitresultLorA(:,:,:,hist_files,i)); title(strcat(allpeaknames(i),{ ' Height'}));
    
    subplot(5,6,i+18);
    histogram(fitresultLorARenorm(:,:,:,hist_files,i)); title(strcat(allpeaknames(i),{ ' Re-normalised Height'}));
    
end;

subplot(5,6,25);
histogram(fitresulth(:,:,:,hist_files)); title('Horizontal offset');
subplot(5,6,26);
histogram(fitresultv(:,:,:,hist_files)); title('Vertical offset');


%% View a particular fit result
% Specify which z-spectrum you want to see?
ii = 22; % row
jj = 42; % column
slice = 1; % slice
corefile = 3; % file

A1_renorm = fitresultLorARenorm(ii,jj,slice,corefile,1);
A2_renorm = fitresultLorARenorm(ii,jj,slice,corefile,2);
A3_renorm = fitresultLorARenorm(ii,jj,slice,corefile,3);
A4_renorm = fitresultLorARenorm(ii,jj,slice,corefile,4);
A5_renorm = fitresultLorARenorm(ii,jj,slice,corefile,5);
A6_renorm = fitresultLorARenorm(ii,jj,slice,corefile,6);
w01 = fitresultLorw(ii,jj,slice,corefile,1);
w02 = fitresultLorw(ii,jj,slice,corefile,2);
w03 = fitresultLorw(ii,jj,slice,corefile,3);
w04 = fitresultLorw(ii,jj,slice,corefile,4);
w05 = fitresultLorw(ii,jj,slice,corefile,5);
w06 = fitresultLorw(ii,jj,slice,corefile,6);
x01 = fitresultLorx(ii,jj,slice,corefile,1);
x02 = fitresultLorx(ii,jj,slice,corefile,2);
x03 = fitresultLorx(ii,jj,slice,corefile,3);
x04 = fitresultLorx(ii,jj,slice,corefile,4);
x05 = fitresultLorx(ii,jj,slice,corefile,5);
x06 = fitresultLorx(ii,jj,slice,corefile,6);
h = fitresulth(ii,jj,slice,corefile);

% Generate re-normalised peaks from parameters
water_renorm = A1_renorm*lorentzian_norm(PPMToFit-h,x01,w01);
amide_renorm = A2_renorm*lorentzian_norm(PPMToFit-h,x02,w02);
amine_renorm = A3_renorm*lorentzian_norm(PPMToFit-h,x03,w03);
hydroxyl_renorm = A4_renorm*lorentzian_norm(PPMToFit-h,x04,w04);
NOE_renorm = A5_renorm*lorentzian_norm(PPMToFit-h,x05,w05);

if pMTEqn == 1;
    MT_renorm = A6_renorm*lorentzian_norm(PPMToFit-h,x06,w06);
elseif pMTEqn == 2;
    MT_renorm = A6_renorm.*superlorentzian_norm(PPMToFit-h,x06,w06,3);
elseif pMTEqn == 3;
    MT_renorm = A6_renorm.*gaussian_line(PPMToFit-h,x06,w06,3);
elseif pMTEqn == 4;
    MT_renorm = A6_renorm.*superlorentzian_mod(PPMToFit-h,x06,w06,3,19,1);
end;

% Prep summed fit, vertical offset, residuals and data points
summed_fit_renorm = water_renorm;
if fitlist(2) == 1;
    summed_fit_renorm = summed_fit_renorm+amide_renorm;
end;
if fitlist(3) == 1;
    summed_fit_renorm = summed_fit_renorm+amine_renorm;
end;
if fitlist(4) == 1;
    summed_fit_renorm = summed_fit_renorm+hydroxyl_renorm;
end;
if fitlist(5) == 1;
    summed_fit_renorm = summed_fit_renorm+NOE_renorm;
end;
if fitlist(6) == 1;
    summed_fit_renorm = summed_fit_renorm+MT_renorm;
end;
data_points_renorm = squeeze(CESTToFitRenormFlipped(ii,jj,slice,:,corefile));
residuals_renorm = data_points_renorm-summed_fit_renorm;

plot_legend = allpeaknames(fitlist==1);

MTRasym_voxel = squeeze(MTRasym(ii,jj,slice,:,corefile));

lw = 1.2;
% Plot fit results
figure,
subplot(8,5,1:30); % plot data points
plot(PPMToFit,1-water_renorm,'LineWidth',lw);
if fitlist(2) == 1;
    hold on;
    plot(PPMToFit,1-amide_renorm,'LineWidth',lw);
end;
if fitlist(3) == 1;
    hold on;
    plot(PPMToFit,1-amine_renorm,'LineWidth',lw);
end;
if fitlist(4) == 1;
    hold on;
    plot(PPMToFit,1-hydroxyl_renorm,'LineWidth',lw);
end;
if fitlist(5) == 1;
    hold on;
    plot(PPMToFit,1-NOE_renorm,'LineWidth',lw);
end;
if fitlist(6) == 1;
    hold on;
    plot(PPMToFit,1-MT_renorm,'LineWidth',lw);
end;
hold on;
plot(PPMToFit,1-summed_fit_renorm,'LineWidth',lw);
hold on;
plot(PPMToFit,MTRasym_voxel,'--','LineWidth',lw);
hold on;
plot(PPMToFit,1-data_points_renorm,'k+','LineWidth',1);
legend([plot_legend,'Summed fit','MTR_a_s_y_m','Data-points']);
axis([min(PPMToFit),max(PPMToFit),-0.2,1]);
axis([-6,6,-0.1,1]);
title(strcat({'Re-normalised fit result for ii='},num2str(ii),{', jj='},num2str(jj),{', slice='},num2str(slice),{', file='},num2str(corefile)));

grid on;
set(gca, 'XDir','reverse')
xlabel('Offset frequency (ppm)');
ylabel('M_z');
subplot(8,5,36:40); % plot residuals
plot(PPMToFit,residuals_renorm,'LineWidth',lw);
axis([min(PPMToFit),max(PPMToFit),-0.1,0.1]);
axis([-6,6,-0.1,0.1]);
grid on;
set(gca, 'XDir','reverse')
title('Fit residuals');
xlabel('Offset frequency (ppm)');

%% Plot parameter maps
peaknum = 2; % 1 = water, 2 = amide, 3 = amine, 4 = hydroxyl, 5 = NOE, 6 = MT
slice = 1; % slice
corefile = 2; % file
peak_name = allpeaknames(peaknum);

figure,
subplot(3,2,1);
imagesc(fitresultLorARenorm(:,:,slice,corefile,peaknum)); colorbar; % caxis([0.1,0.4]);
title(strcat(peak_name,{' height'}));
subplot(3,2,2);
imagesc(fitresultLorx(:,:,slice,corefile,peaknum)); colorbar;
title(strcat(peak_name,{' offset'}));
subplot(3,2,3);
imagesc(fitresultLorw(:,:,slice,corefile,peaknum)); colorbar;
title(strcat(peak_name,{' FWHM'}));
subplot(3,2,4);
imagesc(fitresultv(:,:,slice,corefile)); colorbar;
title('Vertical offset');
subplot(3,2,5);
imagesc(fitresulth(:,:,slice,corefile)); colorbar;
title('Horizontal offset');

%% Generate MTRasym maps for a given ppm range
corefile = 3;
slice = 1;
% Specify integration range. Values must exist in PPMToFit. If a range is
% specified, trapz will be used to integrate the area under the MTRasym
% curve. If ppmmin == ppmmax, a single offset will be used.

ppmmin = 3; % minimum ppm offset for your MTRasym calculation
ppmmax = 4; % maximum ppm offset for your MTRasym calculation
integration_ppms = PPMToFit(PPMToFit >= ppmmin);
MTRasym_to_integrate = MTRasym(:,:,:,PPMToFit >= ppmmin,:);

integration_ppms = integration_ppms(integration_ppms <= ppmmax);
MTRasym_to_integrate = MTRasym_to_integrate(:,:,:,integration_ppms <= ppmmax,:);

num_int_ppms = numel(integration_ppms);


if ppmmin == ppmmax;
    MTRasym_range = MTRasym(:,:,slice,PPMToFit==ppmmin,:);
elseif ppmmin < ppmmax;
    MTRasym_range = trapz(integration_ppms,MTRasym_to_integrate(:,:,:,:,:),4);
else;
    error('ERROR: ppmmin is greater than ppmmax, check specified ppm range.');
end;

figure,
imagesc(MTRasym_range(:,:,slice,1,corefile));
colorbar;
caxis([-0.05 0.3]);
title('MTRasym');

%% Prep matrix for ROI masks

% EXAMPLE DATA:
%  Draw ROI1 in the falcon tube at 2 o'clock in the image.
%  Subsequent ROIs going clock-wise until midnight, then the final in the
%  central tube. This is the order of increasing nicotinamide
%  concentration.

pnROIs_per_file = 3;
ROImasks_justdrawn = zeros(size(RefDataRaw,1),size(RefDataRaw,2),size(RefDataRaw,3),size(RefDataRaw,5),pnROIs_per_file);

%% Draw ROI on RefData
if pDrawROIFlag == 1;
    for i = 1:size(CESTToFitNorm,5); % for each CEST dataset
        for j = 1:size(CESTToFitNorm,3); % for each slice
            RefDataRaw_temp = RefDataRaw(:,:,j,1,i);
            figure,
            for r = 1:pnROIs_per_file; % for each ROI
                if r == 1;
                    imagesc(RefDataRaw_temp); caxis([min(min(RefDataRaw_temp)), max(max(RefDataRaw_temp))]); colorbar;
                end;
                title(strcat({'Draw ROI '},num2str(r),{' (of '},num2str(pnROIs_per_file),{') within slice '},num2str(j),{' (of '},num2str(size(CESTToFitNorm,3)),{') for file '},num2str(i),{' (of '},num2str(size(CESTToFitNorm,5)),{')'}));
                aa=roipoly;
                aa=double(aa);
                aa(aa==0)=nan;
                ROImasks_justdrawn(:,:,j,i,r) = aa;
                RefDataRaw_temp = RefDataRaw_temp.*isnan(aa);
                imagesc(RefDataRaw_temp);
            end;
        end;
    end;
end;

ROImasks = ROImasks_justdrawn;


%% Generate z-spectra for ROI
ROIData = zeros([numel(PPMToFit), size(CESTToFitNorm,3), pNumCESTFilesPostB1, pnROIs_per_file]); % ppms / slices / files
ROIstd = zeros([numel(PPMToFit), size(CESTToFitNorm,3), pNumCESTFilesPostB1, pnROIs_per_file]);

for b = 1:pNumCESTFilesPostB1; % files
    for s = 1:size(CESTToFitNorm,3); % slices
        for c = 1:pnROIs_per_file; % ROIs
            for z = 1:numel(PPMToFit); % ppms
                % Prep
                ROIData_bsci = double(squeeze(CESTToFitRenorm(:,:,s,z,b)).*ROImasks(:,:,s,b,c));
                
                % Output
                ROIData(z,s,b,c) = double(nanmean(nanmean(ROIData_bsci)));
                ROIstd(z,s,b,c) = double(std(std(ROIData_bsci,'omitnan'),'omitnan'));
                
            end;
        end;
    end;
end;
disp('NOTE: Generated z-spectra for ROIs');


%% Flip ROI data for fitting
ROINormFlipped = 1 - ROIData;

%% Lorentzian ROI Fit 1: Prepare Fit Parameters
% already covered in voxel-by-voxel code above!

%% Lorentzian ROI Fit 2: Prepare fit options
% already covered in voxel-by-voxel code above!

%% Lorentzian ROI Fit 3: Initialise ROI matrices
% Initialise fit result matrices
% Lorentzians
fitresultROILorA = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4),NLorFits);
fitresultROILorw = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4),NLorFits);
fitresultROILorx = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4),NLorFits);
% Offsets
fitresultROIh = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));
fitresultROIv = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));

% Initialise GOF matrices
gofROI_sse = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));
gofROI_rsquare = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));
gofROI_dfe = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));
gofROI_adjrsquare = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));
gofROI_rmse = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));

% Initialise lineshape matrices
LorentzianLinesROI = zeros([size(ROINormFlipped),NLorFits]);

% Initialise peak height matrices
peakHeightsROI = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4),NLorFits);
peakAreasROI = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4),NLorFits);

% Initialise re-normalisation scaling factor matrix
sfacROI = nan(size(ROINormFlipped,2),size(ROINormFlipped,3),size(ROINormFlipped,4));

%% Lorentzian Fitting 4: Do the fitting for each ROI
% Suppress warning for removing nans and infs
warning('off','curvefit:prepareFittingData:removingNaNAndInf');

ppmfineforheightLOR = linspace(-5,5,1001); % for calculating peak height
ppmfineforheightMTmod = [PPMToFit(PPMToFit<=-10);linspace(-5,5,1001)';PPMToFit(PPMToFit>=10)]; % for calculating peak height

% Commence fitting
for b = 1:pNumCESTFilesPostB1; % files
    for k = 1:size(ROINormFlipped,2); % slices
        for r = 1:size(ROINormFlipped,4); % ROIs
            zToFit = squeeze(ROINormFlipped(:,k,b,r));
            [xData, yData] = prepareCurveData( PPMToFit, zToFit);
            % Only fit if there are at least as many points as variables
            if numel(xData) > numel(initialVals);
                opts.Weights = ones(numel(xData),1);
                [fitresulti,gofi,outputi] = fit(xData,yData,ft,opts);
                % Store fit results
                % For each of the 6 possible Lorentzians
                for l = 1:NLorFits;
                    if fitlist(l) == 1;
                        A_l = eval(strcat('fitresulti.A',num2str(l)));
                        w_l = eval(strcat('fitresulti.w0',num2str(l)));
                        x_l = eval(strcat('fitresulti.x0',num2str(l)));
                        line_l = A_l*lorentzian_norm(ppmfineforheightLOR,x_l,w_l);
                        fitresultROILorA(k,b,r,l) = A_l;
                        fitresultROILorw(k,b,r,l) = w_l;
                        fitresultROILorx(k,b,r,l) = x_l;
                        %LorentzianLinesROI(:,k,b,r,l) = line_l;
                        if l < 6;
                            peakHeightsROI(k,b,r,l) = max(line_l);
                        end;
                        
                        if l == 6; % if we fitted a modified superlorentzian, overwrite the previously calculated peak height, which was calculated as a lorentzian
                            if pMTEqn == 4;
                                line_MT_i = A_l*superlorentzian_mod(ppmfineforheightMTmod,x_l,w_l,3,19,1);
                                peakHeightsROI(k,b,r,l) = max(line_MT_i);
                            end;
                        end;
                    end;
                end;
                % For vertical and horizontal offsets
                fitresultROIh(k,b,r) = fitresulti.h;
                fitresultROIv(k,b,r) = fitresulti.v;
                
                % GOF measures
                gofROI_sse(k,b,r) = gofi.sse;
                gofROI_rsquare(k,b,r) = gofi.rsquare;
                gofROI_dfe(k,b,r) = gofi.dfe;
                gofROI_adjrsquare(k,b,r) = gofi.adjrsquare;
                gofROI_rmse(k,b,r) = gofi.rmse;
                
            end;
            fprintf('NOTE: Completed Lorentzian fit for ROI %d of %d in slice %d of %d, from file %d of %d \n', r, pnROIs_per_file, k, size(ROINormFlipped,2),b,pNumCESTFilesPostB1);
        end;
    end;
end;

% Normalise fit results
peakHeightsROINorm = peakHeightsROI./repmat((1-fitresultROIv),[1,1,1,6]);
fitresultROILorARenorm = fitresultROILorA./repmat(1-fitresultROIv,[1,1,1,6]);

% Re-normalise ROIData
ROIDataRenorm = ROIData./repmat((1-reshape(fitresultROIv,1,size(fitresultROIv,1),size(fitresultROIv,2),size(fitresultROIv,3))),[numel(PPMToFit),1,1,1]);
ROIDataRenormFlipped = 1 - ROIDataRenorm;

disp('NOTE: ROIData has been re-normalised --> ROIDataRenorm');
disp('NOTE: ROINormFlipped has been re-normalised --> ROIRenormFlipped');
disp('NOTE: Voxel-by-voxel fitting is complete');

%% Generate MTRasym for ROI
ppmmin = 3;
ppmmax = 4;

MTRasymROI = flip(ROIData(1:((end-1)/2),:,:,:),1)-ROIData(((end-1)/2)+2:end,:,:,:);
MTRasymROI = cat(1,zeros((size(ROIData,1)+1)/2,size(ROIData,2),pNumCESTFilesPostB1,pnROIs_per_file),MTRasymROI);
disp('NOTE: Generated MTRasym data for ROIs');

if ppmmin == ppmmax;
    MTRasym_range_ROI = MTRasymROI(PPMToFit==ppmmin,:,:,:);
elseif ppmmin < ppmmax;
    MTRasym_range_ROI = trapz(PPMToFit(PPMToFit>=ppmmin & PPMToFit<=ppmmax),MTRasymROI(PPMToFit>=ppmmin & PPMToFit<=ppmmax,:,:,:),1);
else;
    error('ERROR: ppmmin is greater than ppmmax, check specified ppm range.');
end;

%% View a particular fit result ROI (re-normalised)
sliceROI = 1;
corefileROI = 3;
ROI = 2;

% Fetch renormalised parameters
A1ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,1);
A2ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,2);
A3ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,3);
A4ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,4);
A5ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,5);
A6ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,6);

w01ROI = fitresultROILorw(sliceROI,corefileROI,ROI,1);
w02ROI = fitresultROILorw(sliceROI,corefileROI,ROI,2);
w03ROI = fitresultROILorw(sliceROI,corefileROI,ROI,3);
w04ROI = fitresultROILorw(sliceROI,corefileROI,ROI,4);
w05ROI = fitresultROILorw(sliceROI,corefileROI,ROI,5);
w06ROI = fitresultROILorw(sliceROI,corefileROI,ROI,6);
x01ROI = fitresultROILorx(sliceROI,corefileROI,ROI,1);
x02ROI = fitresultROILorx(sliceROI,corefileROI,ROI,2);
x03ROI = fitresultROILorx(sliceROI,corefileROI,ROI,3);
x04ROI = fitresultROILorx(sliceROI,corefileROI,ROI,4);
x05ROI = fitresultROILorx(sliceROI,corefileROI,ROI,5);
x06ROI = fitresultROILorx(sliceROI,corefileROI,ROI,6);
hROI = fitresultROIh(sliceROI,corefileROI,ROI);
vROI = fitresultROIv(sliceROI,corefileROI,ROI);

% Create lineshapes
waterROI = A1ROI*lorentzian_norm(PPMToFit-hROI,x01ROI,w01ROI);
amideROI = A2ROI*lorentzian_norm(PPMToFit-hROI,x02ROI,w02ROI);
amineROI = A3ROI*lorentzian_norm(PPMToFit-hROI,x03ROI,w03ROI);
hydroxylROI = A4ROI*lorentzian_norm(PPMToFit-hROI,x04ROI,w04ROI);
NOEROI = A5ROI*lorentzian_norm(PPMToFit-hROI,x05ROI,w05ROI);
if pMTEqn == 1;
    MTROI = A6ROI*lorentzian_norm(PPMToFit-hROI,x06ROI,w06ROI);
elseif pMTEqn == 2;
    MTROI = A6ROI.*superlorentzian_norm(PPMToFit-hROI,x06ROI,w06ROI,3);
elseif pMTEqn == 3;
    MTROI = A6ROI.*gaussian_line(PPMToFit-hROI,x06ROI,w06ROI,3);
elseif pMTEqn == 4;
    MTROI = A6ROI.*superlorentzian_mod(PPMToFit-hROI,x06ROI,w06ROI,3,19,1);
end;


% Plot fit results
plot_legend = allpeaknames(fitlist==1);

summed_fit_ROI = waterROI;
if fitlist(2) == 1;
    summed_fit_ROI = summed_fit_ROI+amideROI;
end;
if fitlist(3) == 1;
    summed_fit_ROI = summed_fit_ROI+amineROI;
end;
if fitlist(4) == 1;
    summed_fit_ROI = summed_fit_ROI+hydroxylROI;
end;
if fitlist(5) == 1;
    summed_fit_ROI = summed_fit_ROI+NOEROI;
end;
if fitlist(6) == 1;
    summed_fit_ROI = summed_fit_ROI+MTROI;
end;

raw_data_ROI = squeeze(ROIDataRenormFlipped(:,sliceROI,corefileROI,ROI));
residuals = raw_data_ROI-summed_fit_ROI;
lw = 1.2;

MTRasym_range_ROI_voxel = squeeze(MTRasymROI(:,:,corefileROI,ROI));

% Individual lineshapes
figure,
subplot(8,5,1:30);
plot(PPMToFit,1-waterROI,'LineWidth',lw);
if fitlist(2) == 1;
    hold on;
    plot(PPMToFit,1-amideROI,'LineWidth',lw);
end;
if fitlist(3) == 1;
    hold on;
    plot(PPMToFit,1-amineROI,'LineWidth',lw);
end;
if fitlist(4) == 1;
    hold on;
    plot(PPMToFit,1-hydroxylROI,'LineWidth',lw);
end;
if fitlist(5) == 1;
    hold on;
    plot(PPMToFit,1-NOEROI,'LineWidth',lw);
end;
if fitlist(6) == 1;
    hold on;
    plot(PPMToFit,1-MTROI,'LineWidth',lw);
end;
hold on;
plot(PPMToFit,1-summed_fit_ROI,'-','LineWidth',lw);
hold on;
plot(PPMToFit, MTRasym_range_ROI_voxel,'--','LineWidth',lw);

hold on;
plot(PPMToFit,1-raw_data_ROI,'+','LineWidth',2);
hold off;
legend([plot_legend,'Summed fit','MTR_a_s_y_m','Data-points']);
axis([-7,7,-0.2,1]);
title(strcat({'Normalised fit result for ROI '},num2str(ROI),{' from file '},num2str(corefileROI)));
set(gca, 'XDir','reverse')
xlabel('Offset frequency (ppm)');
ylabel('M_z');
grid on;

subplot(8,5,36:40);
plot(PPMToFit,residuals,'LineWidth',lw);
axis([-7,7,-0.1,0.1]);
set(gca, 'XDir','reverse');
xlabel('Offset frequency (ppm)');
grid on;
title('Fit Residuals');
%% Plot representative fits from selected ROIs (up to 3 ROIs)
file = 1;
ROInums = [1,2,3];
MTRscaling = 5; % 1 = do not scale the MTRasym

fontsize = 10;
plot_legend = allpeaknames(fitlist==1);

figure,
for i = 1:numel(ROInums);
    sliceROI = 1;
    corefileROI = file;
    ROI = ROInums(i);
    
    % Fetch parameters
    % Not renormalised
%     A1ROI = fitresultROILorA(sliceROI,corefileROI,ROI,1);
%     A2ROI = fitresultROILorA(sliceROI,corefileROI,ROI,2);
%     A3ROI = fitresultROILorA(sliceROI,corefileROI,ROI,3);
%     A4ROI = fitresultROILorA(sliceROI,corefileROI,ROI,4);
%     A5ROI = fitresultROILorA(sliceROI,corefileROI,ROI,5);
%     A6ROI = fitresultROILorA(sliceROI,corefileROI,ROI,6);

    % Renormalised
    A1ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,1);
    A2ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,2);
    A3ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,3);
    A4ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,4);
    A5ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,5);
    A6ROI = fitresultROILorARenorm(sliceROI,corefileROI,ROI,6);
    
    w01ROI = fitresultROILorw(sliceROI,corefileROI,ROI,1);
    w02ROI = fitresultROILorw(sliceROI,corefileROI,ROI,2);
    w03ROI = fitresultROILorw(sliceROI,corefileROI,ROI,3);
    w04ROI = fitresultROILorw(sliceROI,corefileROI,ROI,4);
    w05ROI = fitresultROILorw(sliceROI,corefileROI,ROI,5);
    w06ROI = fitresultROILorw(sliceROI,corefileROI,ROI,6);
    x01ROI = fitresultROILorx(sliceROI,corefileROI,ROI,1);
    x02ROI = fitresultROILorx(sliceROI,corefileROI,ROI,2);
    x03ROI = fitresultROILorx(sliceROI,corefileROI,ROI,3);
    x04ROI = fitresultROILorx(sliceROI,corefileROI,ROI,4);
    x05ROI = fitresultROILorx(sliceROI,corefileROI,ROI,5);
    x06ROI = fitresultROILorx(sliceROI,corefileROI,ROI,6);
    hROI = fitresultROIh(sliceROI,corefileROI,ROI);
    vROI = fitresultROIv(sliceROI,corefileROI,ROI);
    
    % Create lineshapes
    waterROI = A1ROI*lorentzian_norm(PPMToFit-hROI,x01ROI,w01ROI);
    amideROI = A2ROI*lorentzian_norm(PPMToFit-hROI,x02ROI,w02ROI);
    amineROI = A3ROI*lorentzian_norm(PPMToFit-hROI,x03ROI,w03ROI);
    hydroxylROI = A4ROI*lorentzian_norm(PPMToFit-hROI,x04ROI,w04ROI);
    NOEROI = A5ROI*lorentzian_norm(PPMToFit-hROI,x05ROI,w05ROI);
    if pMTEqn == 1;
        MTROI = A6ROI*lorentzian_norm(PPMToFit-hROI,x06ROI,w06ROI);
    elseif pMTEqn == 2;
        MTROI = A6ROI.*superlorentzian_norm(PPMToFit-hROI,x06ROI,w06ROI,3);
    elseif pMTEqn == 3;
        MTROI = A6ROI.*gaussian_line(PPMToFit-hROI,x06ROI,w06ROI,3);
    elseif pMTEqn == 4;
        MTROI = A6ROI.*superlorentzian_mod(PPMToFit-hROI,x06ROI,w06ROI,3,19,1);
    end;
    
    % MTR of this ROI
    MTR_i = MTRasymROI(:,1,corefileROI,ROI);
    MTR_i(1:32)=nan;
    
    % Plot fit results
    summed_fit_ROI = waterROI;
    if fitlist(2) == 1;
        summed_fit_ROI = summed_fit_ROI+amideROI;
    end;
    if fitlist(3) == 1;
        summed_fit_ROI = summed_fit_ROI+amineROI;
    end;
    if fitlist(4) == 1;
        summed_fit_ROI = summed_fit_ROI+hydroxylROI;
    end;
    if fitlist(5) == 1;
        summed_fit_ROI = summed_fit_ROI+NOEROI;
    end;
    if fitlist(6) == 1;
        summed_fit_ROI = summed_fit_ROI+MTROI;
    end;
    
    raw_data_ROI = squeeze(ROIDataRenormFlipped(:,sliceROI,corefileROI,ROI));
    residuals = raw_data_ROI-summed_fit_ROI;
    
    % Individual lineshapes
    subplot(27,5,[1:20]+(i-1)*35);
    plot(PPMToFit,1-waterROI);
    if fitlist(2) == 1;
        hold on;
        plot(PPMToFit,1-amideROI);
    end;
    if fitlist(3) == 1;
        hold on;
        plot(PPMToFit,1-amineROI);
    end;
    if fitlist(4) == 1;
        hold on;
        plot(PPMToFit,1-hydroxylROI);
    end;
    if fitlist(5) == 1;
        hold on;
        plot(PPMToFit,1-NOEROI);
    end;
    if fitlist(6) == 1;
        hold on;
        plot(PPMToFit,1-MTROI);
    end;
    
    hold on;
    plot(PPMToFit,1-summed_fit_ROI);
    hold on;
    plot(PPMToFit,MTRscaling*MTR_i,'--');
    hold on;
    plot(PPMToFit,1-raw_data_ROI,'+','LineWidth',1.5);
    hold off;
    set(gca, 'XDir','reverse')
    xticks([-7:7]);
    yticks([0,0.2,0.4,0.6,0.8,1]);
    
    grid on;
    ylabel('M_z','FontSize',10);
    if i == 1;
        lgd = legend([plot_legend,'Summed fit',strcat({'MTR_a_s_y_m x '},num2str(MTRscaling)),'Data-points']);
    end;
    
    
    axis([-7.5,7.5,-0.2,1]);
    if i == 1;
        % Title of first plot
        title('Title of first plot','FontSize',10);
    elseif i == 2;
        % Title of second plot
        title('And second','FontSize',10);
    elseif i == 3;
        % Title of third plot etc...
        title('And third','FontSize',10);
    end;
    g = gca;
    g.FontSize = fontsize;
    
    subplot(27,5,(26:30)+(i-1)*35);
    plot(PPMToFit,residuals,'LineWidth',1);
    axis([-7.5,7.5,-0.2,0.2]);
    xticks([-7:7]);
    %yticks([-0.4,0,0.4]);
    set(gca, 'XDir','reverse')
    title('Fit residuals');
    if i == 3;
        xlabel('Offset frequency (ppm)','FontSize',10);
    end;
    g = gca;
    g.FontSize = fontsize;
end;

CapWidth = 0.9;
CapHorPos = (1-CapWidth)/2;
CapVertPos = 0.1;
CapHeight = (1.2/8)-CapVertPos;

t = annotation('textbox',[CapHorPos CapVertPos CapWidth CapHeight],'String','FIGURE caption: Add your text here.', 'FontSize',10,'FontWeight','bold','EdgeColor','none');

%% Plot fitted peak heights for all z-spectra from each ROI in a given file/slice
sliceROI = 1;
corefileROI = 1;
ROI = [1:3];
peakROI = 2; % 1 = water, 2 = amide, 3 = amine, 4 = hydroxyl, 5 = NOE, 6 = MT

figure,
plot(ROI,squeeze(peakHeightsROINorm(sliceROI,corefileROI,ROI,peakROI)),'+','LineWidth',2);
title('Peak heights for each ROI');
xlabel('ROI');
ylabel('M_z');

%% Plot fitted peak heights for averaged z-spectra from each file/slice for a specified ROI
sliceROI = 1;
corefileROI = [1:3];
ROI = 1;
peakROI = 2; % 1 = water, 2 = amide, 3 = amine, 4 = hydroxyl, 5 = NOE, 6 = MT

figure,
plot(corefileROI,squeeze(peakHeightsROINorm(sliceROI,corefileROI,ROI,peakROI)),'+','LineWidth',2);
title(strcat({'Peak heights for each file from ROI '},num2str(ROI)));
xlabel('File');
ylabel('M_z');


%% Plot z-spectra and MTRasym curves for ROIs
corefile = 1;
slice = 1;

figure,
for i = 1:pnROIs_per_file;
    errorbar(PPMToFit,squeeze(ROIData(:,slice,corefile,i)),squeeze(ROIstd(:,slice,corefile,i)));
    hold on;
end;
plot(PPMToFit,squeeze(MTRasymROI(:,slice,corefile,:)));
hold off;
grid on;
axis([-5,5,-0.2,1]);
title('Z-spectra for all ROIs'); axis([-5,5,-0.2,1]);
legend('ROI1 name','ROI2 name','etc','etc','etc');
set(gca, 'XDir','reverse')
