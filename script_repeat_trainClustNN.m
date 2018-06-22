clear; clc
%% params
numRepeats = 10;

Nc = 1;
if Nc == 10
    maxIters = 1000;
elseif Nc ==1
    maxIters = 5000;
else
    error('Specify maxIters for given Nc')
end

%% init
thres_f_avg = zeros(1,maxIters+1);
errTrain_avg = zeros(1,maxIters+1);
errTest_avg = zeros(1,maxIters+1);
fRaw_avg = zeros(1,maxIters);
fEff_avg = zeros(1,maxIters);
fCalc_avg = zeros(1,maxIters);
thetaWnorm_avg = zeros(1,maxIters+1);
deltaThetaW_mag_avg = zeros(1,maxIters);
deltaThetaW_corr_avg = zeros(maxIters);

%% run (ENSURE THERE ARE NO LOOPS AND NO PLOTTING IN script_trainClustNN.m !!!)
for repeat = 1:numRepeats
    script_trainClustNN
    
    thres_f_avg = thres_f_avg + thres_f;
    errTrain_avg = errTrain_avg + errorTrain;
    errTest_avg = errTest_avg + errorTest;
    fRaw_avg = fRaw_avg + fraw;
    fEff_avg = fEff_avg + fthres_eff;
    fCalc_avg = fCalc_avg + fthres;
    thetaWnorm_avg = thetaWnorm_avg + thetaW_norm;
    deltaThetaW_mag_avg = deltaThetaW_mag_avg + deltaThetaW_mag;
    deltaThetaW_corr_avg = deltaThetaW_corr_avg + deltaThetaW_corr;
end
thres_f_avg = thres_f_avg/numRepeats;
errTrain_avg = errTrain_avg/numRepeats;
errTest_avg = errTest_avg/numRepeats;
fRaw_avg = fRaw_avg/numRepeats;
fEff_avg = fEff_avg/numRepeats;
fCalc_avg = fCalc_avg/numRepeats;
thetaWnorm_avg = thetaWnorm_avg/numRepeats;
deltaThetaW_mag_avg = deltaThetaW_mag_avg/numRepeats;
deltaThetaW_corr_avg = deltaThetaW_corr_avg/numRepeats;

resultStr = sprintf('AVERAGE E_{train}=%g, E_{test}=%g', errTrain_avg(end), errTest_avg(end));

%% plot results
set(0,'defaultAxesXLimSpec', 'tight')
fig=figure;

subplot(3,2,1)
plot(thres_f_avg)
ylabel('Thres')
title(sprintf('AVG Thres to make f=%g',f))

subplot(3,2,2)
plot(computeTheseErrs, errTrain_avg(computeTheseErrs))
hold on
plot(computeTheseErrs, errTest_avg(computeTheseErrs))
ylabel('Error')
legend({'Train', 'Test'})
title(sprintf('%s %s\n%s Batch=%d',paramsStr, miscStrT, resultStr, batchSize))

subplot(3,2,3);
plot(fEff_avg); hold on
plot(fCalc_avg)
plot(fRaw_avg)
plot(f*ones(size(fthres)), 'k--')
title('AVG Fraction of nonzero dldW')
ylabel('f')
legend({'Effective', 'Calculated', 'No threshold', 'Desired'})

subplot(3,2,5);
imagesc(deltaThetaW_corr)
title('AVG \Deltaw(t) correlation')
colorbar('South');

subplot(3,2,4);
plot(thetaWnorm_avg)
title('AVG ||w(t)||')
xlabel('Iteration')


subplot(3,2,6);
plot(deltaThetaW_mag_avg)
title('AVG Average (effective) |\Deltaw(t)|')
xlabel('Iteration')


    