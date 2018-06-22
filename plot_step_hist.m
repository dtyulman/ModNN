figure;
for i = 1:4
   subplot(2,4,i)
   idx = 400+i;
   histogram(deltaThetaW(:,idx),[-inf, -eps/100,eps/100, inf])
   sum(deltaThetaW(:,idx)==0)
   title(sprintf('iter=%d, f=%g', idx, fthres_eff(idx)))
   subplot(2,4,i+4)
   histogram(thetaW(:,idx))
end
    
figure;
plot(fthres_eff); hold on
plot(fthres)
plot(fraw)
plot(f*ones(size(fthres)), 'k--')
title('Fraction of nonzero dldW')
xlabel('Iteration')
ylabel('f')
legend({'Effective', 'Calculated', 'No threshold', 'Desired'})
