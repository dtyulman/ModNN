function [Xtr, Ytr, Xv, Yv, Xte, Yte] = loadMNIST(datapath, digits, Ntr, Nv, Nte)
%Inputs:
% digits - array of integers to load
% Ntr, Nv, Nte - number of samples of each digit to load for the 
%                training, validation, and test sets 

Xtr = []; 
Ytr = [];
Xv = []; 
Yv = [];
Xte  = []; 
Yte  = [];

% TODO: make for-loop assign to correct slots here
% Npix = 26*26; %number of pixels
% Ndigs = length(digits); %number of digits;
% 
% Xtr = zeros(Ndigs*Ntr, Npix); 
% Ytr = zeros(Ndigs*Ntr, Ndigs);
% Xv = zeros(Ndigs*Nv, Npix); 
% Yv = zeros(Ndigs*Nv, Ndigs);
% Xte = zeros(Ndigs*Nte, Npix); 
% Yte = zeros(Ndigs*Nte, Ndigs);

for d = digits
    data = importdata(strcat(datapath, 'mnist_digit_', num2str(d), '.csv'));
    
    Xtr = [Xtr; data(  1:Ntr,:)]; 
    Ytr = [Ytr; d*ones(Ntr,1)];
    
    Xv = [Xv; data(Ntr+1:Ntr+Nv,:)]; 
    Yv = [Yv; d*ones(Nv,1)];
    
    Xte  = [Xte ; data(Ntr+Nv+1:Ntr+Nv+Nte,:)]; 
    Yte  = [Yte ; d*ones(Nte,1)];
end

%normalize data
Xtr = 2.*Xtr./255 - 1;
Xv = 2.*Xv./255 - 1 ;
Xte  = 2.*Xte./255 - 1;

%convert to one-hot vector
Ytr = class2onehot(Ytr);
Yv = class2onehot(Yv);
Yte = class2onehot(Yte);

end


function Yh = class2onehot(Y)
% convert class Y = {0, ..., 9} to one-hot vector e.g. 2 -> [0, 0, 1, 0, ..., 0]
classes = unique(Y);
Yh = zeros(length(Y), length(classes));
for i = 1:length(classes)
    Yh(Y==classes(i),i) = 1;
end

end