function Y = onehot2class(Yh)

Y = zeros(size(Yh,1), 1);
for i = 1:length(Y)
    Y(i) = find(Yh(i,:))-1;
end


