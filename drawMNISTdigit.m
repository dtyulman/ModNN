function drawMNISTdigit(x, y, titlestr)

xydim = sqrt(length(x));
if mod(xydim,1) ~= 0
    error('Not square image')
end

imshow(reshape(x,xydim,xydim)', 'InitialMagnification', 'fit')

if nargin == 3
    title(sprintf(titlestr,y))
elseif nargin == 2
    title(['Class: ' num2str(y)])
end