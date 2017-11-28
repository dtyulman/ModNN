function fx = softmax(x)
    e_x = exp(x - max(x));
    fx = e_x/sum(e_x);
end